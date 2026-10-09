from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
import sys

import numpy as np

from ...core.simulator import NativeSimulator
from ...core.types import Chunk
from .actions import build_candidates


class PrefixReached(Exception):
    pass


class PrefixSink:
    def __init__(self, simulator):
        self.simulator = simulator
        self.index = 0
        self.episode_buffer = {"episode_index": 0}

    def _get_image_file_path(self, *, episode_index, image_key, frame_index):
        # The original recorder can write its two scratch images as usual.
        # Reuse two filenames; do not create a temporary dataset or long video.
        return self.simulator.sources.work / (image_key.replace(".", "_") + ".jpg")

    def add_frame(self, frame, task=None, **kwargs):
        simulator = self.simulator
        positions, path_index, target_z = None, None, None
        for stack in inspect.stack(context=0):
            values = stack.frame.f_locals
            if stack.function == "follow_path" and "result" in values and "i" in values:
                positions = np.asarray(values["result"]["position"], dtype=np.float64)
                path_index = int(values["i"])
            if stack.function in ("move_to_pose_with_screw", "move_to_pose_with_RRT") and "pose" in values:
                target_z = float(values["pose"].p[2])
        if positions is not None and target_z is not None and target_z < 0.10:
            remaining = len(positions) - path_index - 1
            if remaining == simulator.press_lead_steps:
                if simulator.env.unwrapped.press_history:
                    raise RuntimeError("The template boundary must precede the first button press")
                simulator.paused_frame = {key: np.array(value, copy=True) for key, value in frame.items()}
                simulator.reference_color = simulator.decision_button
                simulator.remaining_path = positions[path_index + 1:].copy()
                simulator.pending_target_z = target_z
                transitions = simulator.chunk.end - simulator.chunk.start
                simulator.chunk = Chunk(self.index, self.index + transitions, 1, "post_step")
                simulator.case["decision_phase"] = "button_descent"
                simulator.case["planned_descent_steps_remaining"] = remaining
                raise PrefixReached
        if simulator.config.get("capture_history", False):
            simulator._capture_prefix_views.append(np.stack([
                frame["observation.images.image"], frame["observation.images.wrist_image"],
            ]))
        self.index += 1

    def save_episode(self):
        raise RuntimeError("The motion template did not reach its decision boundary")


class LightSwitchSimulator(NativeSimulator):
    def initialize(self):
        self.decision_button = str(self.case.get("decision_button", "blue"))
        self.press_lead_steps = int(self.case.get("press_lead_steps", self.config.get("press_lead_steps", 23)))
        if self.decision_button not in ("red", "blue") or self.press_lead_steps < 1:
            raise ValueError("Choose a valid button and a positive planned descent lead")
        repository = self.sources.repos / "robomme_benchmark"
        sys.path.insert(0, str(repository / "src"))
        sys.path.insert(0, str(repository / "scripts"))
        self.original = self.sources.load_path(repository / self.config["source_script"], "online_original_independent_lightswitch")
        import gymnasium as gym

        self.env = gym.make(
            "CausalLightSwitchDIYHaiMachine", obs_mode="rgb+depth+segmentation",
            control_mode="pd_joint_pos", render_mode="rgb_array", reward_mode="dense",
        )
        original = self.original
        self.planner = original.base.FailAwarePandaArmMotionPlanningSolver(
            self.env, debug=False, vis=False,
            base_pose=self.env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False, print_env_info=False,
        )
        original.base._patch_planner(self.planner)
        self.transform = original.AbsoluteEEFActionTransform(self.env, self.planner)
        self.remaining_path = None
        self.pending_target_z = None
        self.reference_color = None
        self._capture_prefix_views = []
        positions = self.case.get("button_positions_xy_m", {"red": list(original.FIXED_CLOSE_RED_XY), "blue": list(original.FIXED_CLOSE_BLUE_XY)})
        configurations = {name: list(colors) for name, colors in original.CAUSAL_CONFIGURATIONS}
        controls = self.case.get("control_colors", configurations[self.case.get("causal_class", "red_only")])
        self.env.unwrapped.configure_episode(
            red_button_xy=positions["red"], blue_button_xy=positions["blue"],
            control_button_colors=controls, initial_lamp_on=bool(self.case.get("initial_lamp_on", False)),
        )
        observation, _ = self.env.reset(seed=self.seed)
        sink = PrefixSink(self)
        recorder = original.base.EpisodeRecorder(sink, self.env, 0, action_transform=self.transform)
        native_step = self.env.step
        native_close_gripper = self.planner.close_gripper
        self.planner.gripper_state = self.planner.CLOSED

        def set_closed(*args, gripper_state=None, **kwargs):
            self.planner.gripper_state = self.planner.CLOSED if gripper_state is None else gripper_state

        def observed_step(action):
            result = native_step(action)
            recorder.add(result[0], action)
            return result

        self.env.step = observed_step
        self.planner.close_gripper = set_closed
        try:
            recorder.add(observation, original.base._hold_action(self.env, gripper=-1.0))
            recorder.set_phase("press_" + self.decision_button)
            result = original.base.solve_button(self.env, self.planner, self.env.unwrapped.buttons[self.decision_button])
            if isinstance(result, int) and result == -1:
                raise RuntimeError("Button motion-template planning failed")
        except PrefixReached:
            pass
        finally:
            self.env.step = native_step
            self.planner.close_gripper = native_close_gripper
        if not hasattr(self, "paused_frame"):
            raise RuntimeError("Button motion template did not reach a prepress decision boundary")
        self.frame_index = self.chunk.start

    def captured_prefix_views(self):
        return self._capture_prefix_views

    def current(self):
        frame = self.paused_frame
        return {
            "views": np.stack([frame["observation.images.image"], frame["observation.images.wrist_image"]]),
            "state": frame["observation.state"],
        }

    def state_arrays(self):
        base = self.original.base
        elapsed = base._vec(self.env.unwrapped.elapsed_steps)[0]
        return float(elapsed), base._vec(self.env.unwrapped.agent.robot.get_qpos()), base._vec(self.env.unwrapped.agent.robot.get_qvel())

    def fingerprint(self):
        digest = hashlib.sha256(super().fingerprint().encode())
        digest.update(self.original.base._button_lamp_state(self.env).tobytes())
        return digest.hexdigest()

    def build_candidates(self):
        self.bank = build_candidates(self)

    def execute(self, action_index):
        if self.frame_index != self.chunk.start:
            raise RuntimeError("Light Switch is no longer at its recorded post-step start")
        candidate = next(item for item in self.bank if item.index == action_index)
        commands = self.native_bank[action_index]
        frames = [self.current()["views"]]
        actions = [self.paused_frame["action"].copy()]
        diagnostics = []
        transform = self.original.AbsoluteEEFActionTransform(self.env, self.planner)
        previous = self.transform.previous_euler_xyz
        transform.previous_euler_xyz = None if previous is None else previous.copy()
        events_before = len(self.env.unwrapped.press_history)
        for local, command in enumerate(commands, start=1):
            obs, _, _, _, _ = self.env.step(command)
            self.frame_index += 1
            frames.append(np.stack([self.original.base._rgb(obs, "base_camera"), self.original.base._rgb(obs, "hand_camera")]))
            actions.append(transform(command))
            diagnostics.append({
                "frame_index": self.frame_index, "native_joint_action": command,
                "model_action": actions[-1],
                "button_lamp_state": self.original.base._button_lamp_state(self.env),
            })
        actual = np.stack(actions)
        return {
            "views": np.stack(frames), "actions": actual,
            "planned_actions": candidate.model_actions, "native_actions": commands,
            "native_control_frame_indices": self.chunk.indices[1:],
            "native_action_semantics": "original_pd_joint_pos_seven_joints_plus_gripper",
            "recorded_actions": actual, "recorded_frame_indices": self.chunk.indices,
            "native_frame_indices": self.chunk.indices, "source_frame_indices": self.chunk.indices,
            "padded": np.zeros(self.chunk.length, dtype=bool),
            "native_control_iterations": len(commands),
            "native_log": diagnostics, "conditioning": candidate.conditioning,
            "conditioning_max_abs_error": float(np.max(np.abs(actual - candidate.model_actions))),
            "press_events": list(self.env.unwrapped.press_history[events_before:]),
            "actual_control_timestep": float(self.env.unwrapped.control_timestep),
            "final_fingerprint": self.fingerprint(),
        }
