from __future__ import annotations

import sys

import numpy as np

from ...core.simulator import NativeSimulator, libero_observation
from .actions import build_candidates, targets_for_support


class BalanceSimulator(NativeSimulator):
    def initialize(self):
        sys.path.insert(0, str(self.sources.repos / "LIBERO"))
        self.original = self.sources.load(self.config["source_script"], "online_original_workspace_balance")
        self.physics = self.sources.config(self.config["source_config"])
        original, config = self.original, self.physics
        if config["variant"] != "workspace_random_combined_train20_test10":
            raise ValueError("Only the formal workspace-random balance configuration is supported")
        ratio_index = int(self.case.get("ratio_index", 9))
        ratio = float(config["right_to_left_mass_ratios"][ratio_index])
        spec = original.mass_spec(config, ratio_index, ratio)
        xml = self.sources.work / "assets" / (spec["category"] + ".xml")
        original.write_ratio_xml(xml, spec)
        from libero.libero.envs.base_object import OBJECTS_DICT
        for key, asset_type in tuple(OBJECTS_DICT.items()):
            if asset_type.__name__ == str(spec["class_name"]):
                del OBJECTS_DICT[key]
        original.register_ratio_beam(spec, xml)
        bddl = self.sources.work / "bddl" / (spec["category"] + ".bddl")
        original.write_bddl(bddl, spec)
        self.beam_name = str(spec["beam_name"])
        self.ratio_plan = [entry for entry in original.create_episode_plan(config) if int(entry["ratio_index"]) == ratio_index]
        reference_index = int(self.case.get("reference_support_index", 7))
        reference = next(entry for entry in self.ratio_plan if int(entry["support_bin_index"]) == reference_index)
        beam_xy = np.asarray(config["beam_base_xy_m"], dtype=np.float64) + [reference["beam_front_back_x_offset_m"], reference["beam_left_right_y_offset_m"]]
        if "beam_xy_m" in self.case:
            beam_xy = np.asarray(self.case["beam_xy_m"], dtype=np.float64)
        native_case = original.LiberoPushBoxCase(
            case_id=f"online_workspace_balance_{ratio_index}",
            friction_mu=float(config["fixed_friction_mu"]), bddl_file=str(bddl),
            box_name=self.beam_name, max_steps=240,
            camera_resolution=int(config["camera_resolution"]), control_freq=float(config["fps"]),
        )
        self.env, _ = original.create_env_with_sampler_retries(native_case, seed=int(config["support_seed"]) + ratio_index)
        _, initialization = original.initialize_pregripped_episode(
            self.env, beam_name=self.beam_name, beam_xy=beam_xy,
            settle_steps=int(config["pregrip_settle_steps"]),
        )
        self.initial_beam_pos, beam_quat, _ = original.object_state(self.env, self.beam_name)
        rod_pos, rod_quat, _ = original.object_state(self.env, original.ROD_NAME)
        self.beam_axis = original.quat_rotation_matrix(beam_quat)[:, 0]
        rod_axis = original.quat_rotation_matrix(rod_quat)[:, 0]
        self.insertion_axis = rod_axis if np.dot(rod_axis, self.initial_beam_pos - rod_pos) >= 0 else -rod_axis
        self.rod_minus_eef = np.asarray(initialization["rod_minus_eef_m"], dtype=np.float64)
        self.rod_minus_eef[2] = float(config["nominal_rod_minus_eef_z_m"])
        self.targets = targets_for_support(self, float(reference["sampled_support_offset_m"]))
        trajectory = config["trajectory"]
        self.phase_lengths = [int(trajectory[key + "_steps"]) for key in ("approach", "insert", "lift", "hold")]
        self.phase_ends = np.cumsum(self.phase_lengths)
        self.phase_caps = [float(trajectory[key + "_max_delta_action"]) for key in ("approach", "insert", "lift", "hold")]
        self.case.update({"right_to_left_mass_ratio": ratio, "beam_xy_m": beam_xy.tolist(), "initialization": initialization})
        self.gripper_ids = original.gripper_geom_ids(self.env)
        self.rod_ids = original.geom_ids(self.env, original.ROD_NAME)
        self.beam_ids = original.geom_ids(self.env, self.beam_name)

    def phase_index(self, frame):
        return min(int(np.searchsorted(self.phase_ends, frame, side="right")), 3)

    def current(self):
        return libero_observation(self.env, self.original)

    def build_candidates(self):
        self.bank = build_candidates(self)

    def activate(self, candidate):
        super().activate(candidate)
        self.targets = targets_for_support(self, candidate.parameters["support_offset_m"])

    def step_native(self):
        phase = self.phase_index(self.frame_index)
        absolute = self.targets[phase]
        obs = self.env._last_obs
        native = np.zeros(7, dtype=np.float64)
        native[:3] = np.clip(4.0 * (absolute - np.asarray(obs["robot0_eef_pos"])), -self.phase_caps[phase], self.phase_caps[phase])
        native[-1] = 1.0
        original = self.original
        beam_pos, beam_quat, _ = original.object_state(self.env, self.beam_name)
        diagnostic = {
            "phase": ("direct_approach_to_beam_gap", "insert_under_beam", "lift_at_sampled_support", "hold_balance_result")[phase],
            "native_action": native.copy(), "model_action": np.r_[absolute, np.zeros(4)].astype(np.float32),
            "beam_tilt_deg": float(original.beam_tilt_degrees(beam_quat)),
            "beam_lift_m": float(beam_pos[2] - self.initial_beam_pos[2]),
            "grasp": bool(original.robosuite_grasping(self.env)),
            "rod_beam_contact": bool(original.has_contact(self.env, self.rod_ids, self.beam_ids)),
            "direct_gripper_beam_contact": bool(original.has_contact(self.env, self.gripper_ids, self.beam_ids)),
        }
        self.env.step(native)
        return diagnostic
