from __future__ import annotations

import copy
import sys

import numpy as np

from ...core.simulator import NativeSimulator, fastwam_action, libero_observation
from .actions import build_candidates


class FrictionSimulator(NativeSimulator):
    def initialize(self):
        # The original vanilla collectors manage their own LIBERO import path.
        # Plus overrides this method before ANY vanilla imports have occurred.
        sys.path.insert(0, str(self.sources.repos / "LIBERO"))
        self.original = self.sources.load(self.config["source_script"], "online_original_event80")
        self.base = self.original.base
        self.physics = self.original.configure_current_dataset(self.base.load_config(self.original.CONFIG_PATH))
        self.sources.config(self.config["source_config"])
        self._create_environment()

    def _create_environment(self):
        mu = float(self.case.get("friction_mu", 0.05))
        definitions = self.physics["actions"]
        reference = int(self.case.get("reference_action_index", 0))
        definition = next(item for item in definitions if int(item["action_id"]) == reference)
        bddl = self.base.write_hidden_bddl(self.physics, bddl_dir=self.sources.work / "bddl", geometry_id="online_event80")
        case = self.base.build_fixed_case(
            self.physics, mu=mu, action_cfg=definition, case_id="online_event80",
            bddl_file=bddl, camera_resolution=int(self.config["camera_resolution"]),
        )
        case = self.decorate_case(case)
        self.env = self.base.LiberoPushBoxEnv(case, repo_root=self.sources.root, seed=self.seed)
        self.env.reset()
        self.base.preposition_fixed_start(self.env)
        self.env.step_count = 0
        self.env._last_scripted_action = np.zeros(7, dtype=np.float64)
        self.env._last_scripted_phase = None
        initial, _ = self.env.box_pose()
        self.initial_x = float(initial[0])
        self.contact_frame = None
        self.stop_frame = None
        self.forced_stop = False

    def decorate_case(self, case):
        return case

    def current(self):
        return libero_observation(self.env, self.base)

    def build_candidates(self):
        self.bank = build_candidates(self)
        if self.config.get("execution_mode", "fixed_sequence") == "fixed_sequence":
            for candidate in self.bank:
                candidate.conditioning = "fixed_sequence"

    def activate(self, candidate):
        super().activate(candidate)
        if self.frame_index >= 70:
            raise RuntimeError("Choose the push action before the original frame-70 push onset")
        updated = copy.copy(self.env.case)
        object.__setattr__(updated, "pusher_push_action_end", float(candidate.parameters["A"]))
        object.__setattr__(updated, "pusher_push_steps", int(candidate.parameters["push_steps"]))
        object.__setattr__(updated, "hai_action_id", candidate.index)
        object.__setattr__(updated, "hai_action_profile", self.base.profile_for_steps(int(candidate.parameters["push_steps"])).tolist())
        self.env.case = updated

    def step_native(self):
        fixed = self.selected is not None and self.config.get("execution_mode", "fixed_sequence") == "fixed_sequence"
        if fixed:
            index = min((self.frame_index - self.chunk.start) // self.chunk.stride, len(self.selected.model_actions) - 1)
            command = np.asarray(self.selected.model_actions[index], dtype=np.float64).copy()
            command[-1] = 1.0 - 2.0 * command[-1]
            _, _, _, info = self.env.step(command)
            info["push_box"]["phase"] = "fixed_sequence"
        elif self.forced_stop:
            _, _, _, info = self.env.step(self.original.brake_action(self.env.case))
            info["push_box"]["phase"] = "event_stop"
        else:
            _, _, _, info = self.env.step()
        row = dict(info["push_box"])
        box_x, vx = float(row["box_xyz"][0]), float(row["box_vxy"][0])
        if self.contact_frame is None and (box_x - self.initial_x > self.original.CONTACT_MOVE_M or abs(vx) > self.original.CONTACT_SPEED_MPS):
            self.contact_frame = self.frame_index
        if self.contact_frame is not None and not self.forced_stop and not fixed:
            elapsed = self.frame_index - self.contact_frame
            threshold = max(self.original.CONTACT_SPEED_MPS, self.original.TRIGGER_VX_RATIO * float(self.env.case.pusher_push_action_end))
            if elapsed >= self.original.HOLD_AFTER_CONTACT and (abs(vx) >= threshold or elapsed >= self.original.MAX_CONTACT_HOLD):
                self.forced_stop = True
                self.stop_frame = self.frame_index + 1
        native = np.asarray(row.pop("action"), dtype=np.float64)
        return {
            **row, "native_action": native, "model_action": fastwam_action(native),
            "contact_frame": self.contact_frame, "event_stop_frame": self.stop_frame,
        }
