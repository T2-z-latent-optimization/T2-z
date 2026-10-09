from __future__ import annotations

import sys

import numpy as np

from ...core.simulator import NativeSimulator, fastwam_action, libero_observation
from .actions import build_candidates


class CollisionSimulator(NativeSimulator):
    def load_physics(self):
        entry = self.sources.load(self.config["source_script"], "online_original_mass_noleak")
        self.physics = self.sources.config(self.config["source_config"])
        self.collector, self.demo, pose = entry.prepare_runtime(self.physics)
        reference = int(self.case.get("reference_action_index", 0))
        pose["object_center_z_m"] = float(self.physics["canonical_object_center_z_m"]) + float(self.physics["action_height_offsets_m"][reference])

    def initialize(self):
        sys.path.insert(0, str(self.sources.repos / "LIBERO"))
        self.load_physics()
        demo, collector = self.demo, self.collector
        bddl = demo.write_two_box_bddl(self.sources.work)
        self.env = collector.LiberoPushBoxEnv(demo.make_case(bddl_file=bddl), repo_root=self.sources.root, seed=self.seed)
        self.env.reset()
        target_mass = float(self.case.get("target_mass_kg", 0.25))
        demo.set_object_mass(self.env, demo.PROJECTILE_NAME, float(demo.PROJECTILE_MASS_KG))
        demo.set_object_mass(self.env, demo.TARGET_NAME, target_mass)
        demo.set_object_contact_properties(self.env, demo.PROJECTILE_NAME, rgba=(0.10, 0.35, 0.95, 1.0))
        demo.set_object_contact_properties(self.env, demo.TARGET_NAME, rgba=(0.95, 0.25, 0.05, 1.0))
        collector.set_zero_object_velocity(demo, self.env, demo.PROJECTILE_NAME)
        collector.set_zero_object_velocity(demo, self.env, demo.TARGET_NAME)
        collector.place_objects_at_exact_initial_poses(demo, self.env)
        self.env._last_obs = self.env._refresh_obs()
        self.projectile_geoms = demo.object_geom_ids(self.env, demo.PROJECTILE_NAME)
        self.target_geoms = demo.object_geom_ids(self.env, demo.TARGET_NAME)
        self.setup_steps = int(demo.establish_projectile_contact(self.env, self.projectile_geoms))
        if not demo.robot_projectile_contact(self.env, self.projectile_geoms):
            raise RuntimeError("Original projectile contact preparation failed")
        self.alignment = collector.place_objects_at_exact_initial_poses(demo, self.env)
        if abs(self.alignment["lateral_offset_m"]) > 1e-6:
            raise RuntimeError("Original two-box initial alignment is invalid")
        self.env._last_obs = self.env._refresh_obs()
        demo.set_controller_translation_scale(self.env, demo.controller_translation_limits(self.env), float(demo.CONTROLLER_TRANSLATION_SCALE))
        initial, _, _ = demo.object_state(self.env, demo.TARGET_NAME)
        self.initial_target_xyz = initial.copy()
        self.first_collision = None
        self.last_robot_contact = -1

    def current(self):
        return libero_observation(self.env, self.collector)

    def build_candidates(self):
        self.bank = build_candidates(self)

    def activate(self, candidate):
        super().activate(candidate)
        self.commands = self.native_bank[candidate.index]

    def step_native(self):
        if self.selected is None:
            raise RuntimeError("Collision query starts at native frame 0; no launch may precede selection")
        native = self.commands[self.frame_index]
        _, _, _, info = self.env.step(native)
        actual = np.asarray(info["push_box"]["action"], dtype=np.float64)
        if not np.array_equal(actual, native):
            raise RuntimeError("Original collision controller changed a fixed command")
        demo = self.demo
        projectile_xyz, _, projectile_vel = demo.object_state(self.env, demo.PROJECTILE_NAME)
        target_xyz, _, target_vel = demo.object_state(self.env, demo.TARGET_NAME)
        robot_projectile = bool(demo.robot_projectile_contact(self.env, self.projectile_geoms))
        robot_target = bool(demo.robot_projectile_contact(self.env, self.target_geoms))
        if robot_projectile:
            self.last_robot_contact = self.frame_index
        if self.first_collision is None and (np.linalg.norm(target_xyz[:2] - self.initial_target_xyz[:2]) > 1e-4 or np.linalg.norm(target_vel[:2]) > 5e-3):
            self.first_collision = self.frame_index
        return {
            "native_action": actual, "model_action": fastwam_action(actual),
            "projectile_xyz": projectile_xyz, "projectile_velocity": projectile_vel,
            "target_xyz": target_xyz, "target_velocity": target_vel,
            "robot_projectile_contact": robot_projectile, "robot_target_contact": robot_target,
            "block_block_contact": bool(demo.contact_between(self.env, self.projectile_geoms, self.target_geoms)),
            "first_collision_frame": self.first_collision,
            "clean_collision_so_far": not robot_target and (self.first_collision is None or self.first_collision > self.last_robot_contact),
        }
