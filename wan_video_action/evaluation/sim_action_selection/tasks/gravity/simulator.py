from __future__ import annotations

import numpy as np

from ...core.simulator import NativeSimulator
from .actions import build_candidates


class GravitySimulator(NativeSimulator):
    native_action_semantics = "launcher_speed_parameter_mps_not_robot_control"

    def initialize(self):
        self.original = self.sources.load(self.config["source_script"], "online_original_gravity")
        import mujoco

        self.mujoco = mujoco
        original = self.original
        gravity = float(self.case.get("gravity_mps2", 9.81))
        self.model = mujoco.MjModel.from_xml_string(original.demo._mjcf(gravity))
        self.data = mujoco.MjData(self.model)
        self.ids = original._scene_ids(self.model)
        self.dof = int(self.model.jnt_dofadr[self.ids["cube_joint"]])
        size = int(self.config["camera_resolution"])
        self.renderers = [mujoco.Renderer(self.model, height=size, width=size) for _ in range(2)]
        self.cameras = [original._camera_main(), original._camera_auxiliary()]
        self.speed = float(original.SPEED_MIN_MPS)
        self.launched = False
        self.contact_frame = None
        mujoco.mj_forward(self.model, self.data)

    def current(self):
        frames = []
        for renderer, camera in zip(self.renderers, self.cameras):
            renderer.update_scene(self.data, camera=camera)
            frames.append(renderer.render().copy())
        return {"views": np.stack(frames), "state": self.original._cube_state(self.model, self.data, self.ids)}

    def state_arrays(self):
        return float(self.data.time), self.data.qpos, self.data.qvel

    def build_candidates(self):
        self.bank = build_candidates(self)

    def activate(self, candidate):
        super().activate(candidate)
        if self.launched:
            raise RuntimeError("The launcher action must be selected before launch")
        self.speed = float(candidate.parameters["speed_mps"])

    def step_native(self):
        original = self.original
        target_time = (self.frame_index + 1) / float(original.FPS)
        while self.data.time + 0.5 * original.demo.SIM_TIMESTEP < target_time:
            if not self.launched and self.data.time >= original.PRE_ROLL_S - 0.5 * original.demo.SIM_TIMESTEP:
                self.data.qvel[self.dof] = self.speed
                self.launched = True
            # This is the original guided step, not a replacement integrator.
            self.mujoco.mj_step(self.model, self.data)
            if self.launched and self.contact_frame is None and original._has_contact(self.data, self.ids["cube_geom"], self.ids["table_geom"]):
                self.contact_frame = self.frame_index + 1
        raw = np.asarray([self.speed, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        return {"model_action": raw, "native_action": np.asarray([self.speed], dtype=np.float32), "launcher_speed_mps": self.speed, "first_table_contact_frame": self.contact_frame, "simulation_time": float(self.data.time)}

    def native_last_frame(self):
        if self.contact_frame is not None:
            return self.contact_frame + int(self.original.POST_CONTACT_FRAMES)
        return int(round((self.original.MAX_EPISODE_S + self.original.PRE_ROLL_S) * self.original.FPS))

    def episode_ends_on_observation(self):
        return self.frame_index >= self.native_last_frame()

    def close(self):
        try:
            renderers, self.renderers = getattr(self, "renderers", []), []
            for renderer in renderers:
                self.original.demo._close_renderer(renderer)
        finally:
            self.sources.close()
