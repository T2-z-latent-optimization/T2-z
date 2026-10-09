import numpy as np

from ..friction.simulator import FrictionSimulator


class MultiBackgroundSimulator(FrictionSimulator):
    def initialize(self):
        self.sources.pin_libero(plus=True)
        self.sources.disable_optional_wand()
        self.matched = self.sources.load(self.config["source_script"], "online_original_matched_plus")
        self.experiment = self.sources.config(self.config["source_config"])
        if len(self.experiment["environments"]) != 5:
            raise ValueError("Formal Multi Background requires the fixed five-background source configuration")
        index = int(self.case.get("background_index", 0))
        environment = self.experiment["environments"][index]
        self.case["environment_id"] = environment["environment_id"]
        presets = {item["preset_id"]: item for item in self.experiment["background_randomization"]["presets"]}
        self.visual_setup = {
            "environment": environment,
            "background_preset": presets[environment["background_preset_id"]],
            "experiment": self.experiment, "seed": self.seed,
        }
        self.bind_visual_setup()
        self.visual_environment = environment
        self.original = self.matched.segmented
        self.base = self.original.base
        self.physics = self.original.configure_current_dataset(self.base.load_config(self.original.CONFIG_PATH))
        self._create_environment()
        self.stable_count = 0
        self.previous_views = None
        self.previous_eef = None
        self.adaptive_last = None

    def bind_visual_setup(self):
        self.matched._ACTIVE_SETUP.clear()
        self.matched._ACTIVE_SETUP.update(self.visual_setup)
        self.matched.official.legacy._NATIVE_PRESETS = [self.visual_setup["background_preset"]]

    def current(self):
        self.bind_visual_setup()
        return super().current()

    def decorate_case(self, case):
        return self.matched.build_visual_overlay_case(case, environment=self.visual_environment)

    def step_native(self):
        self.bind_visual_setup()
        if self.selected is not None and self.config.get("execution_mode", "fixed_sequence") == "fixed_sequence":
            return super().step_native()
        before = self.current()["views"]
        row = super().step_native()
        after_eef = np.asarray(row["eef_xyz"], dtype=np.float64)
        settings = self.experiment["adaptive_episode_end"]
        if self.previous_views is not None and self.previous_eef is not None:
            differences = np.mean(np.abs(before.astype(np.float32) - self.previous_views.astype(np.float32)), axis=(1, 2, 3))
            speed = float(np.linalg.norm(after_eef - self.previous_eef) * self.fps)
            stable = (
                self.stop_frame is not None
                and self.frame_index - self.stop_frame >= int(settings["minimum_frames_after_event_stop"])
                and bool(np.all(differences <= float(settings["visual_mean_absolute_difference_threshold"])))
                and speed <= float(settings["eef_speed_threshold_mps"])
            )
            self.stable_count = self.stable_count + 1 if stable else 0
            if self.stable_count >= int(settings["consecutive_stable_frames"]):
                self.adaptive_last = self.frame_index
            row["adaptive_stable_frames"] = self.stable_count
        self.previous_views = before.copy()
        self.previous_eef = after_eef.copy()
        return row

    def native_last_frame(self):
        if self.selected is not None and self.config.get("execution_mode", "fixed_sequence") == "fixed_sequence":
            return max(int(self.env.case.max_steps) - 1, int(self.chunk.end))
        return int(self.adaptive_last) if self.adaptive_last is not None else int(self.env.case.max_steps) - 1
