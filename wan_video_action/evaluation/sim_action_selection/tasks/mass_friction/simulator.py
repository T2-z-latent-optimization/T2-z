from ..mass_collision.simulator import CollisionSimulator


class MassFrictionSimulator(CollisionSimulator):
    def load_physics(self):
        self.original = self.sources.load(self.config["source_script"], "online_original_mass_friction_balanced")
        self.physics = self.sources.config(self.config["source_config"])
        boundary = self.sources.load_path(self.original.BOUNDARY_COLLECTOR, "online_original_mass_friction_boundary")
        self.collector = boundary.load_module(boundary.BASE_COLLECTOR, "online_original_joint_collision")
        self.demo = self.collector.load_collision_demo()
        self.collector.configure_demo(self.demo, self.physics)
        implementation = self.physics["friction_implementation"]
        router = boundary.install_target_friction_router(
            self.demo,
            projectile_mu=float(self.physics["projectile_table_friction_mu"]),
            projectile_priority=int(implementation["projectile_geom_priority"]),
            target_priority=int(implementation["target_geom_priority"]),
            table_priority=int(implementation["table_geom_priority"]),
        )
        if "pair_index" in self.case:
            pairs = self.original.build_balanced_pairs(
                self.physics["target_masses_kg"], self.physics["target_table_friction_values"],
                seed=int(self.physics["sampling_policy"]["selection_seed"]),
            )
            pair = pairs[int(self.case["pair_index"])]
            self.case["target_mass_kg"] = pair["target_mass_kg"]
            self.case["target_table_friction_mu"] = pair["target_table_friction_mu"]
        router["target_mu"] = float(self.case.get("target_table_friction_mu", 0.05))
