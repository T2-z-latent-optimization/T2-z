"""Online, prediction-only action selection against the original simulators.

Heavy model and physics imports are deliberately confined to worker processes.
"""

TASKS = (
    "friction", "gravity", "mass_collision", "mass_balance",
    "mass_friction", "light_switch", "multi_background",
)
