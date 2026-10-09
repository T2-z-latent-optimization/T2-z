import numpy as np

from ...core.simulator import fastwam_action
from ...core.types import Candidate


def build_candidates(simulator):
    bank = []
    simulator.native_bank = {}
    for definition in simulator.physics["actions"]:
        index = int(definition["action_id"])
        native = simulator.collector.build_fixed_actions(simulator.demo, definition, int(simulator.physics["recorded_steps"]))
        simulator.native_bank[index] = native
        raw = np.stack([fastwam_action(row) for row in native])
        indices = np.minimum(simulator.chunk.indices, len(raw) - 1)
        bank.append(Candidate(index, float(index), dict(definition), raw[indices]))
    return bank
