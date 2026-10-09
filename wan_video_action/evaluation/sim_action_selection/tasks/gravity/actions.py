import numpy as np

from ...core.types import Candidate


def build_candidates(simulator):
    bank = []
    for index, speed in enumerate(np.linspace(simulator.original.SPEED_MIN_MPS, simulator.original.SPEED_MAX_MPS, simulator.original.SPEED_COUNT)):
        raw = np.zeros((simulator.chunk.length, 7), dtype=np.float32)
        raw[:, 0] = speed
        bank.append(Candidate(index, float(index), {"speed_mps": float(speed)}, raw))
    return bank
