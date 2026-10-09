import numpy as np

from ...core.types import Candidate


def targets_for_support(simulator, offset):
    configuration = simulator.physics
    trajectory = configuration["trajectory"]
    point = simulator.initial_beam_pos + simulator.beam_axis * float(offset)
    final = point - simulator.insertion_axis * float(trajectory["gripper_to_support_lever_m"])
    final[2] = float(configuration["nominal_beam_z_m"]) + float(trajectory["under_beam_rod_center_z_offset_m"])
    approach = final - simulator.insertion_axis * float(trajectory["staging_backoff_m"])
    approach[2] = final[2]
    lifted = final.copy()
    lifted[2] = float(configuration["nominal_beam_z_m"]) + float(trajectory["lifted_rod_center_z_offset_m"])
    return [center - simulator.rod_minus_eef for center in (approach, final, lifted, lifted)]


def build_candidates(simulator):
    bank = []
    for entry in simulator.ratio_plan:
        index = int(entry["support_bin_index"])
        offset = float(entry["sampled_support_offset_m"])
        targets = targets_for_support(simulator, offset)
        rows = []
        for frame in simulator.chunk.indices:
            phase = simulator.phase_index(int(frame))
            rows.append(np.r_[targets[phase], np.zeros(4)].astype(np.float32))
        bank.append(Candidate(index, float(offset), {"support_offset_m": offset}, np.stack(rows), "exact_absolute_targets_feedback_native_delta"))
    return bank
