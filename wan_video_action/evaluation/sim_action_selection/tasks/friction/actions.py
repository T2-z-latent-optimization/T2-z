from __future__ import annotations

import copy

import numpy as np

from ...core.simulator import fastwam_action
from ...core.types import Candidate


def build_candidates(simulator):
    candidates = []
    for definition in simulator.physics["actions"]:
        case = copy.copy(simulator.env.case)
        object.__setattr__(case, "pusher_push_action_end", float(definition["A"]))
        object.__setattr__(case, "pusher_push_steps", int(definition["push_steps"]))
        object.__setattr__(case, "hai_action_id", int(definition["action_id"]))
        object.__setattr__(case, "hai_action_profile", simulator.base.profile_for_steps(int(definition["push_steps"])).tolist())
        shadow = copy.copy(simulator.env)
        shadow.case = case
        shadow._last_scripted_action = np.array(simulator.env._last_scripted_action, copy=True)
        commands = []
        for frame in range(simulator.chunk.start, simulator.chunk.end + 1):
            shadow.step_count = frame
            if frame >= 70 + int(definition["push_steps"]):
                native = simulator.original.brake_action(case)
            else:
                native = shadow._scripted_gripper_push_action(shadow._scripted_phase())
            commands.append(fastwam_action(native))
        candidates.append(Candidate(
            int(definition["action_id"]), float(definition["action_id"]), dict(definition),
            np.stack(commands)[::simulator.chunk.stride],
            conditioning="nominal_frozen_feedback_event_stop_unknown",
        ))
    return candidates
