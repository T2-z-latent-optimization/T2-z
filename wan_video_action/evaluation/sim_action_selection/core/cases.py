from __future__ import annotations

import copy


PARAMETERS = {
    "friction": {"friction_mu", "reference_action_index"},
    "multi_background": {"friction_mu", "background_index", "reference_action_index"},
    "mass_collision": {"target_mass_kg", "reference_action_index"},
    "mass_friction": {"pair_index", "target_mass_kg", "target_table_friction_mu", "reference_action_index"},
    "gravity": {"gravity_mps2"},
    "mass_balance": {"ratio_index", "reference_support_index", "beam_xy_m"},
    "light_switch": {"causal_class", "control_colors", "initial_lamp_on",
                     "button_positions_xy_m", "decision_button", "press_lead_steps"},
}


def prepare_cases(task, cases):
    allowed = PARAMETERS[task] | {"seed", "prediction_seed", "case_id", "support_actions"}
    result = []
    for values in cases:
        unexpected = set(values) - allowed
        if unexpected:
            raise ValueError(f"Cases accept only physical parameters and online controls: {sorted(unexpected)}")
        case = copy.deepcopy(values)
        if task == "light_switch":
            case["candidate_initializations"] = {
                str(index): {"decision_button": color}
                for index, color in enumerate(("red", "blue"))
            }
        result.append(case)
    return result
