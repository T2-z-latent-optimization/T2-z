from ...core.perception import extract_state, terminal_value


def outcome(views, fps, configuration, rules):
    state = extract_state("multi_background", views, fps, configuration)
    return state, terminal_value(state, rules)
