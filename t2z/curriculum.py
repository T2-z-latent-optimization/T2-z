def context_model_cycle(active_indices, context_steps, model_steps):
    return (
        ('all_context', context_steps, active_indices, active_indices),
        ('model', model_steps, active_indices, []),
    )


def activation_interval(args):
    interval = int(getattr(args, 'grouped_context_curriculum_activation_interval', 2))
    if interval < 1:
        raise ValueError('Activation interval must be a positive number of C/model cycles')
    return interval


def activation_period_steps(args):
    cycle_steps = (
        int(args.grouped_context_curriculum_all_context_steps)
        + int(args.grouped_context_curriculum_model_steps)
    )
    return int(args.grouped_context_curriculum_new_context_steps) + activation_interval(args) * cycle_steps


def phase_for_step(args, group_order, step):
    total_groups = int(args.grouped_context_curriculum_total_groups) or len(group_order)
    total_groups = min(total_groups, len(group_order))
    initial_groups = min(int(args.grouped_context_curriculum_initial_groups), total_groups)
    add_groups = int(args.grouped_context_curriculum_add_groups)
    if initial_groups <= 0 or add_groups <= 0:
        raise ValueError('Curriculum requires positive initial and added group counts')
    initial_steps = int(args.grouped_context_curriculum_initial_model_steps)
    new_steps = int(args.grouped_context_curriculum_new_context_steps)
    context_steps = int(args.grouped_context_curriculum_all_context_steps)
    model_steps = int(args.grouped_context_curriculum_model_steps)
    interval = activation_interval(args)
    if initial_steps < 0 or min(new_steps, context_steps, model_steps) <= 0:
        raise ValueError('Curriculum phase durations must be positive')
    step = int(step)
    if step < 1:
        raise ValueError('Curriculum steps are one-based')

    def state(round_id, phase, samples, trainable, start, end, active_count, new_count):
        return {
            'round': round_id, 'phase': phase,
            'sample_group_indices': samples, 'train_context_indices': trainable,
            'phase_start': start, 'phase_end': end,
            'active_count': active_count, 'new_count': new_count,
        }

    if step <= initial_steps:
        active = group_order[:initial_groups]
        return state(0, 'model', active, [], 1, initial_steps, len(active), len(active))
    offset = initial_steps
    selected_count = initial_groups
    active_indices = group_order[:selected_count]
    new_indices = active_indices
    round_id = 1
    completed_cycles = 0
    while True:
        if completed_cycles == 0:
            start, end = offset + 1, offset + new_steps
            if start <= step <= end:
                return state(round_id, 'new_context', new_indices, new_indices, start, end, len(active_indices), len(new_indices))
            offset = end
        for phase, duration, samples, trainable in context_model_cycle(
            active_indices, context_steps, model_steps
        ):
            start, end = offset + 1, offset + duration
            if start <= step <= end:
                return state(round_id, phase, samples, trainable, start, end, len(active_indices), len(new_indices))
            offset = end
        completed_cycles += 1
        if completed_cycles < interval:
            continue
        if selected_count == total_groups:
            break
        new_end = min(selected_count + add_groups, total_groups)
        new_indices = group_order[selected_count:new_end]
        selected_count = new_end
        active_indices = group_order[:selected_count]
        completed_cycles = 0
        round_id += 1
    active = group_order[:total_groups]
    post_context_steps = int(getattr(args, 'grouped_context_post_curriculum_cycle_steps', 0) or 0)
    if post_context_steps > 0:
        duration = post_context_steps + model_steps
        cycle_index, local_step = divmod(step - offset - 1, duration)
        start = offset + cycle_index * duration + 1
        if local_step < post_context_steps:
            return state(round_id + 1 + cycle_index, 'all_context', active, active, start, start + post_context_steps - 1, len(active), 0)
        start += post_context_steps
        return state(round_id + 1 + cycle_index, 'model', active, [], start, start + model_steps - 1, len(active), 0)
    return state(round_id + 1, 'model', active, [], offset + 1, step, len(active), 0)
