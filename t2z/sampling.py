import os
import random
from t2z.chunks import ChunkSelector


def _eligible(row, field):
    value = row
    for key in field.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f'Dataset row lacks {field!r}')
        value = value[key]
    if not isinstance(value, bool):
        raise TypeError(f'Dataset field {field!r} must be boolean')
    return value


def sample_update(**kwargs):
    groups = kwargs['grouped_indices']
    rows = kwargs['rows']
    allowed = kwargs.get('allowed_friction_values')
    allowed = None if allowed is None else [float(value) for value in allowed]
    args = kwargs['args']
    selector = ChunkSelector.from_environment()
    episode_key = str(getattr(args, 'grouped_context_episode_key', 'episode_index'))
    pools = {}
    for env, action_groups in groups.items():
        if allowed is not None and not any(abs(float(env) - value) <= 1e-5 for value in allowed):
            continue
        episodes = {}
        for indices in action_groups.values():
            for index in indices:
                if selector.accepts(rows[index]):
                    episode = rows[index][episode_key]
                    episodes.setdefault(episode, []).append(index)
        if episodes:
            pools[env] = episodes
    if not pools:
        raise ValueError('No eligible environments in the current curriculum phase')
    episode_count = int(kwargs['actions_per_update'])
    microbatches = int(kwargs.get('microbatches_per_update', int(kwargs['friction_groups']) * episode_count))
    if episode_count < 1 or microbatches < 1:
        raise ValueError('Episode and microbatch counts must be positive')
    env_slots = (microbatches + episode_count - 1) // episode_count
    seed = int(getattr(args, 'seed', 0) or 0)
    step = int(kwargs.get('update_idx', 0))
    rank = int(os.environ.get('RANK', '0'))
    rng = random.Random(seed + 1000003 * step + 1009 * rank)
    values = list(pools)
    envs = rng.sample(values, env_slots) if len(values) >= env_slots else rng.choices(values, k=env_slots)
    numbers = [rng.getrandbits(63) for _ in range(episode_count)]
    result = []
    for env in envs:
        episodes = pools[env]
        names = sorted(episodes, key=int)
        for number in numbers:
            result.append(selector.choose(rng, episodes[names[number % len(names)]], rows))
    return result[:microbatches]
