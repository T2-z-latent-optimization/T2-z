import json
import random
from wan_video_action.methods.baselines.environment_sampling import sample_slots
from wan_video_action.methods.baselines.window_sampling import WindowIndexSelector
from wan_video_action.methods.baselines.dinov2_amortized import EnvironmentEpisodeSampler, SupportQueryEpisodeIndices
from wan_video_action.methods.baselines.ttt_kvb.sampling import EnvironmentStreamSampler, StreamEpisode


class EpisodeBatchSampler:

    def __init__(self, *, metadata_path, seed, environments_per_rank, episodes_per_environment, window_kind_field='sampling_kind', preferred_window_kind='precise', preferred_window_probability=0.5):
        with open(metadata_path, encoding='utf-8') as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]
        self.groups = {}
        for index, row in enumerate(self.rows):
            if row.get('dataset_split') != 'train':
                raise ValueError(f'Non-training row in pooled manifest: {index}')
            env = int(row['environment_index'])
            episode = int(row['episode_index'])
            self.groups.setdefault(env, {}).setdefault(episode, []).append(index)
        self.envs_per_rank = int(environments_per_rank)
        self.episodes_per_env = int(episodes_per_environment)
        self.windows = WindowIndexSelector(self.rows, mode='uniform_episode_then_window', kind_field=window_kind_field, preferred_kind=preferred_window_kind, preferred_probability=preferred_window_probability)
        self.seed = int(seed)

    def sample(self, step, rank):
        rng = random.Random(self.seed + int(step) * 104729 + int(rank) * 1000003)
        result = []
        for env in sample_slots(rng, sorted(self.groups), self.envs_per_rank):
            episodes = self.groups[env]
            for episode in sample_slots(rng, sorted(episodes), self.episodes_per_env):
                result.append(self.windows.choose(rng, episodes[episode]))
        return result


class IndependentStreamSampler:

    def __init__(self, *, metadata_path, seed, sequence_length, streams_per_environment=1, environments_per_rank=1, **kwargs):
        self.seed = int(seed)
        self.sequence_length = int(sequence_length)
        self.streams_per_environment = int(streams_per_environment)
        self.pool = EpisodeBatchSampler(metadata_path=metadata_path, seed=seed, environments_per_rank=environments_per_rank, episodes_per_environment=self.sequence_length * self.streams_per_environment, window_kind_field=kwargs['window_kind_field'], preferred_window_kind=kwargs['preferred_window_kind'], preferred_window_probability=kwargs['preferred_window_probability'])

    def sample(self, *, step, process_index, num_processes, environments_per_rank):
        if int(num_processes) < 1 or not 0 <= int(process_index) < int(num_processes):
            raise ValueError('Invalid distributed rank configuration')
        self.pool.envs_per_rank = int(environments_per_rank)
        selected = self.pool.sample(step, process_index)
        groups = {}
        for index in selected:
            env = int(self.pool.rows[index]['environment_index'])
            groups.setdefault(env, []).append(index)
        rng = random.Random(self.seed + step * 15485863 + process_index * 32452843)
        streams = []
        for env, indices in groups.items():
            rng.shuffle(indices)
            for stream_index in range(len(indices) // self.sequence_length):
                start = stream_index * self.sequence_length
                stream_indices = indices[start:start + self.sequence_length]
                if self.streams_per_environment > 1:
                    rng.shuffle(stream_indices)
                streams.append(StreamEpisode(environment_id=env, indices=tuple(stream_indices), action_ids=tuple((int(self.pool.rows[i]['episode_index']) for i in stream_indices))))
        return tuple(streams)


class IndependentSupportSampler:
    def __init__(self, args):
        self.seed = int(args.seed)
        self.pool = episode_sampler(args, 'dinov2', 'trajectories_per_environment')
        values = str(args.dinov2_support_k_choices or args.dinov2_support_k)
        self.choices = tuple(int(value.strip()) for value in values.split(',') if value.strip())
        if not self.choices or min(self.choices) < 1 or max(self.choices) > self.pool.episodes_per_env:
            raise ValueError('Support counts must fit within the sampled environment batch')

    def sample(self, *, step, process_index, num_processes, environments_per_rank):
        self.pool.envs_per_rank = int(environments_per_rank)
        selected = self.pool.sample(step, process_index)
        groups = {}
        for index in selected:
            groups.setdefault(int(self.pool.rows[index]['environment_index']), []).append(index)
        rng = random.Random(self.seed + step * 15485863 + process_index * 32452843)
        episodes = []
        for env, indices in groups.items():
            supports = rng.sample(indices, rng.choice(self.choices))
            episodes.append(SupportQueryEpisodeIndices(environment_id=env, support_index=supports[0], additional_support_indices=tuple(supports[1:]), query_indices=tuple(indices)))
        return tuple(episodes)


def episode_sampler(args, prefix, count_key='episodes_per_environment'):
    return EpisodeBatchSampler(
        metadata_path=args.dataset_metadata_path, seed=args.seed,
        environments_per_rank=getattr(args, prefix + '_environments_per_rank'),
        episodes_per_environment=getattr(args, prefix + '_' + count_key),
        window_kind_field=getattr(args, prefix + '_window_kind_field', 'sampling_kind'),
        preferred_window_kind=getattr(args, prefix + '_preferred_window_kind', 'precise'),
        preferred_window_probability=getattr(args, prefix + '_preferred_window_probability', 0.5),
    )


def build_dino_sampler(args, support_k_choices):
    if args.environment_sampling_scope == 'independent_rank':
        return IndependentSupportSampler(args)
    return EnvironmentEpisodeSampler(
        metadata_path=args.dataset_metadata_path, active_environment_manifest=args.dinov2_active_environment_manifest,
        seed=args.seed, support_k=args.dinov2_support_k, support_k_choices=support_k_choices,
        trajectories_per_environment=args.dinov2_trajectories_per_environment,
        environment_key=args.dinov2_environment_key, action_key=args.dinov2_action_key,
        distinct_actions=args.dinov2_distinct_actions, window_sampling_mode=args.dinov2_window_sampling_mode,
        window_kind_field=args.dinov2_window_kind_field, preferred_window_kind=args.dinov2_preferred_window_kind,
        preferred_window_probability=args.dinov2_preferred_window_probability,
    )


def build_ttt_sampler(args):
    settings = dict(
        metadata_path=args.dataset_metadata_path, active_environment_manifest=args.ttt_active_environment_manifest,
        seed=args.seed, sequence_length=args.ttt_sequence_length,
        environment_key=args.ttt_environment_key, action_key=args.ttt_action_key,
        distinct_actions=args.ttt_distinct_actions, window_sampling_mode=args.ttt_window_sampling_mode,
        window_kind_field=args.ttt_window_kind_field, preferred_window_kind=args.ttt_preferred_window_kind,
        preferred_window_probability=args.ttt_preferred_window_probability,
    )
    if args.environment_sampling_scope == 'independent_rank':
        return IndependentStreamSampler(**settings, streams_per_environment=args.ttt_streams_per_environment, environments_per_rank=args.ttt_environments_per_rank)
    return EnvironmentStreamSampler(**settings)
