import math
import os


def field_value(row, field):
    value = row
    for key in field.split('.'):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(f'Dataset row lacks {field!r}')
        value = value[key]
    return value


class ChunkSelector:
    def __init__(self, mode='eligible', eligible_field='trainable_chunk', weight_field='chunk_weight'):
        if mode not in ('eligible', 'weighted', 'uniform'):
            raise ValueError(f'Unknown chunk selection mode: {mode!r}')
        self.mode = mode
        self.eligible_field = eligible_field
        self.weight_field = weight_field

    @classmethod
    def from_options(cls, options):
        return cls(options.get('chunk_selection_mode', 'eligible'),
                   options.get('trainable_chunk_field', 'trainable_chunk'),
                   options.get('chunk_weight_field', 'chunk_weight'))

    @classmethod
    def from_environment(cls):
        return cls(os.environ.get('T2Z_CHUNK_SELECTION_MODE', 'eligible'),
                   os.environ.get('T2Z_TRAINABLE_CHUNK_FIELD', 'trainable_chunk'),
                   os.environ.get('T2Z_CHUNK_WEIGHT_FIELD', 'chunk_weight'))

    def weight(self, row):
        value = field_value(row, self.weight_field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'{self.weight_field!r} must be numeric')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{self.weight_field!r} must be finite and non-negative')
        return value

    def accepts(self, row):
        if self.mode == 'eligible':
            value = field_value(row, self.eligible_field)
            if not isinstance(value, bool):
                raise TypeError(f'{self.eligible_field!r} must be boolean')
            return value
        if self.mode == 'weighted':
            self.weight(row)
        return True

    def choose(self, rng, indices, rows):
        candidates = [index for index in indices if self.accepts(rows[index])]
        if not candidates:
            raise ValueError('Episode has no eligible chunks')
        if self.mode == 'weighted':
            weights = [self.weight(rows[index]) for index in candidates]
            total = math.fsum(weights)
            if not math.isfinite(total) or total <= 0:
                raise ValueError('Episode chunk weights must have a positive finite sum')
            return rng.choices(candidates, weights=weights, k=1)[0]
        return rng.choice(candidates)

    def validate_episodes(self, rows, environment_key, episode_key='episode_index'):
        if self.mode != 'weighted':
            return
        weights = {}
        for row in rows:
            key = (row[environment_key], row[episode_key])
            weights.setdefault(key, []).append(self.weight(row))
        for key, values in weights.items():
            total = math.fsum(values)
            if not math.isfinite(total) or total <= 0:
                raise ValueError(f'Environment/episode {key!r} has no positive finite chunk-weight sum')


def configure_chunk_selection(options):
    selector = ChunkSelector.from_options(options)
    os.environ['T2Z_CHUNK_SELECTION_MODE'] = selector.mode
    os.environ['T2Z_TRAINABLE_CHUNK_FIELD'] = selector.eligible_field
    os.environ['T2Z_CHUNK_WEIGHT_FIELD'] = selector.weight_field
    return selector
