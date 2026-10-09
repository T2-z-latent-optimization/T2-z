from t2z.chunks import ChunkSelector


class WindowIndexSelector:
    def __init__(self, rows, *, mode='uniform_episode_then_window', kind_field='sampling_kind',
                 preferred_kind='precise', preferred_probability=0.5):
        if mode != 'uniform_episode_then_window':
            raise ValueError(f'Unknown episode/window sampling mode: {mode!r}')
        self.rows = rows
        self.enabled = True
        self.selector = ChunkSelector.from_environment()

    def choose(self, rng, indices):
        return self.selector.choose(rng, indices, self.rows)
