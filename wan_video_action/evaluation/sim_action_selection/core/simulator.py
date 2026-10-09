from __future__ import annotations
import hashlib
import json
import numpy as np
from .sources import Sources
from .runtime_paths import configure_libero_paths
from .types import Chunk

class NativeSimulator:
    """A recorded frame is not an additional control step.

    Subclasses reuse the source initialization and implement one original native
    iteration. Only this base class samples observations. It never scales time.
    """

    def __init__(self, config: dict, case: dict, source_root: str):
        (self.config, self.case) = (config, dict(case))
        spec = config['chunk']
        self.chunk = Chunk(**spec)
        self.sources = Sources(source_root)
        if config.get('task') != 'light_switch':
            configure_libero_paths(self.sources, plus=config.get('task') == 'multi_background')
        self.frame_index = 0
        self.seed = int(case.get('seed', 0))
        self.bank = []
        self.selected = None
        self.env = None
        self.fps = float(config['fps'])

    def initialize(self):
        raise NotImplementedError

    def current(self) -> dict:
        raise NotImplementedError

    def step_native(self) -> dict:
        raise NotImplementedError

    def activate(self, candidate):
        self.selected = candidate

    def candidates(self):
        return [candidate.to_wire() for candidate in self.bank]

    def set_candidate_actions(self, reference):
        if self.selected is not None or self.frame_index != self.chunk.start:
            raise RuntimeError('Reference commands can only be installed at the untouched decision boundary')
        indices = np.asarray(reference['action_indices'])
        actions = np.asarray(reference['model_actions'], dtype=np.float32)
        frames = np.asarray(reference['native_frame_indices'])
        if indices.ndim != 1 or len(set(indices.tolist())) != len(indices):
            raise ValueError('Reference action IDs must be a unique vector')
        if set(indices.tolist()) != {candidate.index for candidate in self.bank}:
            raise ValueError('Reference commands must cover the complete original action bank')
        if not np.array_equal(frames, self.chunk.indices):
            raise ValueError('Reference commands use a different native frame window')
        if actions.ndim != 3 or len(actions) != len(indices) or (not np.isfinite(actions).all()):
            raise ValueError('Reference commands must be a finite candidate/time/channel array')
        replacements = {int(index): values for (index, values) in zip(indices, actions)}
        for candidate in self.bank:
            if replacements[candidate.index].shape != candidate.model_actions.shape:
                raise ValueError('Reference command shape differs from the original action representation')
        for candidate in self.bank:
            candidate.model_actions = replacements[candidate.index].copy()
        return self.candidates()

    def native_last_frame(self) -> int:
        return int(self.config.get('native_last_frame', self.chunk.end))

    def episode_ends_on_observation(self) -> bool:
        return False

    def state_arrays(self):
        data = self.env.inner_env.sim.data
        return (float(data.time), np.asarray(data.qpos), np.asarray(data.qvel))

    def fingerprint(self):
        (time, qpos, qvel) = self.state_arrays()
        digest = hashlib.sha256()
        digest.update(np.asarray([time, self.frame_index], dtype='<f8').tobytes())
        digest.update(np.asarray(qpos, dtype='<f8').tobytes())
        digest.update(np.asarray(qvel, dtype='<f8').tobytes())
        return digest.hexdigest()

    def captured_prefix_views(self):
        """Real recorded rows consumed inside initialize(), before frame_index.

        Most tasks initialize at recorded row zero. Event-driven collectors may
        already have advanced to the query boundary; they must supply their
        actual past observations rather than padding or replaying controls.
        """
        return ()

    def prepare(self):
        self.initialize()
        capture_history = bool(self.config.get('capture_history', False))
        history = list(self.captured_prefix_views()) if capture_history else []
        if capture_history and len(history) != self.frame_index:
            raise ValueError('Captured initialization prefix must contain exactly rows 0..frame_index-1')
        while self.frame_index < self.chunk.start:
            if capture_history:
                history.append(self.current()['views'].copy())
            self.step_native()
            self.frame_index += 1
        observation = self.current()
        capture = {}
        if capture_history:
            history.append(observation['views'].copy())
            capture = {'capture_history_views': np.stack(history), 'capture_frame_indices': np.arange(self.frame_index + 1, dtype=np.int64)}
        self._paused_observation = observation
        self.build_candidates()
        if not self.bank or len({item.index for item in self.bank}) != len(self.bank):
            raise ValueError('Candidate IDs must be nonempty and unique')
        if len({item.order for item in self.bank}) != len(self.bank):
            raise ValueError('Candidate action orders must be unique')
        for candidate in self.bank:
            if candidate.model_actions.shape[0] != self.chunk.length:
                raise ValueError('Candidate conditioning does not match the query window')
        return {**observation, **capture, 'frame_index': self.frame_index, 'native_frame_indices': self.chunk.indices, 'observation_timing': self.chunk.observation_timing, 'fps': self.fps / self.chunk.stride, 'fingerprint': self.fingerprint(), 'sources': self.sources.provenance, 'case': self.case}

    def execute(self, action_index: int):
        if self.frame_index != self.chunk.start:
            raise RuntimeError('The simulator is no longer at the decision boundary')
        matches = [item for item in self.bank if item.index == action_index]
        if len(matches) != 1:
            raise KeyError(f'Unknown action: {action_index}')
        candidate = matches[0]
        self.activate(candidate)
        (observations, commands, diagnostics, frames) = ([], [], [], [])
        (native_commands, control_frames) = ([], [])
        while self.frame_index <= min(self.chunk.end, self.native_last_frame()):
            frame = self.frame_index
            observations.append(self._paused_observation if not observations else self.current())
            frames.append(frame)
            if self.episode_ends_on_observation():
                commands.append(candidate.model_actions[min(len(commands), len(candidate.model_actions) - 1)])
                diagnostics.append({'frame_index': frame, 'recorded_without_next_transition': True})
                break
            row = self.step_native()
            commands.append(np.asarray(row['model_action'], dtype=np.float32))
            native_commands.append(np.asarray(row['native_action'], dtype=np.float32))
            control_frames.append(frame)
            diagnostics.append({'frame_index': frame, **row})
            self.frame_index += 1
        if not observations:
            raise RuntimeError('No native observations were produced')
        requested = self.chunk.indices
        native_frames = np.asarray(frames, dtype=np.int64)
        positions = np.minimum(np.searchsorted(native_frames, requested), len(frames) - 1)
        views = np.stack([observations[index]['views'] for index in positions])
        actual = np.stack([commands[index] for index in positions])
        planned = np.asarray(candidate.model_actions, dtype=np.float32)
        return {'views': views, 'actions': actual, 'planned_actions': planned, 'native_actions': np.stack(native_commands) if native_commands else np.empty((0, 0), dtype=np.float32), 'native_control_frame_indices': np.asarray(control_frames, dtype=np.int64), 'native_action_semantics': getattr(self, 'native_action_semantics', 'original_controller_command'), 'recorded_actions': np.stack(commands), 'recorded_views': np.stack([observation['views'] for observation in observations]), 'recorded_frame_indices': native_frames, 'native_frame_indices': requested, 'source_frame_indices': native_frames[positions], 'padded': requested > native_frames[-1], 'native_control_iterations': sum((not row.get('recorded_without_next_transition', False) for row in diagnostics)), 'native_log': diagnostics, 'conditioning': candidate.conditioning, 'conditioning_max_abs_error': float(np.max(np.abs(actual - planned))), 'final_fingerprint': self.fingerprint()}

    def build_candidates(self):
        raise NotImplementedError

    def close(self):
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.sources.close()

def libero_observation(env, helper):
    obs = env._last_obs if env._last_obs is not None else env._refresh_obs()
    (main, wrist) = helper._obs_to_images(obs)
    return {'views': np.stack([main, wrist]), 'state': helper._obs_to_state(obs)}

def fastwam_action(native):
    raw = np.asarray(native, dtype=np.float32).copy()
    raw[-1] = (1.0 - raw[-1]) / 2.0
    return raw
