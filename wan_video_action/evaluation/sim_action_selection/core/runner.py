from __future__ import annotations
from .cases import prepare_cases
import hashlib
import importlib
import json
from pathlib import Path
import traceback
import numpy as np
from .artifacts import save_state, save_video, write_json
from .capture import encode_observations
from .decision_capture import capture_decision, capture_execution, capture_prefix, select_model_views
from .prediction_cache import CasePredictionCache
from .rpc import WorkerClient
from .scoring import prediction_accuracy
from .selection import executed_score, load_rules, merge_observed_supports, select_action

class Runner:
    """Selection with explicit shared or candidate-specific preparation scenes."""

    def __init__(self, task, config, resources, output):
        (self.task, self.config, self.resources) = (task, config, resources)
        self.model_view_indices = resources.get('model_view_indices', config.get('model_view_indices'))
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.rules = load_rules(task)
        self.perception = importlib.import_module('wan_video_action.evaluation.sim_action_selection.tasks.' + task + '.perception')
        self.model = None

    def simulator(self, case, log_path):
        configuration = dict(self.config)
        profile = self.resources.get('capture_profile') or {}
        configuration['capture_history'] = bool(profile.get('use_history', False) and profile.get('mode', 'raw') != 'raw')
        return WorkerClient(self.resources['simulator_python'], 'simulator', {'task': self.task, 'config': configuration, 'case': case, 'source_root': self.resources['source_root'], 'isolate_candidate_scenes': self.resources.get('isolate_candidate_scenes'), 'candidate_worker_log_dir': str(Path(log_path).with_suffix('')) + '_scenes'}, log_path, timeout=self.resources['worker_timeout'])

    def _prefix_frame_count(self):
        if self.task != 'mass_friction':
            return 1
        return int(self.resources.get('decision_prefix_frames', self.config.get('decision_prefix_frames', 3)))

    def _capture_prediction(self, case, candidate, candidates, paused, directory, name):
        frame_count = self._prefix_frame_count()
        fps = self.config.get('fps', paused['fps'])
        profile = self.resources.get('decision_capture_profile', self.resources.get('capture_profile'))
        if frame_count <= 1:
            return capture_decision(paused, profile, fps)
        profile = self.resources.get('decision_capture_profile')
        if not profile:
            profile = {'mode': 'jpeg_h264', 'jpeg_quality': 95, 'crf': 18, 'gop': 2}
            inherited = self.resources.get('capture_profile') or {}
            if inherited.get('mode') == 'jpeg_h264':
                profile.update(inherited)
        index = int(candidate['index'])
        prefix_case = dict(case)
        initializations = prefix_case.pop('candidate_initializations', None)
        if initializations:
            prefix_case.update(initializations[str(index)])
            if 'target_mass_kg' in prefix_case and 'target_table_friction_mu' in prefix_case:
                prefix_case.pop('pair_index', None)
        prefix_case['reference_action_index'] = index
        boundary = int(paused.get('frame_index', 0))
        configuration = dict(self.config)
        configuration.update({'native_last_frame': boundary + frame_count - 1, 'execution_mode': 'fixed_sequence', 'capture_history': False})
        action_bank = {'action_indices': np.asarray([int(item['index']) for item in candidates], dtype=np.int64), 'model_actions': np.stack([np.asarray(item['model_actions'], dtype=np.float32) for item in candidates]), 'native_frame_indices': np.asarray(paused['native_frame_indices'], dtype=np.int64)}
        simulator = WorkerClient(self.resources['simulator_python'], 'simulator', {'task': self.task, 'config': configuration, 'case': prefix_case, 'source_root': self.resources['source_root'], 'isolate_candidate_scenes': False}, directory / f'{name}_prefix_simulator.log', timeout=self.resources['worker_timeout'])
        try:
            prefix_boundary = simulator.call('prepare')
            simulator.call('set_candidate_actions', action_bank)
            executed = simulator.call('execute', index)
        finally:
            simulator.shutdown()
        transition_indices = set(range(boundary, boundary + frame_count - 1))
        transitions = [row for row in executed.get('native_log', []) if int(row['frame_index']) in transition_indices]
        if len(transitions) != len(transition_indices) or any(('block_block_contact' not in row for row in transitions)):
            raise ValueError('Preimpact capture requires native collision records for every used transition')
        if any((row['block_block_contact'] for row in transitions)):
            raise ValueError('Candidate encoding prefix must end before the block collision')
        planned = np.asarray(candidate['model_actions'], dtype=np.float32)
        errors = [float(np.max(np.abs(np.asarray(row['model_action'], dtype=np.float32) - planned[int(row['frame_index']) - boundary]))) for row in transitions]
        if any((not np.isfinite(error) or error > 1e-06 for error in errors)):
            raise ValueError('Candidate prefix controls differ from the model action sequence')
        (captured, audit) = capture_prefix(paused, executed, profile, fps, frame_count)
        audit.update({'candidate_action_index': index, 'candidate_specific_prefix': True, 'isolated_preview_simulator': True, 'main_simulator_advanced_for_encoding': False, 'postcollision_observations_used': False, 'native_contact_check_passed': True, 'maximum_used_transition_action_error': max(errors, default=0.0), 'prefix_boundary_frame': int(prefix_boundary.get('frame_index', 0))})
        np.savez_compressed(directory / f'{name}_encoding_prefix.npz', raw=np.asarray(executed['recorded_views'])[:frame_count], native_frame_indices=np.asarray(executed['recorded_frame_indices'])[:frame_count], model_current=captured)
        write_json(directory / f'{name}_prefix_native_log.json', transitions)
        return (captured, audit)

    def _supports(self, case, decision_directory):
        if getattr(self, '_case_supports', None) is not None:
            return self._case_supports
        if self.resources['method'] == 'standard':
            return []
        action_ids = case.get('support_actions', self.resources['support_actions'])
        supports = []
        for (number, action_index) in enumerate(action_ids):
            support_case = dict(case)
            support_case.pop('candidate_initializations', None)
            if self.task == 'light_switch':
                support_case['decision_button'] = ('red', 'blue')[int(action_index)]
            if self.task != 'light_switch':
                support_case['seed'] = int(case.get('seed', 0)) + 100000 + number
            simulator = self.simulator(support_case, decision_directory / f'support_{number:02d}.log')
            try:
                support_boundary = simulator.call('prepare')
                observed = simulator.call('execute', int(action_index))
                capture_profile = self.resources.get('capture_profile') or {}
                capture_options = {}
                if capture_profile.get('mode', 'raw') != 'raw':
                    capture_fps = self.config.get('fps')
                    if capture_fps is None:
                        capture_fps = (support_boundary or {}).get('fps')
                    if capture_fps is None or not np.isfinite(capture_fps) or capture_fps <= 0:
                        raise ValueError('Encoded support capture requires a finite positive frame rate')
                    capture_options['fps'] = capture_fps
                (observed['views'], capture) = encode_observations(observed['views'], capture_profile, **capture_options)
                write_json(decision_directory / f'support_{number:02d}_capture.json', capture)
                supports.append({'views': select_model_views(observed['views'], self.model_view_indices), 'actions': observed['actions'], 'action_index': int(action_index)})
                np.savez_compressed(decision_directory / f'support_{number:02d}.npz', views=observed['views'], actions=observed['actions'], native_frame_indices=observed['native_frame_indices'], source_frame_indices=observed['source_frame_indices'])
                write_json(decision_directory / f'support_{number:02d}.json', {'action_index': action_index, 'role': 'independent_calibration_episode_not_query_future', 'case': support_case, 'conditioning': observed['conditioning']})
            finally:
                simulator.shutdown()
        self._case_supports = supports
        return supports

    def run_decision(self, case, number, target_name):
        directory = self.output / f'decision_{number:05d}_{target_name}'
        directory.mkdir(exist_ok=False)
        simulator = self.simulator(case, directory / 'simulator.log')
        try:
            paused = simulator.call('prepare')
            candidates = simulator.call('candidates')
            observation_source = 'native'
            (current_views, capture) = capture_decision(paused, self.resources.get('decision_capture_profile', self.resources.get('capture_profile')), self.config.get('fps', paused['fps']))
            current_views = select_model_views(current_views, self.model_view_indices)
            native_model_views = select_model_views(paused['views'][None], self.model_view_indices)
            capture.update({'model_view_indices': self.model_view_indices, 'source_rgb_sha256': hashlib.sha256(np.ascontiguousarray(native_model_views).tobytes()).hexdigest(), 'model_rgb_sha256': hashlib.sha256(np.ascontiguousarray(current_views).tobytes()).hexdigest(), 'mae_uint8': float(np.abs(native_model_views.astype(np.float32) - current_views.astype(np.float32)).mean())})
            write_json(directory / 'capture.json', capture)
            rgb_hash = hashlib.sha256(np.ascontiguousarray(current_views).tobytes()).hexdigest()
            uid = f'{self.task}:{number}:{target_name}:{rgb_hash[:16]}'
            write_json(directory / 'boundary.json', {key: value for (key, value) in paused.items() if key not in {'views', 'capture_history_views'}})
            if 'capture_history_views' in paused:
                np.savez_compressed(directory / 'capture_history.npz', views=paused['capture_history_views'], native_frame_indices=paused['capture_frame_indices'])
            np.savez_compressed(directory / 'current_observation.npz', views=current_views, native_views=paused['views'][None])
            if self.resources.get('require_exact_conditioning'):
                exact_modes = {'exact_open_loop', 'exact_absolute_targets_feedback_native_delta', 'planned_joint_waypoints_original_pd_joint_pos', 'fixed_sequence'}
                non_exact = [item['conditioning'] for item in candidates if item['conditioning'] not in exact_modes]
                if non_exact:
                    raise ValueError(f'Future commands are feedback-dependent; strict offline conditioning is unavailable for {sorted(set(non_exact))}. No candidate simulator oracle will be used.')
            supports = self._supports(case, directory)
            use_observed_supports = bool(self.resources.get('use_observed_support_outcomes', True))
            context = self.model.call('freeze_context', {
                'decision_uid': uid,
                'supports': supports,
                'collect_support_observations': use_observed_supports,
            })
            support_observations = context.pop('observed_supports', [])
            write_json(directory / 'adaptation.json', context)
            observed_values = []
            if use_observed_supports:
                for (number, support) in enumerate(support_observations):
                    (state, value) = self.perception.outcome(support['views'], paused['fps'], self.resources['tracking'], self.rules)
                    save_state(directory / f'observed_support_{number:02d}_state.npz', state)
                    observed_values.append({**{key: item for (key, item) in support.items() if key != 'views'}, 'selection_value': value})
            write_json(directory / 'observed_support_outcomes.json', observed_values)
            observed_candidates = merge_observed_supports([{'action_index': int(candidate['index']), 'action_order': candidate['order'], 'selection_value': None, 'selection_source': 'missing_prediction'} for candidate in candidates], observed_values, self.rules)
            observed_by_index = {item['action_index']: item for item in observed_candidates if item['selection_source'] == 'observed_support'}
            predictions = []
            predicted_states = {}
            candidate_paused = {}
            candidate_scene_mode = bool(case.get('candidate_initializations'))
            for candidate in candidates:
                index = int(candidate['index'])
                if index in observed_by_index:
                    predictions.append({**observed_by_index[index], 'parameters': candidate['parameters'], 'conditioning': 'observed_support'})
                    continue
                (prediction_paused, prediction_views) = (paused, current_views)
                if candidate_scene_mode:
                    preparation = simulator.call('prepare_candidate', index)
                    prediction_paused = preparation['paused']
                    scene_candidate = preparation['candidate']
                    if int(scene_candidate['index']) != index or scene_candidate['order'] != candidate['order']:
                        raise ValueError('Candidate scene changed its action identity or order')
                    candidate = scene_candidate
                    (prediction_views, candidate_capture) = capture_decision(prediction_paused, self.resources.get('decision_capture_profile', self.resources.get('capture_profile')), self.config.get('fps', prediction_paused['fps']))
                    prediction_views = select_model_views(prediction_views, self.model_view_indices)
                    if prediction_paused['fps'] != paused['fps']:
                        raise ValueError('Candidate scenes must use the same model frame rate')
                    candidate_paused[index] = prediction_paused
                    write_json(directory / f'candidate_{index:02d}_capture.json', candidate_capture)
                    write_json(directory / f'candidate_{index:02d}_boundary.json', {key: value for (key, value) in prediction_paused.items() if key not in {'views', 'capture_history_views'}})
                    np.savez_compressed(directory / f'candidate_{index:02d}_current_observation.npz', views=prediction_views, native_views=prediction_paused['views'][None])
                elif self._prefix_frame_count() > 1:
                    (prediction_views, candidate_capture) = self._capture_prediction(case, candidate, candidates, prediction_paused, directory, f'candidate_{index:02d}')
                    prediction_views = select_model_views(prediction_views, self.model_view_indices)
                    write_json(directory / f'candidate_{index:02d}_capture.json', candidate_capture)
                    np.savez_compressed(directory / f'candidate_{index:02d}_current_observation.npz', views=prediction_views, native_views=prediction_paused['views'][None])
                frames = self.model.call('predict', {'views': prediction_views, 'actions': candidate['model_actions'], 'seed': int(case.get('prediction_seed', self.resources['seed']))})
                expected = (len(prediction_paused['native_frame_indices']),) + tuple(prediction_views.shape[1:])
                if not isinstance(frames, np.ndarray) or frames.dtype != np.uint8 or frames.shape != expected:
                    raise ValueError(f"Model video must be uint8 {expected}, got {getattr(frames, 'shape', None)}")
                if not np.array_equal(frames[0], prediction_views[0]):
                    raise ValueError('Prediction changed its current conditioning frame')
                (state, value) = self.perception.outcome(frames, paused['fps'], self.resources['tracking'], self.rules)
                index = int(candidate['index'])
                predicted_states[index] = state
                save_state(directory / f'predicted_action_{index:02d}_state.npz', state)
                np.savez_compressed(directory / f'candidate_{index:02d}_actions.npz', actions=candidate['model_actions'])
                if self.resources['save_predictions']:
                    save_video(directory / f'predicted_action_{index:02d}.mp4', frames, paused['fps'])
                predictions.append({'action_index': index, 'action_order': candidate['order'], 'selection_value': value, 'selection_source': 'model_prediction', 'parameters': candidate['parameters'], 'conditioning': candidate['conditioning'], 'current_rgb_sha256': hashlib.sha256(np.ascontiguousarray(prediction_views).tobytes()).hexdigest(), 'candidate_scene_mode': 'candidate_specific_scenes' if candidate_scene_mode else 'shared_current_scene', 'current_observation_source': observation_source})
            write_json(directory / 'predictions.json', predictions)
            if simulator.call('fingerprint') != paused['fingerprint']:
                raise RuntimeError('Live simulator advanced or changed during support adaptation/prediction')
            target = next((item for item in self.rules['targets'] if item['name'] == target_name))
            (selected, details) = select_action(predictions, target, self.rules)
            decision = {'decision_uid': uid, 'task': self.task, 'target': target_name, 'case': case, 'current_rgb_sha256': rgb_hash, 'native_frame_indices': paused['native_frame_indices'], 'selection_details': details, 'selected_action_index': None if selected is None else selected['action_index'], 'selected_predicted_value': None if selected is None else selected['selection_value'], 'selected_outcome_source': None if selected is None else selected['selection_source'], 'use_observed_support_outcomes': use_observed_supports, 'simulator_frozen_during_prediction': True, 'candidate_scene_mode': 'candidate_specific_scenes' if candidate_scene_mode else 'shared_current_scene', 'shared_candidate_first_frame': not candidate_scene_mode and self._prefix_frame_count() <= 1, 'candidate_encoding_prefix_frame_count': self._prefix_frame_count(), 'current_observation_source': observation_source}
            write_json(directory / 'selection.json', decision)
            if selected is None:
                result = {**decision, 'target_reached': False, 'outcome_available': False, 'reason': 'no_selected_action'}
                write_json(directory / 'result.json', result)
                return result
            selected_index = int(selected['action_index'])
            selected_paused = candidate_paused.get(selected_index, paused)
            if candidate_scene_mode and selected_index not in candidate_paused:
                selected_paused = simulator.call('prepare_candidate', selected_index)['paused']
            executed = simulator.call('execute', selected_index)
            (observed_views, execution_capture) = capture_execution(selected_paused, executed, self.resources.get('capture_profile'), self.config.get('fps', selected_paused['fps']))
            write_json(directory / 'execution_capture.json', execution_capture)
            (state, value) = self.perception.outcome(observed_views, selected_paused['fps'], self.resources['tracking'], self.rules)
            save_state(directory / 'executed_state.npz', state)
            save_video(directory / 'executed.mp4', observed_views, selected_paused['fps'])
            np.savez_compressed(directory / 'executed_observations.npz', raw=executed['views'], views=observed_views, native_frame_indices=executed['native_frame_indices'], source_frame_indices=executed['source_frame_indices'])
            np.savez_compressed(directory / 'executed_actions.npz', actual=executed['actions'], planned=executed['planned_actions'], native=executed['native_actions'], native_control_frame_indices=executed['native_control_frame_indices'], recorded_full_rate=executed['recorded_actions'], recorded_frame_indices=executed['recorded_frame_indices'], native_frame_indices=executed['native_frame_indices'], source_frame_indices=executed['source_frame_indices'], padded=executed['padded'])
            write_json(directory / 'native_execution.json', {key: value for (key, value) in executed.items() if key not in {'views', 'actions', 'planned_actions', 'native_actions', 'recorded_actions', 'recorded_views'}})
            if self.resources.get('require_exact_conditioning'):
                error = float(executed['conditioning_max_abs_error'])
                if not np.isfinite(error) or error > 1e-06:
                    raise ValueError(f'Executed commands differ from model conditioning: maximum absolute error {error}')
            result = {**decision, **executed_score(value, target), 'executed_value': value, 'prediction_accuracy': prediction_accuracy(predicted_states[int(selected['action_index'])], state, self.rules) if int(selected['action_index']) in predicted_states else {'available': False, 'reason': 'selected_observed_support_not_model_prediction'}, 'native_control_iterations': executed['native_control_iterations'], 'conditioning': executed['conditioning'], 'conditioning_max_abs_error': executed['conditioning_max_abs_error'], 'padded_frame_count': int(np.count_nonzero(executed['padded'])), 'native_frame_indices': executed['native_frame_indices'], 'selected_scene_reproduced_after_prediction': bool(executed.get('selected_scene_reproduced_after_prediction', False))}
            write_json(directory / 'result.json', result)
            return result
        except Exception:
            write_json(directory / 'error.json', {'case': case, 'traceback': traceback.format_exc()})
            raise
        finally:
            simulator.shutdown()

    def run(self, cases, targets):
        cases = prepare_cases(self.task, cases)
        write_json(self.output / 'resolved_configuration.json', {'task': self.task, 'config': self.config, 'resources': self.resources, 'cases': cases, 'targets': targets})
        model_configuration = {
            'method': self.resources['method'],
            'checkpoint': self.resources['checkpoint'],
            'context_table': self.resources.get('context_table'),
            'inference_config': self.config['inference_config'],
            'overrides': self.resources.get('model_overrides', {}),
            'seed': self.resources['seed'],
        }
        self.model = WorkerClient(self.resources['model_python'], 'model', model_configuration, self.output / 'model.log', self.resources['worker_timeout'])
        results = []
        try:
            model_worker = self.model
            for (case_index, case) in enumerate(cases):
                self._case_supports = None
                self.model = CasePredictionCache(model_worker)
                for target in targets:
                    number = len(results)
                    result = self.run_decision(case, number, target)
                    results.append(result)
                    write_json(self.output / 'results.json', results)
                    print(f"[{self.task}] case={case_index} target={target} action={result['selected_action_index']} reached={result['target_reached']}", flush=True)
            summary = {'task': self.task, 'decision_count': len(results), 'executed_target_reach_rate': sum((bool(row['target_reached']) for row in results)) / max(len(results), 1), 'per_target': {target: {'count': sum((row['target'] == target for row in results)), 'reached': sum((row['target'] == target and bool(row['target_reached']) for row in results))} for target in targets}, 'metric_scope': 'observed_selected_action_target_reach_not_full_offline_oracle_ranking', 'offline_oracle_dependent_decisions': sum((bool(row.get('oracle_required')) for row in results))}
            write_json(self.output / 'summary.json', summary)
            return summary
        finally:
            self.model.shutdown()
