from __future__ import annotations
from .methods import OnlineMethod, method_parser
import argparse
import json
from pathlib import Path
import numpy as np

class WanPredictor:
    """Use T2-z's existing pipeline, action loader, and Ours adaptation loss.

    The model process never imports a simulator or receives hidden physics,
    target labels, candidate GT videos, or future query observations.
    """

    def __init__(self, configuration):
        import torch
        from t2z.config import flatten, load_config, resolve
        from t2z.inference.ours import add_adaptation_config, _freeze_pipe
        from wan_video_action.data import LoadCobotAction
        from wan_video_action.parsers import add_general_config
        from wan_video_action.pipelines.wan_video_action import load_checkpoint_weights
        from wan_video_action.utils import set_global_seed
        self.torch = torch
        self.configuration = configuration
        self.method = configuration.get('method', 'ours')
        specification = load_config(resolve(configuration['inference_config']))
        values = flatten(load_config(resolve(specification['training_configs'][self.method])))
        values.update(specification.get('inference', {}))
        if self.method == 'ours':
            values.update(specification.get('ours', {}))
        parser = add_general_config(argparse.ArgumentParser())
        add_adaptation_config(parser)
        args = method_parser(parser, self.method).parse_args([])
        for (key, value) in values.items():
            setattr(args, key, value)
        for (key, value) in configuration.get('overrides', {}).items():
            if value is not None:
                setattr(args, key, value)
        args.seed = int(configuration.get('seed', getattr(args, 'seed', 0)))
        args.video_light_augmentation_enabled = False
        args.stage2_known_context_keys = ''
        args.num_history_frames = 1
        checkpoint = resolve(configuration['checkpoint'])
        if self.method == 'standard':
            args.ckpt_path = str(checkpoint)
        for key in ('model_paths', 'model_config_path', 'ckpt_path', 'action_stat_path'):
            value = getattr(args, key, None)
            if value:
                setattr(args, key, str(resolve(value)))
        for (name, path) in (('evaluation checkpoint', checkpoint), ('base checkpoint', Path(args.ckpt_path)), ('action normalization statistics', Path(args.action_stat_path)), ('base model', Path(args.model_paths)), ('model configuration', Path(args.model_config_path))):
            if not path.exists():
                raise FileNotFoundError(f'Missing {name}: {path}. Supply the corresponding resource override; no fabricated fallback is used.')
        with open(args.action_stat_path) as stream:
            stats = json.load(stream)
        stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats
        self.action_loader = LoadCobotAction(action_type=args.action_type, stat=stat, output_dim=int(args.action_dim), num_frames=int(args.num_frames))
        self.action_loader._get_min_max()
        set_global_seed(args.seed)
        self.online_method = OnlineMethod(self.method, args, self.configuration)
        self.pipe = self.online_method.pipe
        if self.method == 'ours':
            load_checkpoint_weights(self.pipe, str(checkpoint))
        _freeze_pipe(self.pipe)
        self.args = args
        self.context = None
        self.context_report = {}

    def video_tensor(self, views):
        array = np.asarray(views)
        if array.ndim != 5 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError('Video must be uint8 [time, views, height, width, RGB]')
        if tuple(array.shape[2:4]) != (int(self.args.height), int(self.args.width)):
            raise ValueError('Simulator cameras must match training resolution; implicit resizing is disabled')
        video = self.torch.as_tensor(array.copy(), dtype=self.torch.float32).permute(1, 4, 0, 2, 3)
        video = video * (2.0 / 255.0) - 1.0
        return video.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)

    def action_tensor(self, raw):
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError('Raw actions must be a finite [time, channels] array')
        (low, high) = self.action_loader._get_min_max()
        if array.shape[1] != len(low) or len(high) != len(low):
            raise ValueError(f'Recorded action width {array.shape[1]} does not match normalization width {len(low)}')
        normalized = self.action_loader._normalize_bound(array, low, high)
        packed = self.action_loader._match_output_dim(normalized)
        return self.torch.as_tensor(packed, device=self.pipe.device, dtype=self.pipe.torch_dtype)[None]

    def freeze_context(self, request):
        if self.method in ('dino', 'ttt', 'lora'):
            observed = request.get('supports', [])
            supports = [{'video': self.video_tensor(item['views']), 'action': self.action_tensor(item['actions'])} for item in observed]
            (self.context, report) = self.online_method.freeze(supports, request.get('decision_uid'))
            if request.get('collect_support_observations', False):
                report['observed_supports'] = [{'action_index': int(item['action_index']), 'views': item.get('observed_views', item['views'])} for item in observed]
            return report
        from t2z.inference.ours import _adapt_ttt_state, _context_table_mean_tensor, _load_grouped_context_table
        self.context = None
        if self.method != 'ours':
            self.context_report = {'method': self.method, 'adaptation': False}
            return self.context_report
        table_path = self.configuration.get('context_table')
        if not table_path:
            raise ValueError('Ours requires the saved context table for its original table-mean initializer')
        table = _load_grouped_context_table(table_path)
        initial = _context_table_mean_tensor(table, device=self.pipe.device, dtype=self.torch.float32)
        supports = []
        observed_supports = []
        collect_observations = bool(request.get('collect_support_observations', False))
        for support in request.get('supports', []):
            supports.append({'video': self.video_tensor(support['views']), 'action': self.action_tensor(support['actions'])})
            if collect_observations:
                observed_supports.append({'action_index': int(support['action_index']), 'views': np.asarray(support['views'], dtype=np.uint8), 'source': 'independent_calibration_episode'})
        if not supports:
            raise ValueError("Ours needs actual past support rollouts or explicit support metadata indices; it cannot adapt using a query's future")
        (context, losses, metrics, trajectory) = _adapt_ttt_state(self.pipe, supports, self.args, initial_context=initial, target_context=None, trajectory_meta={'decision_uid': request['decision_uid']})
        self.context = context.detach().clone()
        self.context_report = {'initializer': 'table_mean_not_physics_lookup', 'support_count': len(supports), 'losses': losses, 'metrics': metrics, 'trajectory': trajectory, 'context': self.context.float().cpu().numpy(), 'frozen_for_all_candidates': True, 'observed_supports': observed_supports}
        return self.context_report

    def predict(self, request):
        if set(request) - {'views', 'actions', 'seed'}:
            raise ValueError('Prediction requests may contain only the current RGB, actions, and random seed')
        views = np.asarray(request['views'])
        if views.shape[0] != 1:
            raise ValueError('Online inference may condition on the current frame only')
        input_video = self.video_tensor(views)
        action = self.action_tensor(request['actions'])
        if action.shape[1] != int(self.args.num_frames):
            raise ValueError('Candidate actions must match the original evaluation chunk length')
        if self.method == 'ours' and self.context is None:
            raise RuntimeError('Adapt and freeze a context before predicting candidates')
        with self.online_method.inference_context():
            output = self.pipe(input_video=input_video, action=action, physical_context=self.context, seed=int(request['seed']), rand_device='cpu', tiled=False, height=int(self.args.height) * int(input_video.shape[0]), width=int(self.args.width), num_frames=int(self.args.num_frames), num_history_frames=1, cfg_scale=float(self.args.cfg_scale), num_inference_steps=int(self.args.num_inference_steps), use_history_condition_noise_in_inference=True, progress_bar_cmd=lambda iterable, *args, **kwargs: iterable, output_type='floatpoint')
        expected = (views.shape[1], 3, int(self.args.num_frames), int(self.args.height), int(self.args.width))
        if tuple(output.shape) != expected or not self.torch.isfinite(output).all():
            raise ValueError(f'Pipeline decoded an invalid video; expected finite VCTHW {expected}')
        frames = output.detach().float().cpu().permute(2, 0, 3, 4, 1).numpy()
        frames = np.clip((frames + 1.0) * 127.5, 0, 255).astype(np.uint8)
        frames[0] = views[0]
        if getattr(self.args, 'prediction_video_roundtrip', False):
            from .prediction_video import roundtrip_prediction_video
            initial = input_video[:, :, 0].detach().float().cpu().permute(0, 2, 3, 1).numpy()
            frames[0] = np.clip((initial + 1.0) * 127.5, 0, 255).astype(np.uint8)
            frames = roundtrip_prediction_video(frames, float(self.args.fps), int(self.args.quality))
            frames[0] = views[0]
        return frames

    def close(self):
        if hasattr(self, 'online_method'):
            self.online_method.close()
        self.context = None
        self.pipe = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
