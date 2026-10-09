import json
from wan_video_action.data import RoboTwinUnifiedDataset
from wan_video_action.data.data_utils import pack_paths
from wan_video_action.data.operators import LoadCobotAction, ResolvePromptEmbPath, create_video_operator
from wan_video_action.data.history_prefix import CanonicalTargetEEF, HistoryPrefixDataset


def configure_dataset(dataset, args):
    if getattr(args, 'canonical_target_eef', False):
        with open(args.action_stat_path) as stream:
            stats = json.load(stream)['eef_target']
        dataset.special_operator_map['action'] = CanonicalTargetEEF(args, stats)
    if getattr(args, 'history_anchor_windows', False):
        dataset = HistoryPrefixDataset(dataset, args)
    return dataset


def build_dataset(args, runtime_config):
    special_operator_map = {}
    if runtime_config['text_enabled'] and 'prompt_emb' in runtime_config['data_file_keys']:
        special_operator_map['prompt_emb'] = ResolvePromptEmbPath(base_path=args.dataset_base_path)
    with open(args.action_stat_path, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    stat = {args.action_type: stats[args.action_type]} if args.action_type in stats else stats
    dataset = RoboTwinUnifiedDataset(base_path=args.dataset_base_path, metadata_path=args.dataset_metadata_path, repeat=args.dataset_repeat, data_file_keys=runtime_config['data_file_keys'], main_data_operator=create_video_operator(base_path=args.dataset_base_path, max_pixels=args.max_pixels, height=args.height, width=args.width, height_division_factor=16, width_division_factor=16, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, resize_mode=args.resize_mode, frame_stride=args.frame_stride, pad_short=args.pad_short_chunks), special_operator_map=special_operator_map)
    pack_paths(dataset.data, ('video', 'start_frame', 'end_frame'), ('action', 'start_frame', 'end_frame'))
    if 'action' in runtime_config['data_file_keys']:
        dataset.special_operator_map['action'] = LoadCobotAction(base_path=args.dataset_base_path, action_type=args.action_type, stat=stat, num_frames=args.num_frames, time_division_factor=args.time_division_factor, time_division_remainder=args.time_division_remainder, pad_short=args.pad_short_chunks, output_dim=args.action_dim, frame_stride=args.frame_stride)
    return configure_dataset(dataset, args)
