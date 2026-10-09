from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import uuid
from . import TASKS
ROOT = Path(__file__).resolve().parents[3]

def indices(value):
    return [] if not value else [int(item.strip()) for item in value.split(',') if item.strip()]

def main():
    parser = argparse.ArgumentParser(description='Evaluate T2-z model action selection through real simulation execution')
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--method', choices=('ours', 'standard', 'dino', 'ttt', 'lora'), default='ours')
    parser.add_argument('--dino-model-path', type=Path)
    parser.add_argument('--context-table', type=Path)
    parser.add_argument('--case', type=Path, help='One original episode specification as JSON')
    parser.add_argument('--cases', type=Path, help='JSONL: one original episode specification per line')
    parser.add_argument('--target', default='all', help='An existing task target name, or all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT.parent / 'TTT4dynamics')
    parser.add_argument('--model-python', type=Path, default=Path(sys.executable))
    parser.add_argument('--simulator-python', type=Path)
    parser.add_argument('--task-config', type=Path, help='Override the simulation task configuration')
    parser.add_argument('--capture-profile', type=Path, help='Optional JSON profile for model-observation encoding; native physics remains unchanged')
    parser.add_argument('--model-view-indices', help='Native camera indices consumed by the trained model, e.g. 0 or 0,1')
    parser.add_argument('--model-paths', type=Path)
    parser.add_argument('--base-checkpoint', type=Path)
    parser.add_argument('--action-stats', type=Path)
    parser.add_argument('--use-observed-support-outcomes', action=argparse.BooleanOptionalAction, default=True, help='Reuse observed support outcomes for their actions, as in offline evaluation')
    parser.add_argument('--support-actions', help='Independent native calibration action IDs, e.g. 0,4; an empty string disables them')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--worker-timeout', type=float, default=1800)
    parser.add_argument('--save-predictions', action='store_true')
    parser.add_argument('--require-exact-conditioning', action='store_true', help='Reject nominal feedback-dependent action conditioning rather than claim strict offline parity')
    args = parser.parse_args()
    if args.case and args.cases:
        parser.error('Choose --case or --cases, not both')
    task_directory = Path(__file__).parent / 'tasks' / args.task
    configuration = json.loads((args.task_config or task_directory / 'config.json').read_text())
    if args.case:
        cases = [json.loads(args.case.read_text())]
    elif args.cases:
        cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    else:
        cases = [dict(configuration['default_case'])]
    if not cases:
        parser.error('At least one case is required')
    for case in cases:
        case.setdefault('seed', args.seed)
    from .core.selection import load_rules
    from ...evaluation.config import load_evaluation_config
    target_names = [target['name'] for target in load_rules(args.task)['targets']]
    targets = target_names if args.target == 'all' else [args.target]
    if any((target not in target_names for target in targets)):
        parser.error(f"Targets for {args.task}: {', '.join(target_names)}")
    if args.method == 'ours' and args.context_table is None:
        automatic = args.checkpoint.resolve().parent / 'context_table.json'
        if not automatic.is_file():
            parser.error('Ours requires --context-table, or context_table.json beside the checkpoint')
        args.context_table = automatic
    source_root = args.source_root.resolve()
    simulator_python = args.simulator_python
    if simulator_python is None:
        repository = source_root.parent / 'robomme_benchmark' if args.task == 'light_switch' else source_root
        simulator_python = repository / '.venv' / 'bin' / 'python'
    for (name, path) in (('simulator Python', simulator_python), ('model Python', args.model_python), ('checkpoint', args.checkpoint)):
        if not path.exists():
            parser.error(f'Missing {name}: {path}')
    tracking = configuration['tracking']
    if isinstance(tracking, str):
        tracking = load_evaluation_config(ROOT / tracking)
    output = args.output or ROOT / 'outputs' / 'sim_action_selection' / (args.task + '_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    if output.exists() and any(output.iterdir()):
        parser.error(f'Refusing to overwrite an existing evaluation: {output}')
    override_names = {'model_paths': args.model_paths, 'ckpt_path': args.base_checkpoint, 'action_stat_path': args.action_stats}
    resources = {'model_view_indices': configuration.get('model_view_indices') if args.model_view_indices is None else indices(args.model_view_indices), 'capture_profile': configuration.get('capture_profile') if args.capture_profile is None else json.loads(args.capture_profile.read_text()), 'source_root': str(source_root), 'model_python': str(args.model_python.absolute()), 'simulator_python': str(simulator_python.absolute()), 'checkpoint': str(args.checkpoint.resolve()), 'method': args.method, 'context_table': None if args.context_table is None else str(args.context_table.resolve()), 'use_observed_support_outcomes': args.use_observed_support_outcomes, 'support_actions': configuration['support_actions'] if args.support_actions is None else indices(args.support_actions), 'tracking': tracking, 'seed': args.seed, 'worker_timeout': args.worker_timeout, 'save_predictions': args.save_predictions, 'require_exact_conditioning': args.require_exact_conditioning, 'model_overrides': {key: str(path.resolve()) for (key, path) in override_names.items() if path is not None}}
    if args.dino_model_path is not None:
        resources['model_overrides']['dinov2_model_path'] = str(args.dino_model_path.expanduser().absolute())
    from .core.runner import Runner
    summary = Runner(args.task, configuration, resources, output).run(cases, targets)
    print(json.dumps(summary, indent=2))
    print(f'Evaluation artifacts: {output}')
