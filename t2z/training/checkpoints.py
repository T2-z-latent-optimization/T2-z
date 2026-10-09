import fcntl
import json
import os
from pathlib import Path
import time
import torch
from safetensors.torch import save_file


def atomic_json(path, value):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.partial')
    with temporary.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class AllocationFinished(Exception):
    pass


class ProtectedCheckpointLogger:

    def __init__(self, cfg, method):
        self.cfg = cfg
        self.method = method
        prefix = {'standard': 'standard', 'dino': 'dinov2', 'ttt': 'ttt'}[method]
        self.max_updates = int(cfg[prefix + '_max_updates'])
        raw_steps = cfg.get('protected_checkpoint_steps', '')
        self.protected_steps = {int(value) for value in str(raw_steps).split(',') if value.strip()}
        self.output_prefix = prefix
        self.output = Path(cfg['output_path'])
        self.num_steps = int(cfg.get('ttt_start_step', 0))
        self.previous_time = time.monotonic()
        self.deadline = float(os.environ.get(cfg.get('checkpoint_deadline_env', 'T2Z_TRAINING_DEADLINE_EPOCH'), 'inf'))
        self.output.mkdir(parents=True, exist_ok=True)
        if int(cfg['checkpoint_keep_last']) < 1:
            raise ValueError('checkpoint_keep_last must be positive')
        self.stop_flag = None
        self.saved = set()
        self.lock = None

    def global_chunks(self, accelerator):
        envs = int(self.cfg[self.output_prefix + '_environments_per_rank'])
        if self.method == 'ttt':
            count = int(self.cfg['ttt_sequence_length']) * int(self.cfg.get('ttt_streams_per_environment', 1))
        elif self.method == 'dino':
            count = int(self.cfg['dinov2_trajectories_per_environment'])
        else:
            count = int(self.cfg['standard_episodes_per_environment'])
        return accelerator.num_processes * envs * count

    def save(self, accelerator, model, *, protected=False, reason='periodic'):
        accelerator.wait_for_everyone()
        state = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            directory = self.output / 'protected' if protected else self.output
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f'step-{self.num_steps}.safetensors'
            marker = directory / f'.step-{self.num_steps}.safetensors.complete'
            metadata = directory / f'step-{self.num_steps}.metadata.json'
            if target.exists() or marker.exists() or metadata.exists():
                raise FileExistsError(f'Refusing to overwrite checkpoint bundle: {target}')
            exported = accelerator.unwrap_model(model).export_trainable_state_dict(state, remove_prefix=self.cfg['remove_prefix_in_ckpt'])
            exported = {k: v.detach().cpu().contiguous() for k, v in exported.items()}
            if self.method == 'ttt' and not any(('ttt_kvb_memory' in k for k in exported)):
                raise RuntimeError('Refusing to save a checkpoint without TTT memory parameters.')
            if self.method == 'ttt' and not any(('ttt_kvb_gate' in k for k in exported)):
                raise RuntimeError('Refusing to save a checkpoint without TTT gates.')
            if self.method == 'dino' and not (any(k.startswith('wan.') for k in exported) and any(k.startswith('support_encoder.') for k in exported)):
                raise RuntimeError('DINO checkpoint must include both the world model and support encoder')
            temporary = directory / f'.step-{self.num_steps}.{os.getpid()}.partial'
            save_file(exported, str(temporary), metadata={'step': str(self.num_steps), 'method': self.method, 'reason': reason})
            with temporary.open('rb') as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            atomic_json(metadata, {'step': self.num_steps, 'reason': reason, 'protected': protected, 'method': self.method, 'config': self.cfg, 'optimizer_resume_state_saved': False})
            atomic_json(marker, {'checkpoint': target.name, 'step': self.num_steps})
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            print(f'[checkpoint] saved={target} reason={reason}', flush=True)
            if not protected:
                completed = sorted((p for p in directory.glob('step-*.safetensors') if p.stem.removeprefix('step-').isdigit() and (directory / ('.' + p.name + '.complete')).is_file()), key=lambda p: int(p.stem.removeprefix('step-')))
                for old in completed[:-int(self.cfg['checkpoint_keep_last'])]:
                    old.unlink()
                    (directory / f'{old.stem}.metadata.json').unlink()
                    (directory / ('.' + old.name + '.complete')).unlink()
        del state
        self.saved.add(self.num_steps)
        accelerator.wait_for_everyone()

    def on_step_end(self, accelerator, model, save_steps, *, loss):
        self.num_steps += 1
        if self.stop_flag is None:
            self.stop_flag = torch.zeros((), device=accelerator.device, dtype=torch.int32)
            if accelerator.is_main_process:
                self.lock = (self.output / '.training.lock').open('a')
                fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                atomic_json(self.output / f'{self.method}_run_settings.json', {'config': self.cfg, 'deadline_epoch': self.deadline, 'sampling_scope': self.cfg['environment_sampling_scope'], 'world_size': accelerator.num_processes, 'global_chunks_per_optimizer_update': self.global_chunks(accelerator)})
        if accelerator.is_main_process:
            self.stop_flag.fill_(int(time.time() >= self.deadline))
        if accelerator.num_processes > 1:
            torch.distributed.broadcast(self.stop_flag, src=0)
        stopping = bool(self.stop_flag.item())
        if self.num_steps % int(self.cfg['log_steps']) == 0:
            now = time.monotonic()
            if accelerator.is_main_process:
                record = {'step': self.num_steps, 'loss': float(loss.detach().cpu()), 'seconds_per_step': (now - self.previous_time) / int(self.cfg['log_steps']), 'timestamp': time.time(), 'global_chunks': self.global_chunks(accelerator)}
                with (self.output / 'loss.jsonl').open('a') as handle:
                    handle.write(json.dumps(record) + '\n')
                print(f'[{self.method}_train] ' + json.dumps(record), flush=True)
            self.previous_time = now
        final = self.num_steps == self.max_updates
        if self.num_steps in self.protected_steps or stopping or final:
            reason = 'deadline' if stopping else 'final' if final else 'protected_step'
            self.save(accelerator, model, protected=True, reason=reason)
        elif self.num_steps % int(save_steps) == 0:
            self.save(accelerator, model)
        if stopping:
            accelerator.end_training()
            raise AllocationFinished()

    def on_training_end(self, accelerator, model, save_steps):
        if self.num_steps not in self.saved:
            self.save(accelerator, model, protected=True, reason='final')
        accelerator.end_training()


def build_logger(args, method):
    if args.checkpoint_policy == 'protected':
        return ProtectedCheckpointLogger(vars(args), method)
    from scripts.train import TimedRetentionModelLogger
    return TimedRetentionModelLogger(args.output_path, remove_prefix_in_ckpt=args.remove_prefix_in_ckpt, save_minutes=args.checkpoint_save_minutes, keep_last=args.checkpoint_keep_last, log_steps=args.log_steps)
