from __future__ import annotations

from tempfile import TemporaryDirectory

import torch


def method_parser(parser, method):
    if method == "dino":
        from t2z.training.dino import add_dinov2_config
        return add_dinov2_config(parser)
    if method == "ttt":
        from t2z.training.ttt import add_ttt_kvb_config
        return add_ttt_kvb_config(parser)
    return parser


class OnlineMethod:
    def __init__(self, method, args, configuration):
        from t2z.inference.standard import build_pipeline

        self.method = method
        self.args = args
        self.temporary = None
        self.controller = None
        self.adapters = []
        self.model = None
        self.support_encoder = None
        self.prepared = False
        checkpoint = str(configuration["checkpoint"])
        if method == "dino":
            from t2z.config import resolve
            from t2z.inference.dino import materialize_wan_checkpoint, load_support_encoder

            self.temporary = TemporaryDirectory(prefix="t2z_online_dino_")
            args.ckpt_path = self.temporary.name + "/world_model.safetensors"
            args.dinov2_checkpoint_path = checkpoint
            args.dinov2_model_path = str(resolve(args.dinov2_model_path))
            materialize_wan_checkpoint(checkpoint, args.ckpt_path)
            self.pipe = build_pipeline(args)
            self.support_encoder = load_support_encoder(args, self.pipe.device)
        elif method == "ttt":
            from t2z.inference.ttt_model import build_training_faithful_model

            args.ttt_checkpoint_path = checkpoint
            self.model, installation = build_training_faithful_model(args)
            self.pipe = self.model.pipe
            self.controller = installation.controller
            self.original_model_fn = self.pipe.model_fn
        else:
            if method in ("standard", "lora"):
                args.ckpt_path = checkpoint
            elif method != "ours":
                raise ValueError(f"Unknown online method: {method}")
            self.pipe = build_pipeline(args)
            if method == "lora":
                from t2z.inference.lora import install_lora

                targets = args.lora_target_modules
                targets = targets.split(",") if isinstance(targets, str) else targets
                self.adapters = install_lora(
                    self.pipe.dit, targets={str(value).strip() for value in targets},
                    rank=int(args.lora_rank), alpha=float(getattr(args, "lora_alpha", 8.0)),
                    seed=int(getattr(args, "adapter_seed", args.seed)),
                )

    def freeze(self, supports, decision_uid):
        if not supports:
            raise ValueError(f"{self.method} requires observed simulator interactions")
        context = None
        report = {"method": self.method, "support_count": len(supports),
                  "decision_uid": decision_uid, "frozen_for_all_candidates": True}
        if self.method == "dino":
            with torch.no_grad():
                encoded = tuple(self.support_encoder.extract_visual_features(item["video"]) for item in supports)
                context = self.support_encoder.project_supports(
                    visual_features=tuple(item[0] for item in encoded),
                    actions=tuple(item["action"] for item in supports),
                    frame_indices=tuple(item[1] for item in encoded),
                    frame_counts=tuple(item[2] for item in encoded),
                )[0].detach()
            report["context"] = context.float().cpu().numpy()
            report["support_reduction"] = "mean"
        elif self.method == "ttt":
            from t2z.inference.ttt import GroupedSameTimestepSupportQueryModelFn
            from t2z.inference.ttt_stream import _prepare_support_model_inputs

            self.pipe.model_fn = self.original_model_fn
            self.controller.clear()
            replays = tuple(
                _prepare_support_model_inputs(
                    self.model, item, noise_seed=int(self.args.seed) + 1_000_003 + ordinal * 104729,
                ) for ordinal, item in enumerate(supports)
            )
            self.pipe.model_fn = GroupedSameTimestepSupportQueryModelFn(
                pipe=self.pipe, controller=self.controller,
                original_model_fn=self.original_model_fn, support_replays=replays,
            )
            report["protocol"] = "same_timestep_support_write_query_update_then_read"
        elif self.method == "lora":
            from t2z.inference.lora import adapt_lora

            defaults = {"lora_steps": 50, "lora_learning_rate": 1.0e-4,
                        "lora_gradient_clip": 1.0}
            for name, value in defaults.items():
                if not hasattr(self.args, name):
                    setattr(self.args, name, value)
            parameters = [parameter for _, adapter in self.adapters
                          for parameter in (adapter.lora_a, adapter.lora_b)]
            for parameter in parameters:
                parameter.requires_grad_(True)
            try:
                with torch.enable_grad():
                    report["losses"] = adapt_lora(self.pipe, supports, self.adapters, self.args)
            finally:
                for parameter in parameters:
                    parameter.requires_grad_(False)
            report["support_reduction"] = "mean"
        self.prepared = True
        return context, report

    def inference_context(self):
        return torch.no_grad() if self.method == "ttt" else torch.inference_mode()

    def close(self):
        if self.controller is not None:
            self.controller.clear()
            self.pipe.model_fn = self.original_model_fn
        self.pipe = self.model = self.support_encoder = None
        self.adapters = []
        if self.temporary is not None:
            self.temporary.cleanup()
