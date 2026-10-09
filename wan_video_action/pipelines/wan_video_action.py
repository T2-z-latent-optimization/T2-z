import torch
from tqdm import tqdm
from typing import Optional, Union
from einops import rearrange

from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.diffusion.base_pipeline import PipelineUnit
from diffsynth.core.device.npu_compatible_device import get_device_type
from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.models.wan_video_dit import sinusoidal_embedding_1d

from ..models.wan_video_action_encoder import WanVideoActionEncoder
from ..models.physical_context import PhysicalContextEncoder, PhysicalResidualAdapterBank
from ..models.wan_video_vae import apply_wan_vae_compat


def _prepare_history_condition_latents(
    self: WanVideoPipeline,
    inputs_shared: dict,
    *,
    use_history_condition_noise_in_inference: bool,
):
    first_frame_latents = inputs_shared.get("first_frame_latents")
    if first_frame_latents is None:
        return None, 0
    latents = inputs_shared.get("latents")
    if latents is None:
        return None, 0

    if first_frame_latents.ndim == 4:
        first_frame_latents = first_frame_latents.unsqueeze(0)

    history_t = min(int(first_frame_latents.shape[2]), int(latents.shape[2]))
    if history_t <= 0:
        return None, 0

    conditioning_latents = first_frame_latents[:, :, :history_t].clone()
    inputs_shared["latents"][:, :, :history_t] = conditioning_latents

    if (
        use_history_condition_noise_in_inference
        and getattr(self, "action_injection_mode", "none") == "adaln"
        and history_t > 1
    ):
        noise = inputs_shared.get("noise")
        if not isinstance(noise, torch.Tensor):
            raise RuntimeError("Expected `noise` tensor for history-conditioned inference, but it was missing.")
        small_timestep_idx = max(0, len(self.scheduler.timesteps) - 50)
        small_timestep = self.scheduler.timesteps[small_timestep_idx].unsqueeze(0).to(
            dtype=self.torch_dtype,
            device=self.device,
        )
        conditioning_latents[:, :, 1:history_t] = self.scheduler.add_noise(
            conditioning_latents[:, :, 1:history_t],
            noise[:, :, 1:history_t],
            small_timestep,
        )
        inputs_shared["latents"][:, :, 1:history_t] = conditioning_latents[:, :, 1:history_t]
    return conditioning_latents, history_t


def _restore_history_condition_latents(
    inputs_shared: dict,
    *,
    conditioning_latents: Optional[torch.Tensor],
    history_t: int,
) -> None:
    if conditioning_latents is None or history_t <= 0:
        return
    inputs_shared["latents"][:, :, :history_t] = conditioning_latents[:, :, :history_t]


def _resolve_arg(args, name: str, default):
    if args is None:
        return default
    return getattr(args, name, default)


def _append_context_tokens(context: Optional[torch.Tensor], tokens: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if tokens is None:
        return context
    if context is None:
        return tokens
    return torch.cat([context, tokens], dim=1)


def _parse_adapter_layers(layers: Optional[str], num_layers: int) -> Optional[list[int]]:
    if layers is None:
        return None
    layers = str(layers).strip().lower()
    if layers in {"", "all"}:
        return None
    if layers in {"last_quarter", "last-quarter"}:
        start = (3 * num_layers) // 4
        return list(range(start, num_layers))
    if layers.startswith("last:"):
        count = int(layers.split(":", 1)[1])
        if count <= 0:
            raise ValueError(f"last adapter layer count must be positive, got {count}.")
        start = max(0, num_layers - count)
        return list(range(start, num_layers))
    if layers.startswith("uniform:"):
        count = int(layers.split(":", 1)[1])
        if count <= 0:
            raise ValueError(f"uniform adapter layer count must be positive, got {count}.")
        if count >= num_layers:
            return list(range(num_layers))
        if count == 1:
            return [num_layers // 2]
        return sorted(set(round(i * (num_layers - 1) / (count - 1)) for i in range(count)))
    return [int(part.strip()) for part in layers.split(",") if part.strip()]


def _build_wan2_action_units(pipe: WanVideoPipeline):
    selected = [
        unit for unit in pipe.units
        if unit.__class__.__name__ in {
            "WanVideoUnit_ShapeChecker",
            "WanVideoUnit_NoiseInitializer",
        }
    ]
    selected.append(WanVideoUnit_InputVideoEmbedder())
    selected.append(WanVideoUnit_ImageEmbedderFused())
    selected.append(WanVideoUnit_ActionEmbedder())
    if getattr(pipe, "physical_context_encoder", None) is not None:
        selected.append(WanVideoUnit_PhysicalContextEmbedder())
    if getattr(pipe, "background_context_encoder", None) is not None:
        selected.append(WanVideoUnit_BackgroundContextEmbedder())
    return selected


def _install_wan_video_action_call(pipeline: WanVideoPipeline) -> None:
    @torch.no_grad()
    def __call__(
        self: WanVideoPipeline,
        input_video: Optional[torch.Tensor] = None,
        denoising_strength: Optional[float] = 1.0,
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames: int = 81,
        num_history_frames: int = 1,
        action: Optional[torch.Tensor] = None,
        physical_context: Optional[torch.Tensor] = None,
        background_context: Optional[torch.Tensor] = None,
        extra_condition_tokens: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        tiled: bool = True,
        tile_size: tuple[int, int] = (30, 52),
        tile_stride: tuple[int, int] = (15, 26),
        use_history_condition_noise_in_inference: bool = False,
        progress_bar_cmd=tqdm,
        output_type: str = "quantized",
        **_: dict,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        input_video = input_video.to(dtype=self.torch_dtype, device=self.device)

        inputs_posi = {}
        inputs_nega = {}
        inputs_shared = {
            "vace_reference_image": None,
            "input_video": input_video,
            "num_views": int(input_video.shape[0]),
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_history_frames": num_history_frames,
            "action": action,
            "physical_context": physical_context,
            "background_context": background_context,
            "extra_condition_tokens": extra_condition_tokens,
            "cfg_scale": cfg_scale,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
        }

        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        conditioning_latents, history_t = _prepare_history_condition_latents(
            self,
            inputs_shared,
            use_history_condition_noise_in_inference=use_history_condition_noise_in_inference,
        )
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        use_gradient_checkpointing = self.use_gradient_checkpointing
        use_gradient_checkpointing_offload = self.use_gradient_checkpointing_offload

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred_posi = self.model_fn(
                **models,
                **inputs_shared,
                timestep=timestep,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            )
            noise_pred = noise_pred_posi
            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            _restore_history_condition_latents(
                inputs_shared,
                conditioning_latents=conditioning_latents,
                history_t=history_t,
            )

        self.load_models_to_device(["vae"])
        latents = inputs_shared["latents"]
        num_views = int(inputs_shared.get("num_views", 1))
        if latents.shape[-2] % num_views != 0:
            raise ValueError(
                f"Latent height {latents.shape[-2]} is not divisible by num_views={num_views}."
            )
        latents_by_view = rearrange(latents, "b c t (v h) w -> (b v) c t h w", v=num_views, h=latents.shape[-2] // num_views)
        video = self.vae.decode(latents_by_view, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if output_type == "quantized":
            video = self.vae_output_to_video(video)
        elif output_type == "floatpoint":
            pass
        else:
            raise ValueError(f"Unsupported output_type='{output_type}', expected 'quantized' or 'floatpoint'.")
        if use_history_condition_noise_in_inference and history_t > 0:
            history_to_copy = min(
                int(num_history_frames),
                int(video.shape[2]),
                int(input_video.shape[2]),
            )
            if history_to_copy > 0:
                video[:, :, :history_to_copy] = input_video[:, :, :history_to_copy].to(
                    dtype=video.dtype,
                    device=video.device,
                )

        return video

    if getattr(pipeline, "_wrapped_call_class", None) is not None:
        WrappedPipeline = pipeline._wrapped_call_class
    else:
        WrappedPipeline = type(
            f"{pipeline.__class__.__name__}ActionPatched",
            (pipeline.__class__,),
            {},
        )
        WrappedPipeline.__call__ = __call__
        pipeline._wrapped_call_class = WrappedPipeline
    pipeline.__class__ = WrappedPipeline


def configure_ti2v_text_off_dit(dit):
    dit.use_text_embedding = False
    dit.has_text_input = True
    dit.has_image_input = False
    dit.fuse_vae_embedding_in_latents = True


def load_checkpoint_weights(pipe, ckpt_path: str):
    print(f"Loading training weights from checkpoint: {ckpt_path}")
    state_dict = load_state_dict(ckpt_path, torch_dtype=pipe.torch_dtype, device="cpu")

    dit = pipe.dit
    action_encoder = pipe.action_encoder
    physical_context_encoder = getattr(pipe, "physical_context_encoder", None)
    background_context_encoder = getattr(pipe, "background_context_encoder", None)
    physical_adapter_bank = getattr(pipe, "physical_adapter_bank", None)

    action_prefix = "pipe.action_encoder."
    physical_prefix = "pipe.physical_context_encoder."
    background_prefix = "pipe.background_context_encoder."
    adapter_prefix = "pipe.physical_adapter_bank."
    action_state = {
        key[len(action_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(action_prefix)
    }
    physical_state = {
        key[len(physical_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(physical_prefix)
    }
    background_state = {
        key[len(background_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(background_prefix)
    }
    adapter_state = {
        key[len(adapter_prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(adapter_prefix)
    }
    dit_state = {
        key: value
        for key, value in state_dict.items()
        if (
            not key.startswith(action_prefix)
            and not key.startswith(physical_prefix)
            and not key.startswith(background_prefix)
            and not key.startswith(adapter_prefix)
        )
    }

    dit_result = dit.load_state_dict(dit_state, strict=False)
    print(
        f"  - Loaded dit keys: {len(dit_state)} "
        f"(missing={len(dit_result.missing_keys)}, unexpected={len(dit_result.unexpected_keys)})"
    )

    action_result = action_encoder.load_state_dict(action_state, strict=False)
    print(
        f"  - Loaded action_encoder keys: {len(action_state)} "
        f"(missing={len(action_result.missing_keys)}, unexpected={len(action_result.unexpected_keys)})"
    )
    if physical_context_encoder is not None and physical_state:
        physical_result = physical_context_encoder.load_state_dict(physical_state, strict=False)
        print(
            f"  - Loaded physical_context_encoder keys: {len(physical_state)} "
            f"(missing={len(physical_result.missing_keys)}, unexpected={len(physical_result.unexpected_keys)})"
        )
    if background_context_encoder is not None and background_state:
        background_result = background_context_encoder.load_state_dict(background_state, strict=False)
        print(
            f"  - Loaded background_context_encoder keys: {len(background_state)} "
            f"(missing={len(background_result.missing_keys)}, unexpected={len(background_result.unexpected_keys)})"
        )
    if physical_adapter_bank is not None and adapter_state:
        adapter_result = physical_adapter_bank.load_state_dict(adapter_state, strict=False)
        print(
            f"  - Loaded physical_adapter_bank keys: {len(adapter_state)} "
            f"(missing={len(adapter_result.missing_keys)}, unexpected={len(adapter_result.unexpected_keys)})"
        )


def build_wan_video_action_pipeline(
    torch_dtype: torch.dtype = torch.bfloat16,
    device: Union[str, torch.device] = get_device_type(),
    model_configs: list[ModelConfig] = None,
    tokenizer_config: ModelConfig = None,
    redirect_common_files: bool = True,
    vram_limit: float = None,
    ckpt_path: Optional[str] = None,
    action_dim: int = 14,
    action_mode: str = "adaln",
    physical_context_mode: str = "none",
    physical_context_dim: int = 128,
    physical_context_tokens: int = 1,
    physical_context_hidden_dim: Optional[int] = None,
    physical_context_init_std: float = 0.0,
    physical_context_init_value: float = 0.0,
    physical_context_input_norm: str = "layernorm",
    physical_context_projection: str = "mlp",
    physical_context_temporal_position: str = "none",
    background_context_enabled: bool = False,
    background_context_dim: int = 32,
    background_context_tokens: int = 1,
    background_context_hidden_dim: Optional[int] = None,
    background_context_init_std: float = 0.0,
    background_context_init_value: float = 0.0,
    background_context_input_norm: str = "layernorm",
    background_context_temporal_position: str = "none",
    physical_adapter_mode: str = "none",
    physical_adapter_rank: int = 16,
    physical_adapter_layers: str = "all",
    physical_adapter_gate_init: float = 0.0,
    args: Optional[object] = None,
):
    action_dim = int(_resolve_arg(args, "action_dim", action_dim))
    action_mode = str(_resolve_arg(args, "action_mode", action_mode))
    physical_context_mode = str(_resolve_arg(args, "physical_context_mode", physical_context_mode)).lower()
    physical_context_dim = int(_resolve_arg(args, "physical_context_dim", physical_context_dim))
    physical_context_tokens = int(_resolve_arg(args, "physical_context_tokens", physical_context_tokens))
    physical_context_hidden_dim = _resolve_arg(args, "physical_context_hidden_dim", physical_context_hidden_dim)
    if physical_context_hidden_dim in (0, "0"):
        physical_context_hidden_dim = None
    elif physical_context_hidden_dim is not None:
        physical_context_hidden_dim = int(physical_context_hidden_dim)
    physical_context_init_std = float(_resolve_arg(args, "physical_context_init_std", physical_context_init_std))
    physical_context_init_value = float(_resolve_arg(args, "physical_context_init_value", physical_context_init_value))
    physical_context_input_norm = str(_resolve_arg(args, "physical_context_input_norm", physical_context_input_norm)).lower()
    physical_context_projection = str(_resolve_arg(args, "physical_context_projection", physical_context_projection)).lower()
    physical_context_temporal_position = str(_resolve_arg(args, "physical_context_temporal_position", physical_context_temporal_position)).lower()
    background_context_enabled = bool(_resolve_arg(args, "background_context_enabled", background_context_enabled))
    background_context_dim = int(_resolve_arg(args, "background_context_dim", background_context_dim))
    background_context_tokens = int(_resolve_arg(args, "background_context_tokens", background_context_tokens))
    background_context_hidden_dim = _resolve_arg(args, "background_context_hidden_dim", background_context_hidden_dim)
    if background_context_hidden_dim in (0, "0"):
        background_context_hidden_dim = None
    elif background_context_hidden_dim is not None:
        background_context_hidden_dim = int(background_context_hidden_dim)
    background_context_init_std = float(_resolve_arg(args, "background_context_init_std", background_context_init_std))
    background_context_init_value = float(_resolve_arg(args, "background_context_init_value", background_context_init_value))
    background_context_input_norm = str(_resolve_arg(args, "background_context_input_norm", background_context_input_norm)).lower()
    background_context_temporal_position = str(_resolve_arg(args, "background_context_temporal_position", background_context_temporal_position)).lower()
    physical_adapter_mode = str(_resolve_arg(args, "physical_adapter_mode", physical_adapter_mode)).lower()
    physical_adapter_rank = int(_resolve_arg(args, "physical_adapter_rank", physical_adapter_rank))
    physical_adapter_layers = str(_resolve_arg(args, "physical_adapter_layers", physical_adapter_layers))
    physical_adapter_gate_init = float(_resolve_arg(args, "physical_adapter_gate_init", physical_adapter_gate_init))
    if physical_context_mode not in {"none", "token", "modulation", "both"}:
        raise ValueError(
            "physical_context_mode must be one of none, token, modulation, both; "
            f"got {physical_context_mode}."
        )
    if physical_adapter_mode not in {"none", "residual"}:
        raise ValueError(
            "physical_adapter_mode must be one of none, residual; "
            f"got {physical_adapter_mode}."
        )

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
        redirect_common_files=redirect_common_files,
        vram_limit=vram_limit,
    )
    apply_wan_vae_compat(pipe.vae)

    configure_ti2v_text_off_dit(pipe.dit)

    pipe.action_encoder = WanVideoActionEncoder(
        action_dim=int(action_dim),
        dim=pipe.dit.dim,
        num_action_per_chunk=81,
    )
    pipe.action_encoder = pipe.action_encoder.to(dtype=pipe.torch_dtype, device=pipe.device)
    pipe.action_encoder.eval()

    pipe.physical_context_mode = physical_context_mode
    pipe.physical_context_encoder = None
    if physical_context_mode != "none":
        pipe.physical_context_encoder = PhysicalContextEncoder(
            context_dim=physical_context_dim,
            model_dim=pipe.dit.dim,
            num_tokens=physical_context_tokens,
            hidden_dim=physical_context_hidden_dim,
            init_std=physical_context_init_std,
            init_value=physical_context_init_value,
            input_norm=physical_context_input_norm,
            projection=physical_context_projection,
            temporal_position=physical_context_temporal_position,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.physical_context_encoder.eval()

    pipe.background_context_enabled = background_context_enabled
    pipe.background_context_encoder = None
    if background_context_enabled:
        pipe.background_context_encoder = PhysicalContextEncoder(
            context_dim=background_context_dim,
            model_dim=pipe.dit.dim,
            num_tokens=background_context_tokens,
            hidden_dim=background_context_hidden_dim,
            init_std=background_context_init_std,
            init_value=background_context_init_value,
            input_norm=background_context_input_norm,
            temporal_position=background_context_temporal_position,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.background_context_encoder.eval()

    pipe.physical_adapter_mode = physical_adapter_mode
    pipe.physical_adapter_bank = None
    if physical_adapter_mode == "residual":
        adapter_layers = _parse_adapter_layers(physical_adapter_layers, len(pipe.dit.blocks))
        pipe.physical_adapter_bank = PhysicalResidualAdapterBank(
            dim=pipe.dit.dim,
            num_layers=len(pipe.dit.blocks),
            rank=physical_adapter_rank,
            layers=adapter_layers,
            gate_init=physical_adapter_gate_init,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        pipe.physical_adapter_bank.eval()
        pipe.in_iteration_models = tuple(pipe.in_iteration_models) + ("physical_adapter_bank",)

    if ckpt_path is not None:
        load_checkpoint_weights(pipe, ckpt_path)

    pipe.units = _build_wan2_action_units(pipe)
    pipe.action_injection_mode = action_mode
    _install_wan_video_action_call(pipe)

    pipe.model_fn = model_fn_wan_video_action
    return pipe


class WanVideoUnit_ActionEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("action", "num_frames"),
            output_params=("action_emb", "action_mod_emb"),
            onload_model_names=("action_encoder",)
        )

    def process(self, pipe, action=None, num_frames=None):
        if action is None:
            return {}
        if pipe.action_encoder is None:
            raise ValueError("Action encoder is not available in the pipeline.")
        if any(param.device != pipe.device for param in pipe.action_encoder.parameters()):
            pipe.action_encoder = pipe.action_encoder.to(device=pipe.device, dtype=pipe.torch_dtype)

        pipe.load_models_to_device(self.onload_model_names)
        action = torch.as_tensor(action, device=pipe.device, dtype=pipe.torch_dtype)

        target_groups = (int(num_frames) - 1) // 4 + 1
        target_action_frames = 1 + 4 * (target_groups - 1)
        current_action_frames = int(action.shape[1])
        if current_action_frames > target_action_frames:
            action = action[:, :target_action_frames]
        elif current_action_frames < target_action_frames:
            raise ValueError(
                f"Action sequence too short for latent groups: action_frames={current_action_frames}, "
                f"required={target_action_frames}, target_groups={target_groups}"
            )
        action_emb, action_mod_emb = pipe.action_encoder.encode_ti2v2(action)
        return {"action_emb": action_emb, "action_mod_emb": action_mod_emb}


class WanVideoUnit_PhysicalContextEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("physical_context", "action", "num_frames"),
            output_params=("physical_context_emb", "physical_mod_emb"),
            onload_model_names=("physical_context_encoder",),
        )

    def process(self, pipe, physical_context=None, action=None, num_frames=None):
        mode = getattr(pipe, "physical_context_mode", "none")
        if mode == "none":
            return {}
        if pipe.physical_context_encoder is None:
            raise ValueError("physical_context_mode is enabled, but physical_context_encoder is not available.")
        if any(param.device != pipe.device for param in pipe.physical_context_encoder.parameters()):
            pipe.physical_context_encoder = pipe.physical_context_encoder.to(device=pipe.device, dtype=pipe.torch_dtype)

        pipe.load_models_to_device(self.onload_model_names)
        if action is not None:
            batch_size = int(torch.as_tensor(action).shape[0])
        elif physical_context is not None:
            context_tensor = torch.as_tensor(physical_context)
            batch_size = 1 if context_tensor.ndim <= 2 else int(context_tensor.shape[0])
        else:
            batch_size = 1

        target_groups = (int(num_frames) - 1) // 4 + 1
        token_emb, mod_emb = pipe.physical_context_encoder(
            physical_context,
            batch_size=batch_size,
            target_groups=target_groups,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        outputs = {}
        if mode in {"token", "both"}:
            outputs["physical_context_emb"] = token_emb
        if mode in {"modulation", "both"}:
            outputs["physical_mod_emb"] = mod_emb
        return outputs


class WanVideoUnit_BackgroundContextEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("background_context", "action", "num_frames"),
            output_params=("background_context_emb", "background_mod_emb"),
            onload_model_names=("background_context_encoder",),
        )

    def process(self, pipe, background_context=None, action=None, num_frames=None):
        if not getattr(pipe, "background_context_enabled", False):
            return {}
        if pipe.background_context_encoder is None:
            raise ValueError("background_context is enabled, but background_context_encoder is not available.")
        if any(param.device != pipe.device for param in pipe.background_context_encoder.parameters()):
            pipe.background_context_encoder = pipe.background_context_encoder.to(
                device=pipe.device, dtype=pipe.torch_dtype
            )

        pipe.load_models_to_device(self.onload_model_names)
        if action is not None:
            batch_size = int(torch.as_tensor(action).shape[0])
        elif background_context is not None:
            context_tensor = torch.as_tensor(background_context)
            batch_size = 1 if context_tensor.ndim <= 2 else int(context_tensor.shape[0])
        else:
            batch_size = 1

        target_groups = (int(num_frames) - 1) // 4 + 1
        token_emb, mod_emb = pipe.background_context_encoder(
            background_context,
            batch_size=batch_size,
            target_groups=target_groups,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        return {
            "background_context_emb": token_emb,
            "background_mod_emb": mod_emb,
        }


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    """
    Input frame embedder aligned to target history-conditioning behavior:
    - For short input (<=1 or < num_frames), skip VAE-conditioned noise injection
      and use pure noise as latents.
    - Otherwise encode and add scheduler initial noise as usual.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "precomputed_latents", "noise", "num_frames", "tiled", "tile_size", "tile_stride"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",),
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        precomputed_latents,
        noise,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if precomputed_latents is not None:
            input_latents_views = precomputed_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = rearrange(
                input_latents_views,
                "v c t h w -> 1 c t (v h) w",
            )
            if pipe.scheduler.training:
                return {"latents": noise, "input_latents": input_latents}
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}

        if input_video is None:
            return {"latents": noise}

        if int(input_video.shape[2]) <= 1 or (not pipe.scheduler.training and int(input_video.shape[2]) < int(num_frames)):
            return {"latents": noise}

        pipe.load_models_to_device(self.onload_model_names)
        input_video = input_video.to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents_views = pipe.vae.encode(
            input_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        input_latents = rearrange(input_latents_views, "v c t h w -> 1 c t (v h) w")

        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
        return {"latents": latents, "input_latents": input_latents}


def model_fn_wan_video_action(
    dit,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    action_emb: Optional[torch.Tensor] = None,
    action_mod_emb: Optional[torch.Tensor] = None,
    action_injection_mode: str = "none",
    physical_context_emb: Optional[torch.Tensor] = None,
    physical_mod_emb: Optional[torch.Tensor] = None,
    background_context_emb: Optional[torch.Tensor] = None,
    background_mod_emb: Optional[torch.Tensor] = None,
    extra_condition_tokens: Optional[torch.Tensor] = None,
    physical_context_mode: str = "none",
    physical_adapter_bank: Optional[PhysicalResidualAdapterBank] = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    fuse_vae_embedding_in_latents: bool = False,
    fused_condition_latent_frames: Optional[int] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        condition_t = 1 if fused_condition_latent_frames is None else int(fused_condition_latent_frames)
        condition_t = max(0, min(condition_t, latents.shape[2]))
        spatial_token_count = latents.shape[3] * latents.shape[4] // 4
        if timestep.numel() > 1:
            if timestep.numel() != latents.shape[0]:
                raise ValueError("Independent timesteps must match the latent batch size.")
            token_times = timestep.reshape(-1, 1, 1).expand(
                -1, latents.shape[2], spatial_token_count
            ).clone()
            token_times[:, :condition_t] = 0
            t = dit.time_embedding(
                sinusoidal_embedding_1d(dit.freq_dim, token_times.flatten()).reshape(
                    latents.shape[0], -1, dit.freq_dim
                )
            )
        else:
            t = torch.concat(
            [
                torch.zeros((condition_t, spatial_token_count), dtype=latents.dtype, device=latents.device),
                torch.ones((latents.shape[2] - condition_t, spatial_token_count), dtype=latents.dtype, device=latents.device) * timestep,
            ]
            ).flatten()
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, t).unsqueeze(0))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))

    text_token_count = 0
    use_text_embedding = getattr(dit, "use_text_embedding", getattr(dit, "has_text_input", True))
    has_text_input = getattr(dit, "has_text_input", True)

    if use_text_embedding and context is not None:
        context = dit.text_embedding(context)
        text_token_count = context.shape[1]
    elif not has_text_input:
        context = None
    elif not use_text_embedding:
        context = None

    if action_emb is None or action_mod_emb is None:
        raise ValueError("`action:adaln` requires both `action_emb` and `action_mod_emb`.")
    context = _append_context_tokens(context, physical_context_emb)
    context = _append_context_tokens(context, background_context_emb)
    if extra_condition_tokens is not None:
        if extra_condition_tokens.ndim != 3:
            raise ValueError(
                "extra_condition_tokens must have shape [batch, tokens, dim], "
                f"got {tuple(extra_condition_tokens.shape)}."
            )
        if extra_condition_tokens.shape[0] != latents.shape[0]:
            raise ValueError(
                "extra_condition_tokens batch size must match latent batch size, "
                f"got tokens={extra_condition_tokens.shape[0]} and latents={latents.shape[0]}."
            )
        context = _append_context_tokens(context, extra_condition_tokens)
    context = _append_context_tokens(context, action_emb)
    text_token_count = context.shape[1]
    if t.shape[1] % action_mod_emb.shape[1] != 0:
        raise RuntimeError(
            f"Temporal group mismatch: t.shape={tuple(t.shape)}, action_mod_emb.shape={tuple(action_mod_emb.shape)}. "
            "Expected t.shape[1] to be divisible by action_mod_emb.shape[1]."
        )
    target_mod_groups = action_mod_emb.shape[1]
    num_spatial_tokens = t.shape[1] // target_mod_groups
    action_mod_emb = action_mod_emb.unsqueeze(2).repeat(1, 1, num_spatial_tokens, 1).flatten(1, 2)
    t = t + action_mod_emb
    physical_adapter_condition = None
    combined_physical_mod_emb = None
    for mod_name, mod_emb in (
        ("physical_mod_emb", physical_mod_emb),
        ("background_mod_emb", background_mod_emb),
    ):
        if mod_emb is None:
            continue
        if mod_emb.shape[1] != target_mod_groups:
            raise RuntimeError(
                f"Physical temporal group mismatch: {mod_name}.shape={tuple(mod_emb.shape)}, "
                f"expected_groups={target_mod_groups}."
            )
        combined_physical_mod_emb = (
            mod_emb if combined_physical_mod_emb is None else combined_physical_mod_emb + mod_emb
        )
    if combined_physical_mod_emb is not None:
        physical_adapter_condition = combined_physical_mod_emb.mean(dim=1)
        combined_physical_mod_emb = (
            combined_physical_mod_emb.unsqueeze(2)
            .repeat(1, 1, num_spatial_tokens, 1)
            .flatten(1, 2)
        )
        t = t + combined_physical_mod_emb

    if t.ndim == 3:
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    x = latents

    if y is not None and dit.has_image_input and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    if clip_feature is not None and dit.has_image_input and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        if context is None:
            context = clip_embdding
        else:
            context = torch.cat([clip_embdding, context], dim=1)

    x = dit.patchify(x)
    f, h, w = x.shape[2:]

    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()

    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for layer_idx, block in enumerate(dit.blocks):
        if hasattr(block, "cross_attn") and hasattr(block.cross_attn, "text_token_count"):
            block.cross_attn.text_token_count = text_token_count
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)
        if physical_adapter_bank is not None:
            x = physical_adapter_bank(layer_idx, x, conditioning=physical_adapter_condition)

    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))

    return x


class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode the conditioning frame directly into latents for Wan2.2 TI2V.
    """
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "precomputed_latents",
                "input_latents",
                "latents",
                "num_history_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            output_params=(
                "latents",
                "fuse_vae_embedding_in_latents",
                "first_frame_latents",
                "fused_condition_latent_frames",
            ),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        input_video,
        precomputed_latents,
        input_latents,
        latents,
        num_history_frames,
        tiled,
        tile_size,
        tile_stride,
    ):
        if not getattr(pipe.dit, "fuse_vae_embedding_in_latents", False):
            return {}

        if precomputed_latents is not None:
            target_history = max(1, (int(num_history_frames) - 1) // 4 + 1)
            z = input_latents[:, :, :target_history].clone()
            history_t = z.shape[2]
            latents[:, :, :history_t] = z
            return {
                "latents": latents,
                "fuse_vae_embedding_in_latents": True,
                "first_frame_latents": z,
                "fused_condition_latent_frames": int(history_t),
            }

        if input_video is None:
            return {}

        num_history_frames = int(num_history_frames)
        if num_history_frames <= 0:
            raise ValueError("`input_video` must include at least one history frame.")
        if input_video.shape[2] < num_history_frames:
            raise ValueError(
                f"`num_history_frames` ({num_history_frames}) exceeds input video frames ({input_video.shape[2]})."
            )

        pipe.load_models_to_device(self.onload_model_names)
        history_frames = input_video[:, :, :num_history_frames]
        first_frame_latents = history_frames.to(dtype=pipe.torch_dtype, device=pipe.device)
        z_views = pipe.vae.encode(
            first_frame_latents,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        z_views = z_views.to(dtype=pipe.torch_dtype, device=pipe.device)
        z = rearrange(z_views, "v c t h w -> 1 c t (v h) w")

        history_t = z.shape[2]
        latents[:, :, :history_t] = z

        return {
            "latents": latents,
            "fuse_vae_embedding_in_latents": True,
            "first_frame_latents": z,
            "fused_condition_latent_frames": int(history_t),
        }
