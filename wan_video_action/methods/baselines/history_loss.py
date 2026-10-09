"""Observed-history flow matching for multi-frame conditioning."""

import torch


def history_prefix_flow_loss(pipe, inputs_shared, inputs_posi, inputs_nega):
    inputs = {**inputs_shared, **inputs_posi}
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)
    upper = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    lower = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    index = torch.randint(lower, upper, (1,))
    timestep = pipe.scheduler.timesteps[index].to(dtype=pipe.torch_dtype, device=pipe.device)
    clean = inputs["input_latents"]
    noise = torch.randn_like(clean)
    inputs["latents"] = pipe.scheduler.add_noise(clean, noise, timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)
    condition = inputs.get("first_frame_latents")
    expected = (int(inputs["num_history_frames"]) - 1) // 4 + 1
    if condition is None or condition.shape[2] != expected:
        raise ValueError("History VAE latents do not match the requested condition length")
    if not 0 < expected < clean.shape[2]:
        raise ValueError("A history-conditioned batch must contain future latent frames")
    inputs["latents"][:, :, :expected] = condition
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    predicted = pipe.model_fn(**models, **inputs, timestep=timestep)


    loss = torch.nn.functional.mse_loss(
        predicted[:, :, expected:].float(), target[:, :, expected:].float(),
    )
    return loss * pipe.scheduler.training_weight(timestep)
