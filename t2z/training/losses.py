import torch
import torch.nn.functional as F


def flow_matching_loss(pipe, inputs_shared, inputs_posi, inputs_nega, timestep_index=None):
    inputs = {**inputs_shared, **inputs_posi}
    if 'lora' in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs['lora'], hotload=True, verbose=0)
    if timestep_index is None:
        upper = int(inputs.get('max_timestep_boundary', 1) * len(pipe.scheduler.timesteps))
        lower = int(inputs.get('min_timestep_boundary', 0) * len(pipe.scheduler.timesteps))
        timestep_index = torch.randint(lower, upper, (1,))
    timestep = pipe.scheduler.timesteps[timestep_index].reshape(1).to(dtype=pipe.torch_dtype, device=pipe.device)
    clean = inputs['input_latents']
    noise = torch.randn_like(clean)
    inputs['latents'] = pipe.scheduler.add_noise(clean, noise, timestep)
    target = pipe.scheduler.training_target(clean, noise, timestep)
    condition = inputs.get('first_frame_latents')
    observed = (int(inputs['num_history_frames']) - 1) // 4 + 1
    if condition is None or condition.shape[2] != observed:
        raise ValueError('Observed video and conditioning latents are not aligned')
    if not 0 < observed < clean.shape[2]:
        raise ValueError('The loss requires future latent frames after the observed prefix')
    inputs['latents'][:, :, :observed] = condition
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    prediction = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = F.mse_loss(prediction[:, :, observed:].float(), target[:, :, observed:].float())
    return loss * pipe.scheduler.training_weight(timestep)
