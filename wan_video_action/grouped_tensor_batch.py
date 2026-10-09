"""Batched DiT loss for fixed-geometry, single-view grouped Stage1 clips."""

import torch


def grouped_tensor_batch_loss(model, samples):
    if not samples:
        raise ValueError("A tensor microbatch must contain at least one sample.")
    prepared = []
    targets, timesteps, weights = [], [], []
    scheduler = model.pipe.scheduler
    for sample in samples:
        if any(key in sample for key in ("_self_correction_donor_data", "_flow_timestep_index")):
            raise ValueError("Tensor batching does not support self-correction inputs.")
        shared, positive, _ = model._prepare_pipeline_inputs(sample)
        inputs = {**shared, **positive}
        if int(inputs.get("num_views", 1)) != 1:
            raise ValueError("Grouped tensor batching currently requires one camera view.")
        clean = inputs["input_latents"]
        if clean.shape[0] != 1:
            raise ValueError("Each prepared clip must have batch dimension one.")
        lower = int(inputs.get("min_timestep_boundary", 0) * len(scheduler.timesteps))
        upper = int(inputs.get("max_timestep_boundary", 1) * len(scheduler.timesteps))
        timestep_id = torch.randint(lower, upper, (1,))
        timestep = scheduler.timesteps[timestep_id].to(
            device=clean.device, dtype=model.pipe.torch_dtype
        )
        noise = torch.randn_like(clean)
        inputs["latents"] = scheduler.add_noise(clean, noise, timestep)
        targets.append(scheduler.training_target(clean, noise, timestep))
        if "first_frame_latents" in inputs:
            inputs["latents"][:, :, 0:1] = inputs["first_frame_latents"]
        prepared.append(inputs)
        timesteps.append(timestep.reshape(1))
        weights.append(scheduler.training_weight(timestep).reshape(()))

    merged = prepared[0].copy()
    for key in (
        "latents", "context", "action_emb", "action_mod_emb",
        "physical_context_emb", "physical_mod_emb", "background_context_emb",
        "background_mod_emb", "extra_condition_tokens", "clip_feature", "y",
    ):
        values = [inputs.get(key) for inputs in prepared]
        if all(value is None for value in values):
            continue
        if not all(torch.is_tensor(value) for value in values):
            raise ValueError(f"Inconsistent tensor microbatch input: {key}")
        if any(value.shape[1:] != values[0].shape[1:] for value in values):
            raise ValueError(f"Tensor microbatch requires matching {key} geometry.")
        merged[key] = torch.cat(values, dim=0)
    if any(("first_frame_latents" in p) != ("first_frame_latents" in merged) for p in prepared):
        raise ValueError("Mixed first-frame conditioning in a tensor microbatch.")
    models = {name: getattr(model.pipe, name) for name in model.pipe.in_iteration_models}
    prediction = model.pipe.model_fn(
        **models, **merged, timestep=torch.cat(timesteps)
    )
    target = torch.cat(targets, dim=0)
    if "first_frame_latents" in merged:
        prediction, target = prediction[:, :, 1:], target[:, :, 1:]


    losses = [
        model.spatial_flow_mse(prediction[i:i + 1], target[i:i + 1]) * weights[i]
        for i in range(len(samples))
    ]
    return torch.stack(losses).mean()
