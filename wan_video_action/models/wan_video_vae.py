import torch
from einops import rearrange


def _check_is_instance(vae_mod, module, module_class):
    checker = getattr(vae_mod, "check_is_instance", None)
    if callable(checker):
        return checker(module, module_class)
    return isinstance(module, module_class)


def _scale_latents(mu, scale, z_dim):
    if isinstance(scale[0], torch.Tensor):
        scale = [s.to(dtype=mu.dtype, device=mu.device) for s in scale]
        return (mu - scale[0].view(1, z_dim, 1, 1, 1)) * scale[1].view(1, z_dim, 1, 1, 1)
    scale = scale.to(dtype=mu.dtype, device=mu.device)
    return (mu - scale[0]) * scale[1]


def _unscale_latents(z, scale, z_dim):
    if isinstance(scale[0], torch.Tensor):
        scale = [s.to(dtype=z.dtype, device=z.device) for s in scale]
        return z / scale[1].view(1, z_dim, 1, 1, 1) + scale[0].view(1, z_dim, 1, 1, 1)
    scale = scale.to(dtype=z.dtype, device=z.device)
    return z / scale[1] + scale[0]


def _resample_forward(self, x, feat_cache=None, feat_idx=[0]):
    import diffsynth.models.wan_video_vae as vae_mod

    b, c, t, h, w = x.size()
    if self.mode == "upsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                    cache_x = torch.cat(
                        [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                        dim=2,
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)

    t = x.shape[2]
    x = rearrange(x, "b c t h w -> (b t) c h w")
    x = self.resample(x)
    x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

    if self.mode == "downsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
    return x


def _residual_block_forward(self, x, feat_cache=None, feat_idx=[0]):
    import diffsynth.models.wan_video_vae as vae_mod

    h = self.shortcut(x)
    for layer in self.residual:
        if _check_is_instance(vae_mod, layer, vae_mod.CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x + h


def _down_residual_block_forward(self, x, feat_cache=None, feat_idx=[0]):
    x_copy = x.clone()
    for module in self.downsamples:
        x = module(x, feat_cache, feat_idx)
    return x + self.avg_shortcut(x_copy)


def _up_residual_block_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    x_main = x.clone()
    for module in self.upsamples:
        x_main = module(x_main, feat_cache, feat_idx)
    if self.avg_shortcut is not None:
        x_shortcut = self.avg_shortcut(x, first_chunk)
        return x_main + x_shortcut
    return x_main


def _encoder3d_forward(self, x, feat_cache=None, feat_idx=[0]):
    import diffsynth.models.wan_video_vae as vae_mod

    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat(
                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    for layer in self.downsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.middle:
        if _check_is_instance(vae_mod, layer, vae_mod.ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.head:
        if _check_is_instance(vae_mod, layer, vae_mod.CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


def _encoder3d_38_forward(self, x, feat_cache=None, feat_idx=[0]):
    import diffsynth.models.wan_video_vae as vae_mod

    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat(
                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    for layer in self.downsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.middle:
        if _check_is_instance(vae_mod, layer, vae_mod.ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.head:
        if _check_is_instance(vae_mod, layer, vae_mod.CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


def _decoder3d_forward(self, x, feat_cache=None, feat_idx=[0]):
    import diffsynth.models.wan_video_vae as vae_mod

    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat(
                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    for layer in self.middle:
        if _check_is_instance(vae_mod, layer, vae_mod.ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.upsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.head:
        if _check_is_instance(vae_mod, layer, vae_mod.CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


def _decoder3d_38_forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False):
    import diffsynth.models.wan_video_vae as vae_mod

    if feat_cache is not None:
        idx = feat_idx[0]
        cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
        if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
            cache_x = torch.cat(
                [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                dim=2,
            )
        x = self.conv1(x, feat_cache[idx])
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    else:
        x = self.conv1(x)

    for layer in self.middle:
        if _check_is_instance(vae_mod, layer, vae_mod.ResidualBlock) and feat_cache is not None:
            x = layer(x, feat_cache, feat_idx)
        else:
            x = layer(x)

    for layer in self.upsamples:
        if feat_cache is not None:
            x = layer(x, feat_cache, feat_idx, first_chunk)
        else:
            x = layer(x)

    for layer in self.head:
        if _check_is_instance(vae_mod, layer, vae_mod.CausalConv3d) and feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -vae_mod.CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                    dim=2,
                )
            x = layer(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = layer(x)
    return x


def _video_vae_encode(self, x, scale):
    self.clear_cache()
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4

    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
        else:
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)
    mu, _ = self.conv1(out).chunk(2, dim=1)
    return _scale_latents(mu, scale, self.z_dim)


def _video_vae_decode(self, z, scale):
    self.clear_cache()
    z = _unscale_latents(z, scale, self.z_dim)
    iter_ = z.shape[2]
    x = self.conv2(z)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
        else:
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
            out = torch.cat([out, out_], 2)
    return out


def _video_vae38_encode(self, x, scale):
    import diffsynth.models.wan_video_vae as vae_mod

    self.clear_cache()
    x = vae_mod.patchify(x, patch_size=2)
    t = x.shape[2]
    iter_ = 1 + (t - 1) // 4
    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(
                x[:, :, :1, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
        else:
            out_ = self.encoder(
                x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)
    mu, _ = self.conv1(out).chunk(2, dim=1)
    mu = _scale_latents(mu, scale, self.z_dim)
    self.clear_cache()
    return mu


def _video_vae38_decode(self, z, scale):
    import diffsynth.models.wan_video_vae as vae_mod

    self.clear_cache()
    z = _unscale_latents(z, scale, self.z_dim)
    iter_ = z.shape[2]
    x = self.conv2(z)
    for i in range(iter_):
        self._conv_idx = [0]
        if i == 0:
            out = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=True,
            )
        else:
            out_ = self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
            )
            out = torch.cat([out, out_], 2)
    out = vae_mod.unpatchify(out, patch_size=2)
    self.clear_cache()
    return out


def apply_wan_vae_compat(vae) -> None:
    if vae is None:
        return

    import diffsynth.models.wan_video_vae as vae_mod

    if not getattr(vae_mod, "_wm_target_cache_compat_applied", False):
        vae_mod.Resample.forward = _resample_forward
        if hasattr(vae_mod, "Resample38"):
            vae_mod.Resample38.forward = _resample_forward
        vae_mod.ResidualBlock.forward = _residual_block_forward
        if hasattr(vae_mod, "Down_ResidualBlock"):
            vae_mod.Down_ResidualBlock.forward = _down_residual_block_forward
        if hasattr(vae_mod, "Up_ResidualBlock"):
            vae_mod.Up_ResidualBlock.forward = _up_residual_block_forward
        vae_mod.Encoder3d.forward = _encoder3d_forward
        if hasattr(vae_mod, "Encoder3d_38"):
            vae_mod.Encoder3d_38.forward = _encoder3d_38_forward
        vae_mod.Decoder3d.forward = _decoder3d_forward
        if hasattr(vae_mod, "Decoder3d_38"):
            vae_mod.Decoder3d_38.forward = _decoder3d_38_forward
        vae_mod.VideoVAE_.encode = _video_vae_encode
        vae_mod.VideoVAE_.decode = _video_vae_decode
        if hasattr(vae_mod, "VideoVAE38_"):
            vae_mod.VideoVAE38_.encode = _video_vae38_encode
            vae_mod.VideoVAE38_.decode = _video_vae38_decode
        setattr(vae_mod, "_wm_target_cache_compat_applied", True)

    setattr(vae, "_wm_target_cache_compat_applied", True)
    model = getattr(vae, "model", None)
    if model is not None:
        setattr(model, "_wm_target_cache_compat_applied", True)


__all__ = ["apply_wan_vae_compat"]
