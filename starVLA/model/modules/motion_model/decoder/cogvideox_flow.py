import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from einops import rearrange

from .cogvideox_transformer_3d import CogVideoXTransformer3DModel

logger = logging.getLogger(__name__)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    scale_factor_spatial: int,
    num_frames: int,
    transformer_config,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare rotary positional embeddings matching diffusers implementation."""

    grid_height = height // (scale_factor_spatial * transformer_config.patch_size)
    grid_width = width // (scale_factor_spatial * transformer_config.patch_size)

    p = transformer_config.patch_size
    p_t = getattr(transformer_config, "patch_size_t", None)

    base_size_width = transformer_config.sample_width // p
    base_size_height = transformer_config.sample_height // p

    if p_t is None:
        grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            device=device,
        )
    else:
        base_num_frames = (num_frames + p_t - 1) // p_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
            device=device,
        )

    return freqs_cos, freqs_sin


class CogVideoXDecoder_flow(nn.Module):
    """Checkpoint-compatible CogVideoX Motion Expert decoder."""

    def __init__(
        self,
        latent_dim: int = 768,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int = 16,
        num_layers: int = 30,
        num_frames: int = 13,
        frame_size: int = 32,
        patch_size: int = 2,
        patch_size_t: int = 4,
        max_text_seq_length: int = 226,
        text_embed_dim: int = 4096,
        use_rotary_positional_embeddings: bool = True,
        enable_encoder_hidden_states_grad: bool = True,
        scale_factor_spatial: int = 8,
        put_frames_in_channels: int = 1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.frame_size = frame_size
        self.scale_factor_spatial = scale_factor_spatial
        assert num_frames % put_frames_in_channels == 0, "num_frames must be divisible by put_frames_in_channels"
        self.put_frames_in_channels = put_frames_in_channels
        self.input_channels = in_channels * put_frames_in_channels
        self.output_channels = out_channels * put_frames_in_channels
        self.num_frames = num_frames // put_frames_in_channels

        self._setup_custom_model(
            num_attention_heads,
            attention_head_dim,
            self.input_channels,
            self.output_channels,
            num_layers,
            frame_size,
            patch_size,
            patch_size_t,
            max_text_seq_length,
            text_embed_dim,
            use_rotary_positional_embeddings,
            enable_encoder_hidden_states_grad,
        )

        if latent_dim != text_embed_dim:
            self.trunk_to_text_proj = nn.Linear(latent_dim, text_embed_dim)

            with torch.no_grad():
                nn.init.normal_(self.trunk_to_text_proj.weight, mean=0.0, std=0.01)
                if self.trunk_to_text_proj.bias is not None:
                    nn.init.zeros_(self.trunk_to_text_proj.bias)
        else:
            self.trunk_to_text_proj = nn.Identity()

        # Learned padding avoids overwhelming repeated conditioning with zeros.
        self.learnable_padding_tokens = nn.Parameter(torch.randn(1, max_text_seq_length, text_embed_dim) * 0.01)

    def _setup_custom_model(
        self,
        num_attention_heads,
        attention_head_dim,
        in_channels,
        out_channels,
        num_layers,
        frame_size,
        patch_size,
        patch_size_t,
        max_text_seq_length,
        text_embed_dim,
        use_rotary_positional_embeddings,
        enable_encoder_hidden_states_grad,
    ):
        self.cogvideox = CogVideoXTransformer3DModel(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            time_embed_dim=text_embed_dim,
            num_layers=num_layers,
            sample_height=frame_size,
            sample_width=frame_size,
            sample_frames=self.num_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            max_text_seq_length=max_text_seq_length,
            text_embed_dim=text_embed_dim,
            use_rotary_positional_embeddings=use_rotary_positional_embeddings,
            enable_encoder_hidden_states_grad=enable_encoder_hidden_states_grad,
        )

    def unnormalize_act_data(self, data):
        data = rearrange(data, "b t c h w -> b h w t c")
        device = data.device
        data = data * self.data_act_scale.to(device) + self.data_act_bias.to(device)
        return rearrange(data, "b h w t c -> b t c h w")

    def set_data_act_statistics(self, maximum, minimum):
        self.data_act_scale = (maximum - minimum) / 2
        self.data_act_bias = (maximum + minimum) / 2

    def project_latents_to_cogvideox_format(self, latents: torch.Tensor):
        encoder_hidden_states = self.trunk_to_text_proj(latents).to(self.cogvideox.dtype)

        if encoder_hidden_states.shape[1] != self.cogvideox.config.max_text_seq_length:
            current_seq_len = encoder_hidden_states.shape[1]
            target_seq_len = self.cogvideox.config.max_text_seq_length

            if current_seq_len < target_seq_len:
                # Repeat conditioning, then fill the remainder with learned padding.
                batch_size = encoder_hidden_states.shape[0]
                repeat_factor = target_seq_len // current_seq_len
                encoder_hidden_states = encoder_hidden_states.repeat(1, repeat_factor, 1)
                num_padding = target_seq_len - encoder_hidden_states.shape[1]
                padding_tokens = (
                    self.learnable_padding_tokens[:, :num_padding, :]
                    .expand(batch_size, num_padding, -1)
                    .to(encoder_hidden_states.device, encoder_hidden_states.dtype)
                )

                encoder_hidden_states = torch.cat([encoder_hidden_states, padding_tokens], dim=1)
            else:
                encoder_hidden_states = encoder_hidden_states[:, :target_seq_len]

        return encoder_hidden_states

    def forward(
        self,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        trunk_conditioning: torch.Tensor,
        return_hidden_states: bool = False,
    ):
        """Predict motion velocity and optionally expose guidance features.

        Args:
            noisy_latents: Noisy motion latents with shape ``[B, C, T, H, W]``.
            timesteps: Flow timesteps with shape ``[B]``.
            trunk_conditioning: VLM features with shape ``[B, sequence, latent_dim]``.
            return_hidden_states: Return per-layer motion features for Motion Guidance.

        Returns:
            The predicted velocity and, when requested, per-layer motion features.
        """
        batch_size, channels, num_frames, height, width = noisy_latents.shape
        device = noisy_latents.device

        if hasattr(self, "put_frames_in_channels") and self.put_frames_in_channels > 1:
            # Check whether adjacent frames are already packed in channels.
            expected_converted_channels = 3 * self.put_frames_in_channels

            if channels == expected_converted_channels:
                hidden_states = noisy_latents
            else:
                logger.warning(
                    "expected %d packed input channels; converting an unpacked input",
                    expected_converted_channels,
                )
                if num_frames % self.put_frames_in_channels:
                    raise ValueError(
                        f"num_frames ({num_frames}) must be divisible by "
                        f"put_frames_in_channels ({self.put_frames_in_channels})"
                    )

                # Group adjacent frames before packing them in the channel axis.
                grouped_frames = noisy_latents.view(
                    batch_size,
                    channels,
                    num_frames // self.put_frames_in_channels,
                    self.put_frames_in_channels,
                    height,
                    width,
                )

                hidden_states = (
                    grouped_frames.permute(0, 1, 3, 2, 4, 5)
                    .contiguous()
                    .view(
                        batch_size,
                        channels * self.put_frames_in_channels,
                        num_frames // self.put_frames_in_channels,
                        height,
                        width,
                    )
                )
        else:
            hidden_states = noisy_latents

        # CogVideoX consumes video latents in [B, T, C, H, W] order.
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)

        encoder_hidden_states = self.project_latents_to_cogvideox_format(trunk_conditioning)

        image_rotary_emb = None
        if (
            hasattr(self.cogvideox.config, "use_rotary_positional_embeddings")
            and self.cogvideox.config.use_rotary_positional_embeddings
        ):
            if (
                hasattr(self.cogvideox.config, "patch_size_t")
                and self.cogvideox.config.patch_size_t is not None
                and self.cogvideox.config.patch_size_t > 1
            ):
                image_rotary_emb = prepare_rotary_positional_embeddings(
                    height=height * self.scale_factor_spatial,
                    width=width * self.scale_factor_spatial,
                    scale_factor_spatial=self.scale_factor_spatial,
                    num_frames=num_frames,
                    device=device,
                    transformer_config=self.cogvideox.config,
                )
            else:
                image_rotary_emb = prepare_rotary_positional_embeddings(
                    height=height * self.scale_factor_spatial,
                    width=width * self.scale_factor_spatial,
                    scale_factor_spatial=self.scale_factor_spatial,
                    num_frames=num_frames,
                    device=device,
                    transformer_config=self.cogvideox.config,
                )

        output = self.cogvideox(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
            return_hidden_states=return_hidden_states,
        )

        if return_hidden_states:
            model_output, all_hidden_states = output
        else:
            model_output = output[0]
            all_hidden_states = None

        # Restore the public [B, C, T, H, W] motion layout.
        model_output = model_output.permute(0, 2, 1, 3, 4)
        if hasattr(self, "put_frames_in_channels") and self.put_frames_in_channels > 1:
            batch_size_out, channels_out, num_frames_out, height_out, width_out = model_output.shape

            if channels_out == 3 * self.put_frames_in_channels:
                # Split packed channels before restoring the temporal axis.
                model_output = model_output.view(
                    batch_size_out,
                    channels_out // self.put_frames_in_channels,
                    self.put_frames_in_channels,
                    num_frames_out,
                    height_out,
                    width_out,
                )

                model_output = (
                    model_output.permute(0, 1, 3, 2, 4, 5)
                    .contiguous()
                    .view(
                        batch_size_out,
                        channels_out // self.put_frames_in_channels,
                        num_frames_out * self.put_frames_in_channels,
                        height_out,
                        width_out,
                    )
                )

        result = {"video": model_output}
        if return_hidden_states:
            result["hidden"] = all_hidden_states
        return result

    def predict_partial_with_tau(
        self,
        trunk_conditioning: torch.Tensor,
        tau_v: float,
        num_inference_steps: int = 10,
        generator: Optional[torch.Generator] = None,
    ) -> Dict[str, torch.Tensor]:
        """Integrate to ``tau_v`` and return motion features for guidance.

        The method performs explicit Euler updates from noise at ``t=0`` to
        ``t=tau_v``, then evaluates the Motion Expert once more at ``tau_v``
        to collect its hidden states.

        Args:
            trunk_conditioning: VLM features with shape ``[B, sequence, latent_dim]``.
            tau_v: Target flow time in ``[0, 1]``, where zero is noise.
            num_inference_steps: Number of Euler steps for a complete trajectory.
            generator: Optional random generator for deterministic sampling.

        Returns:
            Partial motion with shape ``[B, K, T, D]``, per-layer hidden
            states, and the applied ``tau_v``.
        """
        prompt_embeds = self.project_latents_to_cogvideox_format(trunk_conditioning).to(self.cogvideox.dtype)
        batch_size = prompt_embeds.shape[0]
        device = trunk_conditioning.device
        transformer_dtype = next(self.cogvideox.parameters()).dtype

        latents_shape = (batch_size, self.num_frames, self.input_channels, self.frame_size, self.frame_size)
        latents = torch.randn(latents_shape, generator=generator, device=device, dtype=transformer_dtype)

        num_train_timesteps = 1000
        actual_steps = max(1, math.ceil(tau_v * num_inference_steps))
        dt = tau_v / actual_steps

        image_rotary_emb = None
        encoder_hidden_states = prompt_embeds.to(transformer_dtype)

        # Integrate from noise to tau_v with explicit Euler updates.
        for step in range(actual_steps):
            t_cont = step * dt if tau_v > 0 else 0
            t_discretized_int = int(t_cont * num_train_timesteps)

            timestep = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=transformer_dtype
            )

            with torch.no_grad():
                noise_pred = self.cogvideox(
                    hidden_states=latents,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0].to(transformer_dtype)

            latents = latents + noise_pred * dt
            latents = latents.to(transformer_dtype)

        # A final evaluation returns the guidance hidden state at tau_v.
        t_discretized_final = int(tau_v * num_train_timesteps)
        timestep_final = torch.full(
            size=(batch_size,), fill_value=t_discretized_final, device=device, dtype=transformer_dtype
        )

        latents_for_forward = latents.permute(0, 2, 1, 3, 4)

        outputs = self(
            noisy_latents=latents_for_forward,
            timesteps=timestep_final,
            trunk_conditioning=trunk_conditioning,
            return_hidden_states=True,
        )

        if hasattr(self, "put_frames_in_channels") and self.put_frames_in_channels > 1:
            B, T, C, H, W = latents.shape
            latents = latents.view(B, T, C // self.put_frames_in_channels, self.put_frames_in_channels, H, W)
            latents = (
                latents.permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .view(B, T * self.put_frames_in_channels, C // self.put_frames_in_channels, H, W)
            )

        latents = self.unnormalize_act_data(latents)
        motion = rearrange(latents, "b t c h w -> b (h w) t c")

        return {
            "motion": motion,
            "hidden": outputs["hidden"],
            "tau_v": tau_v,
        }

    def predict_trajectory(
        self,
        trunk_conditioning: torch.Tensor,
        scheduler: FlowMatchEulerDiscreteScheduler = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Generate a complete motion trajectory with Euler integration.

        Args:
            trunk_conditioning: VLM features with shape ``[B, sequence, latent_dim]``.
            scheduler: Optional scheduler used to scale model inputs.
            num_inference_steps: Number of Euler updates.
            guidance_scale: Classifier-free guidance scale.
            generator: Optional random generator for deterministic sampling.

        Returns:
            Motion with shape ``[B, keypoints, frames, motion_dim]``.
        """
        expected_seq_len = getattr(self.cogvideox.config, "max_text_seq_length", trunk_conditioning.shape[-2])
        if trunk_conditioning.shape[-2] != expected_seq_len:
            logger.warning(
                "expected conditioning sequence length %d, received %d",
                expected_seq_len,
                trunk_conditioning.shape[-2],
            )
        do_classifier_free_guidance = guidance_scale > 1.0

        prompt_embeds = self.project_latents_to_cogvideox_format(trunk_conditioning).to(self.cogvideox.dtype)

        batch_size, seq_len, embed_dim = prompt_embeds.shape
        negative_prompt_embeds = (
            torch.randn(batch_size, seq_len, embed_dim, device=prompt_embeds.device, dtype=prompt_embeds.dtype) * 0.1
        )

        if do_classifier_free_guidance and negative_prompt_embeds is not None:
            encoder_hidden_states = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        else:
            encoder_hidden_states = prompt_embeds

        transformer_dtype = next(self.cogvideox.parameters()).dtype
        encoder_hidden_states = encoder_hidden_states.to(transformer_dtype)

        latents_shape = (
            batch_size,
            self.num_frames,
            self.input_channels,
            self.frame_size,
            self.frame_size,
        )
        latents = torch.randn(
            latents_shape, generator=generator, device=trunk_conditioning.device, dtype=transformer_dtype
        )
        num_train_timesteps = 1000
        device = trunk_conditioning.device

        image_rotary_emb = None
        dt = 1.0 / num_inference_steps

        for t in range(num_inference_steps):
            t_cont = t / float(num_inference_steps)
            t_discretized_int = int(t_cont * num_train_timesteps)

            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
            else:
                latent_model_input = latents

            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            timestep = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=transformer_dtype
            )

            with torch.no_grad():
                noise_pred = self.cogvideox(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0].to(transformer_dtype)

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Flow matching uses the Euler update x_(t+1) = x_t + v * dt.
            latents = latents + noise_pred * dt
            latents = latents.to(transformer_dtype)

        if hasattr(self, "put_frames_in_channels") and self.put_frames_in_channels > 1:
            B, T, C, H, W = latents.shape

            # Restore frames that were packed in the channel axis.
            latents = latents.view(B, T, C // self.put_frames_in_channels, self.put_frames_in_channels, H, W)
            latents = (
                latents.permute(0, 1, 3, 2, 4, 5)
                .contiguous()
                .view(B, T * self.put_frames_in_channels, C // self.put_frames_in_channels, H, W)
            )

        latents = self.unnormalize_act_data(latents)
        videos = rearrange(latents, "b t c h w -> b (h w) t c")
        return videos
