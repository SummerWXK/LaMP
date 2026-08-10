"""LaMP policy with checkpoint-compatible module names."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from diffusers import FlowMatchEulerDiscreteScheduler

from starVLA.model.modules.action_model import LaMPActionHead
from starVLA.model.modules.motion_model.decoder.cogvideox_flow import CogVideoXDecoder_flow
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.preprocessing import LaMPPreprocessor

RESULT_KEY_ACTIONS = "normalized_actions"
RESULT_KEY_UNNORMALIZED_ACTIONS = "actions"
RESULT_KEY_MOTION = "motion_flow"


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


class MotionCrossAttention(nn.Module):
    """Inject Motion Expert hidden states into the last Qwen hidden state."""

    def __init__(
        self,
        vl_embed_dim: int,
        motion_dim: int,
        num_heads: int = 40,
        dropout: float = 0.1,
        init_gate_value: float = 0.0,
    ) -> None:
        super().__init__()
        if vl_embed_dim % num_heads:
            raise ValueError("vl_embed_dim must be divisible by the number of guidance heads")
        self.vl_embed_dim = vl_embed_dim
        self.num_heads = num_heads
        self.head_dim = vl_embed_dim // num_heads
        self.motion_proj = nn.Linear(motion_dim, vl_embed_dim)
        self.norm_vl = nn.LayerNorm(vl_embed_dim)
        self.norm_motion = nn.LayerNorm(vl_embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=vl_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.tensor([init_gate_value]))

    def forward(self, vl_hidden: torch.Tensor, motion_hidden: torch.Tensor) -> torch.Tensor:
        motion = self.norm_motion(self.motion_proj(motion_hidden))
        attention, _ = self.cross_attn(
            query=self.norm_vl(vl_hidden),
            key=motion,
            value=motion,
            need_weights=False,
        )
        return vl_hidden + torch.sigmoid(self.gate) * attention


class MotionFlowHead(nn.Module):
    """CogVideoX Motion Expert; Stage 1 loss and training are intentionally not exposed."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        motion_config = config.framework.motion_expert
        action_config = config.framework.action_model
        self.spatial_size = motion_config.get("spatial_size", 20)
        self.num_spatial = self.spatial_size**2
        self.num_temporal = motion_config.get("num_motion_frames", 32)
        self.motion_dim = motion_config.get("motion_dim", 3)
        self.vlm_hidden_size = config.framework.qwenvl.get("vl_hidden_dim", 2560)
        latent_dim = motion_config.get("latent_dim", 1024)

        self.decoder = CogVideoXDecoder_flow(
            latent_dim=latent_dim,
            num_attention_heads=motion_config.get("num_attention_heads", 16),
            attention_head_dim=motion_config.get("attention_head_dim", 64),
            in_channels=self.motion_dim,
            out_channels=self.motion_dim,
            num_layers=motion_config.get("num_layers", 12),
            num_frames=self.num_temporal,
            frame_size=self.spatial_size,
            patch_size=motion_config.get("patch_size", 2),
            patch_size_t=motion_config.get("patch_size_t", 1),
            max_text_seq_length=motion_config.get("max_text_seq_length", 256),
            text_embed_dim=latent_dim,
            use_rotary_positional_embeddings=motion_config.get("use_rotary_embeddings", False),
            enable_encoder_hidden_states_grad=True,
            put_frames_in_channels=motion_config.get("put_frames_in_channels", 2),
        )
        self.vlm_proj = nn.Linear(self.vlm_hidden_size, latent_dim, bias=False)
        self.num_timestep_buckets = action_config.get("num_timestep_buckets", 1000)
        self.num_inference_steps = action_config.get("num_inference_timesteps", 10)
        self.inner_dim = latent_dim
        self._inference_scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=self.num_timestep_buckets)
        self.tau_v_infer = motion_config.get("tau_v_infer", 0.1)
        self.motion_infer_steps = motion_config.get("motion_infer_steps", 10)
        action_min = torch.tensor(motion_config.get("action_min", [-0.05, -0.05, -0.04]))
        action_max = torch.tensor(motion_config.get("action_max", [0.05, 0.05, 0.04]))
        self.decoder.set_data_act_statistics(action_max, action_min)

    @torch.no_grad()
    def predict_partial_with_tau(
        self,
        vl_features: torch.Tensor,
        tau_v: float | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return partial motion and guidance hidden state.

        With the released defaults (tau_v=0.1 and ten nominal integration
        steps), this performs one Euler state update and then a second Motion
        Expert evaluation to obtain the hidden state. It is therefore two
        Motion Expert evaluations, not one network call.
        """
        outputs = self.decoder.predict_partial_with_tau(
            trunk_conditioning=self.vlm_proj(vl_features),
            tau_v=self.tau_v_infer if tau_v is None else tau_v,
            num_inference_steps=self.motion_infer_steps,
            generator=generator,
        )
        hidden = outputs["hidden"]
        return {
            "motion": outputs["motion"],
            "hidden_last": hidden[-1] if isinstance(hidden, list) else hidden,
            "tau_v": outputs["tau_v"],
        }

    @torch.no_grad()
    def predict(
        self,
        vl_features: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.decoder.predict_trajectory(
            trunk_conditioning=self.vlm_proj(vl_features),
            scheduler=self._inference_scheduler,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=1.0,
            generator=generator,
        )


class LaMP(nn.Module):
    """Qwen3-VL + Motion Expert + Motion Guidance + Concat Action Expert."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.framework.name != "LaMP":
            raise ValueError("framework.name must be 'LaMP'")
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config)
        hidden_size = self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = hidden_size
        self.config.framework.qwenvl.num_vl_layers = getattr(
            self.qwen_vl_interface.model.config,
            "num_hidden_layers",
            36,
        )
        self.llm_hidden_size = hidden_size

        motion_config = config.framework.motion_expert
        if not motion_config.get("enabled", True):
            raise ValueError("LaMP v0.1.0 requires the Motion Expert")
        if not motion_config.get("use_guidance", True):
            raise ValueError("LaMP v0.1.0 requires Motion Guidance")
        self.use_motion_expert = True
        self.enable_motion_guidance = True
        self.motion_inner_dim = motion_config.get("latent_dim", 1024)
        self.motion_head = MotionFlowHead(config)
        self.motion_guidance = MotionCrossAttention(
            vl_embed_dim=hidden_size,
            motion_dim=self.motion_inner_dim,
            num_heads=motion_config.get("guidance_heads", 40),
            dropout=motion_config.get("guidance_dropout", 0.1),
            init_gate_value=motion_config.get("guidance_init_gate", 0.0),
        )
        self.action_model = LaMPActionHead(config)

        action_config = config.framework.action_model
        image_size = tuple(config.datasets.vla_data.get("image_size", [224, 224]))
        self.preprocessor = LaMPPreprocessor(
            image_size=image_size,
            state_dim=action_config.get("state_dim", 8),
            action_dim=action_config.get("action_dim", 7),
            action_horizon=action_config.get("future_action_window_size", 9) + 1,
            use_proprio=action_config.get("use_proprio", True),
            mask_state_z=action_config.get("use_mask_z", False),
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def compute_dtype(self) -> torch.dtype:
        return next(self.action_model.parameters()).dtype

    def set_dataset_statistics(self, statistics: dict[str, Any]) -> None:
        self.preprocessor.set_statistics(statistics)

    def encode_vlm(self, batch_images: list[list[Any]], instructions: list[str]) -> torch.Tensor:
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(batch_images, instructions)
        dtype = next(self.qwen_vl_interface.parameters()).dtype
        with _autocast(self.device, dtype):
            outputs = self.qwen_vl_interface(
                **inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1]

    def forward_action(
        self,
        vl_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None,
        motion_hidden: torch.Tensor,
        repeat_steps: int = 1,
    ) -> dict[str, torch.Tensor]:
        if repeat_steps < 1:
            raise ValueError("repeat_steps must be positive")
        vl_features = vl_features.repeat(repeat_steps, 1, 1)
        actions = actions.repeat(repeat_steps, 1, 1)
        motion_hidden = motion_hidden.repeat(repeat_steps, 1, 1)
        state = state.repeat(repeat_steps, 1) if state is not None else None
        guided_vl = self.motion_guidance(vl_features, motion_hidden)
        return {"action_loss": self.action_model(guided_vl, actions, state)}

    def forward(
        self,
        examples: list[dict[str, Any]],
        repeat_steps: int = 2,
        normalization_key: str = "libero",
    ) -> dict[str, torch.Tensor]:
        batch = self.preprocessor.prepare(
            examples,
            device=self.device,
            dtype=self.compute_dtype,
            require_actions=True,
            normalization_key=normalization_key,
        )
        vl_features = self.encode_vlm(batch.images, batch.instructions)
        with torch.no_grad(), _autocast(self.device, self.compute_dtype):
            motion = self.motion_head.predict_partial_with_tau(vl_features)
        with _autocast(self.device, self.compute_dtype):
            return self.forward_action(
                vl_features=vl_features,
                actions=batch.actions,
                state=batch.states,
                motion_hidden=motion["hidden_last"],
                repeat_steps=repeat_steps,
            )

    @torch.no_grad()
    def predict_action(
        self,
        examples: list[dict[str, Any]],
        *,
        unnorm_key: str = "libero",
        generator: torch.Generator | None = None,
        return_motion: bool = False,
    ) -> dict[str, np.ndarray]:
        """Predict normalized and physical actions from raw LIBERO observations."""
        batch = self.preprocessor.prepare(
            examples,
            device=self.device,
            dtype=self.compute_dtype,
            normalization_key=unnorm_key,
        )
        vl_features = self.encode_vlm(batch.images, batch.instructions)
        with _autocast(self.device, self.compute_dtype):
            motion = self.motion_head.predict_partial_with_tau(vl_features, generator=generator)
            guided_vl = self.motion_guidance(vl_features, motion["hidden_last"])
            normalized = self.action_model.predict_action(guided_vl, batch.states, generator=generator)

        normalized_array = normalized.float().cpu().numpy()
        results = {
            RESULT_KEY_ACTIONS: normalized_array,
            RESULT_KEY_UNNORMALIZED_ACTIONS: self.preprocessor.unnormalize_action(normalized_array, unnorm_key),
        }
        if return_motion:
            results[RESULT_KEY_MOTION] = motion["motion"].float().cpu().numpy()
        return results

    @torch.no_grad()
    def predict_motion(
        self,
        examples: list[dict[str, Any]],
        *,
        generator: torch.Generator | None = None,
    ) -> np.ndarray:
        """Run full Motion Expert inference; action prediction uses partial inference instead."""
        batch = self.preprocessor.prepare(
            examples,
            device=self.device,
            dtype=self.compute_dtype,
            require_state=False,
        )
        vl_features = self.encode_vlm(batch.images, batch.instructions)
        with _autocast(self.device, self.compute_dtype):
            motion = self.motion_head.predict(vl_features, generator=generator)
        return motion.float().cpu().numpy()
