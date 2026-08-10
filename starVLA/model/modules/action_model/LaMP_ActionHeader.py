"""Checkpoint-compatible Concat Action Expert for LaMP."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


def timestep_embedding(timestep: torch.Tensor, dimension: int, max_period: int = 100) -> torch.Tensor:
    """Create the sinusoidal embedding used by the released Action Expert."""
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=timestep.dtype, device=timestep.device) / half
    )
    embedding = torch.cat(
        [torch.cos(timestep[:, None] * frequencies[None]), torch.sin(timestep[:, None] * frequencies[None])],
        dim=-1,
    )
    if dimension % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.drop1(self.act(self.fc1(inputs)))
        return self.drop2(self.fc2(inputs))


class Attention(nn.Module):
    def __init__(self, dimension: int, num_heads: int, drop: float = 0.0) -> None:
        super().__init__()
        if dimension % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dimension, dimension * 3, bias=True)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dimension, dimension)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        outputs = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        outputs = outputs.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(outputs))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads=num_heads, drop=drop)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio), drop=drop)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attn(self.norm1(inputs))
        return inputs + self.mlp(self.norm2(inputs))


class LaMPActionHead(nn.Module):
    """Flow-matching Action Expert with the released Concat parameter layout."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        action_config = config.framework.action_model
        qwen_config = config.framework.qwenvl
        if action_config.get("use_adaln", False):
            raise ValueError("LaMP v0.1.0 only supports the checkpoint-compatible Concat Action Expert")

        self.hidden_size = action_config.get("hidden_size", 1024)
        self.vlm_hidden_size = qwen_config.get("vl_hidden_dim", 2560)
        self.depth = action_config.get("depth", 24)
        self.num_heads = action_config.get("num_heads", 16)
        self.mlp_ratio = action_config.get("mlp_ratio", 4.0)
        self.action_dim = action_config.get("action_dim", 7)
        self.state_dim = action_config.get("state_dim", 0)
        self.dim_time = action_config.get("dim_time", 32)
        self.max_len_seq = action_config.get("max_len_seq", 512)
        self.use_proprio = action_config.get("use_proprio", True)
        self.num_inference_timesteps = action_config.get("num_inference_timesteps", 10)
        self.action_horizon = action_config.get("future_action_window_size", 9) + 1

        self.noise_beta_alpha = action_config.get("noise_beta_alpha", 1.5)
        self.noise_beta_beta = action_config.get("noise_beta_beta", 1.0)
        self.noise_s = action_config.get("noise_s", 0.999)
        self.beta_dist = Beta(self.noise_beta_alpha, self.noise_beta_beta)

        self.blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.num_heads, self.mlp_ratio) for _ in range(self.depth)]
        )
        self.vlm_proj = nn.Linear(self.vlm_hidden_size, self.hidden_size)
        action_input_dim = self.action_dim + self.dim_time
        if self.use_proprio and self.state_dim > 0:
            action_input_dim += self.state_dim
        self.action_encoder = nn.Linear(action_input_dim, self.hidden_size)
        self.action_decoder = nn.Linear(self.hidden_size, self.action_dim)
        self.norm = nn.LayerNorm(self.hidden_size)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_len_seq, self.hidden_size), requires_grad=True)
        nn.init.normal_(self.pos_emb, std=0.02)

    def sample_time(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype)
        return sample * self.noise_s + 0.001

    def _validate_state(self, state: torch.Tensor | None, batch_size: int) -> torch.Tensor | None:
        if not self.use_proprio or self.state_dim <= 0:
            return None
        if state is None:
            raise ValueError("state is required because framework.action_model.use_proprio is enabled")
        if state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        if state.shape != (batch_size, self.state_dim):
            raise ValueError(f"state must have shape [{batch_size}, {self.state_dim}], got {tuple(state.shape)}")
        return state

    def _predict_velocity(
        self,
        vl_features: torch.Tensor,
        noisy_actions: torch.Tensor,
        state: torch.Tensor | None,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_actions = noisy_actions.shape[:2]
        state = self._validate_state(state, batch)
        time = timestep_embedding(timestep, self.dim_time).unsqueeze(1).expand(batch, num_actions, self.dim_time)
        action_inputs = [noisy_actions, time]
        if state is not None:
            action_inputs.append(state.unsqueeze(1).expand(batch, num_actions, self.state_dim))
        action_tokens = self.action_encoder(torch.cat(action_inputs, dim=-1))
        hidden = torch.cat([action_tokens, self.vlm_proj(vl_features)], dim=1)
        if hidden.shape[1] > self.pos_emb.shape[1]:
            raise ValueError(f"sequence length {hidden.shape[1]} exceeds max_len_seq={self.pos_emb.shape[1]}")
        hidden = hidden + self.pos_emb[:, : hidden.shape[1]]
        for block in self.blocks:
            hidden = block(hidden)
        return self.action_decoder(self.norm(hidden[:, :num_actions]))

    def forward(
        self,
        vl_features: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = vl_features.shape[0]
        timestep = self.sample_time(batch, vl_features.device, vl_features.dtype)
        noise = torch.randn_like(actions)
        noisy_actions = timestep[:, None, None] * noise + (1 - timestep[:, None, None]) * actions
        velocity = self._predict_velocity(vl_features, noisy_actions, state, timestep)
        return F.mse_loss(velocity, noise - actions)

    @torch.no_grad()
    def predict_action(
        self,
        vl_features: torch.Tensor,
        state: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate the Action Expert velocity field with ten Euler steps."""
        batch = vl_features.shape[0]
        actions = torch.randn(
            batch,
            self.action_horizon,
            self.action_dim,
            device=vl_features.device,
            dtype=vl_features.dtype,
            generator=generator,
        )
        delta_time = -1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            timestep = torch.full(
                (batch,),
                1.0 + step * delta_time,
                device=vl_features.device,
                dtype=vl_features.dtype,
            )
            actions = actions + delta_time * self._predict_velocity(vl_features, actions, state, timestep)
        return actions
