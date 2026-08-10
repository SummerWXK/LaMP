"""Raw LIBERO input preprocessing owned by the LaMP framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class ProcessedBatch:
    images: list[list[Image.Image]]
    instructions: list[str]
    states: torch.Tensor | None
    actions: torch.Tensor | None


class LaMPPreprocessor:
    """Copy, resize, normalize, and batch raw samples without mutating callers."""

    def __init__(
        self,
        image_size: tuple[int, int],
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        use_proprio: bool,
        mask_state_z: bool,
    ) -> None:
        self.image_size = image_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_proprio = use_proprio
        self.mask_state_z = mask_state_z
        self.statistics: dict[str, Any] = {}

    def set_statistics(self, statistics: dict[str, Any]) -> None:
        if not isinstance(statistics, dict):
            raise TypeError("dataset statistics must be a dictionary")
        self.statistics = statistics

    def prepare(
        self,
        examples: list[dict[str, Any]],
        *,
        device: torch.device,
        dtype: torch.dtype,
        require_actions: bool = False,
        require_state: bool = True,
        normalization_key: str = "libero",
    ) -> ProcessedBatch:
        if not examples:
            raise ValueError("examples cannot be empty")

        images: list[list[Image.Image]] = []
        instructions: list[str] = []
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        state_presence = []
        action_presence = []

        for sample in examples:
            sample_images = sample.get("image")
            if not isinstance(sample_images, (list, tuple)) or len(sample_images) != 2:
                raise ValueError("each sample must contain image=[primary_image, wrist_image]")
            images.append([self._copy_and_resize(image) for image in sample_images])

            instruction = sample.get("lang")
            if not isinstance(instruction, str):
                raise TypeError("each sample must contain lang as str")
            instructions.append(instruction)

            state_presence.append(sample.get("state") is not None)
            if state_presence[-1]:
                state = np.array(sample["state"], dtype=np.float32, copy=True)
                if state.shape == (1, self.state_dim):
                    state = state[0]
                if state.shape != (self.state_dim,):
                    raise ValueError(f"state must have shape [{self.state_dim}], got {state.shape}")
                states.append(state)

            action_presence.append(sample.get("action") is not None)
            if action_presence[-1]:
                action = np.array(sample["action"], dtype=np.float32, copy=True)
                expected = (self.action_horizon, self.action_dim)
                if action.shape != expected:
                    raise ValueError(f"action must have shape {expected}, got {action.shape}")
                actions.append(action)

        self._validate_optional_field("state", state_presence)
        self._validate_optional_field("action", action_presence)
        if require_state and self.use_proprio and not all(state_presence):
            raise ValueError("state is required because proprioception is enabled")
        if require_actions and not all(action_presence):
            raise ValueError("action is required for Stage 2 training")

        state_tensor = None
        if all(state_presence):
            normalized_state = self.normalize_state(np.stack(states), normalization_key)
            if self.mask_state_z:
                normalized_state = normalized_state.copy()
                normalized_state[:, 2] = 0
            state_tensor = torch.from_numpy(normalized_state).to(device=device, dtype=dtype)

        action_tensor = None
        if all(action_presence):
            normalized_action = self.normalize_action(np.stack(actions), normalization_key)
            action_tensor = torch.from_numpy(normalized_action).to(device=device, dtype=dtype)

        return ProcessedBatch(images=images, instructions=instructions, states=state_tensor, actions=action_tensor)

    def normalize_state(self, state: np.ndarray, normalization_key: str) -> np.ndarray:
        statistics = self._statistics_for(normalization_key)["state"]
        normalized = self._min_max(state, statistics, continuous_dimensions=self.state_dim - 1)
        normalized[..., -1] = (state[..., -1] > 0.5).astype(np.float32)
        return normalized

    def normalize_action(self, action: np.ndarray, normalization_key: str) -> np.ndarray:
        statistics = self._statistics_for(normalization_key)["action"]
        normalized = self._min_max(action, statistics, continuous_dimensions=self.action_dim - 1)
        normalized[..., -1] = (action[..., -1] > 0.5).astype(np.float32)
        return normalized

    def unnormalize_action(self, action: np.ndarray, normalization_key: str) -> np.ndarray:
        statistics = self._statistics_for(normalization_key)["action"]
        minimum = np.asarray(statistics["min"], dtype=np.float32)
        maximum = np.asarray(statistics["max"], dtype=np.float32)
        output = np.array(action, dtype=np.float32, copy=True)
        dimensions = self.action_dim - 1
        normalized_continuous = np.clip(output[..., :dimensions], -1, 1)
        output[..., :dimensions] = (normalized_continuous + 1) / 2 * (
            maximum[:dimensions] - minimum[:dimensions]
        ) + minimum[:dimensions]
        output[..., -1] = (output[..., -1] > 0.5).astype(np.float32)
        return output

    def _statistics_for(self, normalization_key: str) -> dict[str, Any]:
        key = normalization_key
        if key not in self.statistics and key == "libero" and "franka" in self.statistics:
            key = "franka"
        if key not in self.statistics:
            available = ", ".join(sorted(self.statistics)) or "<none>"
            raise KeyError(f"normalization key {normalization_key!r} is unavailable; choices: {available}")
        return self.statistics[key]

    @staticmethod
    def _min_max(values: np.ndarray, statistics: dict[str, Any], continuous_dimensions: int) -> np.ndarray:
        minimum = np.asarray(statistics["min"], dtype=np.float32)
        maximum = np.asarray(statistics["max"], dtype=np.float32)
        output = np.array(values, dtype=np.float32, copy=True)
        span = maximum[:continuous_dimensions] - minimum[:continuous_dimensions]
        valid = span != 0
        continuous = output[..., :continuous_dimensions]
        continuous[...] = 0
        continuous[..., valid] = (
            2 * (values[..., :continuous_dimensions][..., valid] - minimum[:continuous_dimensions][valid]) / span[valid]
            - 1
        )
        return output

    def _copy_and_resize(self, image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            copied = image.copy()
        elif isinstance(image, np.ndarray):
            copied = Image.fromarray(np.array(image, copy=True))
        else:
            raise TypeError(f"images must be PIL.Image or numpy.ndarray, got {type(image).__name__}")
        return copied.resize(self.image_size) if copied.size != self.image_size else copied

    @staticmethod
    def _validate_optional_field(name: str, presence: list[bool]) -> None:
        if any(presence) and not all(presence):
            raise ValueError(f"{name} must be present in all samples or none")
