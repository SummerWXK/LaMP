"""Raw LIBERO adapter for the four paper suites stored as LeRobot v2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

LOGGER = logging.getLogger(__name__)

LIBERO_SUITES = {
    "libero_spatial": "libero_spatial_no_noops_lerobot",
    "libero_object": "libero_object_no_noops_lerobot",
    "libero_goal": "libero_goal_no_noops_lerobot",
    "libero_10": "libero_10_no_noops_lerobot",
}

PRIMARY_IMAGE_KEYS = ("observation.images.image", "observation.images.primary_image")
WRIST_IMAGE_KEYS = ("observation.images.wrist_image",)
LANGUAGE_KEYS = ("task", "language", "lang")
STATE_KEY = "observation.state"
ACTION_KEY = "action"

GR00T_VIDEO_KEYS = ("video.primary_image", "video.wrist_image")
GR00T_LANGUAGE_KEY = "annotation.human.action.task_description"
GR00T_STATE_KEYS = tuple(f"state.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"))
GR00T_ACTION_KEYS = tuple(f"action.{name}" for name in ("x", "y", "z", "roll", "pitch", "yaw", "gripper"))


class RecordDataset(Protocol):
    fps: int

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> dict[str, Any]: ...


def _build_gr00t_lerobot_reader(dataset_path: Path, *, video_backend: str):
    """Build the repository-bundled LeRobot-v2 reader without external lerobot."""
    from .gr00t_lerobot.datasets import LeRobotSingleDataset, ModalityConfig
    from .gr00t_lerobot.embodiment_tags import EmbodimentTag

    modality_configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=list(GR00T_VIDEO_KEYS)),
        "state": ModalityConfig(delta_indices=[0], modality_keys=list(GR00T_STATE_KEYS)),
        "action": ModalityConfig(delta_indices=list(range(10)), modality_keys=list(GR00T_ACTION_KEYS)),
        "language": ModalityConfig(delta_indices=[0], modality_keys=[GR00T_LANGUAGE_KEY]),
    }
    backend_name = "torchvision_av" if video_backend == "torchvision" else video_backend
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag=EmbodimentTag.FRANKA,
        video_backend=backend_name,
        data_cfg={"lerobot_version": "v2.0"},
    )


class _BundledLeRobotBackend:
    """Expose the mature GR00T LeRobot reader through the raw record interface."""

    def __init__(self, dataset_path: Path, *, video_backend: str) -> None:
        self.reader = _build_gr00t_lerobot_reader(dataset_path, video_backend=video_backend)
        with (dataset_path / "meta" / "info.json").open("r", encoding="utf-8") as handle:
            self.fps = int(json.load(handle)["fps"])

    def __len__(self) -> int:
        return len(self.reader)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self.reader.all_steps[index]
        raw = self.reader.get_step_data(episode_index, frame_index)
        state = np.concatenate([np.asarray(raw[key], dtype=np.float32) for key in GR00T_STATE_KEYS], axis=-1)[0]
        action = np.concatenate([np.asarray(raw[key], dtype=np.float32) for key in GR00T_ACTION_KEYS], axis=-1)
        language = raw[GR00T_LANGUAGE_KEY]
        if isinstance(language, (list, tuple)):
            language = language[0]
        return {
            PRIMARY_IMAGE_KEYS[0]: np.asarray(raw[GR00T_VIDEO_KEYS[0]])[0],
            WRIST_IMAGE_KEYS[0]: np.asarray(raw[GR00T_VIDEO_KEYS[1]])[0],
            STATE_KEY: state,
            ACTION_KEY: action,
            "task": language,
            "episode_index": int(episode_index),
            "frame_index": int(frame_index),
        }


class LiberoRawDataset(Dataset):
    """Return only raw PIL/text/float32 arrays plus provenance metadata."""

    def __init__(
        self,
        root: str | Path,
        suite: str,
        *,
        backend: RecordDataset | None = None,
        max_retries: int = 3,
        video_backend: str = "torchvision",
    ) -> None:
        if suite not in LIBERO_SUITES:
            raise ValueError(f"unsupported LIBERO suite {suite!r}; choose from {sorted(LIBERO_SUITES)}")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.root = Path(root).expanduser()
        self.suite = suite
        self.dataset_key = LIBERO_SUITES[suite]
        self.dataset_path = self.root / self.dataset_key
        self.max_retries = max_retries
        self.video_backend = video_backend
        if backend is None:
            self._validate_v2_layout()
            backend = self._load_lerobot_backend()
        self.backend = backend

    def __len__(self) -> int:
        return len(self.backend)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        errors = []
        for attempt in range(self.max_retries):
            source_index = (index + attempt) % len(self)
            try:
                return self._convert_record(self.backend[source_index], index, source_index)
            except (KeyError, IndexError, TypeError, ValueError, OSError, RuntimeError) as error:
                errors.append(f"index {source_index}: {error}")
                LOGGER.warning(
                    "LIBERO sample %d failed on deterministic attempt %d/%d (source %d): %s",
                    index,
                    attempt + 1,
                    self.max_retries,
                    source_index,
                    error,
                )
        raise RuntimeError(f"unable to read requested LIBERO index {index}; " + " | ".join(errors))

    def _convert_record(self, record: dict[str, Any], requested_index: int, source_index: int) -> dict[str, Any]:
        primary = _to_pil(_first(record, PRIMARY_IMAGE_KEYS))
        wrist = _to_pil(_first(record, WRIST_IMAGE_KEYS))
        language = _first(record, LANGUAGE_KEYS)
        if not isinstance(language, str) or not language:
            raise ValueError("language instruction is empty or not a string")

        state = _to_float32_numpy(record[STATE_KEY])
        if state.shape == (1, 8):
            state = state[0]
        if state.shape != (8,):
            raise ValueError(f"state must have shape [8], got {state.shape}")

        action = _to_float32_numpy(record[ACTION_KEY])
        if action.shape != (10, 7):
            raise ValueError(f"action must have shape [10, 7], got {action.shape}")

        return {
            "image": [primary, wrist],
            "lang": language,
            "state": state,
            "action": action,
            "dataset_key": self.dataset_key,
            "suite": self.suite,
            "episode_index": _scalar(record.get("episode_index")),
            "frame_index": _scalar(record.get("frame_index")),
            "index": source_index,
            "requested_index": requested_index,
        }

    def _validate_v2_layout(self) -> None:
        info_path = self.dataset_path / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"missing LeRobot metadata: {info_path}")
        with info_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("codebase_version") not in {"v2.0", "v2.1"}:
            raise ValueError(f"{self.dataset_path} is not LeRobot v2 (found {metadata.get('codebase_version')!r})")
        features = metadata.get("features", {})
        required = {PRIMARY_IMAGE_KEYS[0], WRIST_IMAGE_KEYS[0], STATE_KEY, ACTION_KEY}
        missing = required - features.keys()
        if missing:
            raise ValueError(f"LeRobot metadata is missing required LIBERO features: {sorted(missing)}")

    def _load_lerobot_backend(self) -> RecordDataset:
        return _BundledLeRobotBackend(self.dataset_path, video_backend=self.video_backend)


def build_dataset(data_config: Any) -> Dataset:
    root = data_config.data_root_dir
    requested = data_config.get("suite", "all")
    suites = list(LIBERO_SUITES) if requested == "all" else [requested]
    datasets = [
        LiberoRawDataset(
            root,
            suite,
            max_retries=data_config.get("max_retries", 3),
            video_backend=data_config.get("video_backend", "torchvision"),
        )
        for suite in suites
    ]
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def collate_raw_samples(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep PIL images, strings, and NumPy arrays untouched by collation."""
    return batch


def build_dataloader(config: Any, **_: Any) -> DataLoader:
    data_config = config.datasets.vla_data
    return DataLoader(
        build_dataset(data_config),
        batch_size=data_config.per_device_batch_size,
        shuffle=data_config.get("shuffle", True),
        num_workers=data_config.get("num_workers", 8),
        pin_memory=data_config.get("pin_memory", True),
        persistent_workers=data_config.get("num_workers", 8) > 0,
        collate_fn=collate_raw_samples,
        drop_last=True,
    )


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record:
            value = record[key]
            if isinstance(value, (list, tuple)) and len(value) == 1:
                return value[0]
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 1 and value.ndim > 3:
                return value[0]
            return value
    raise KeyError(f"none of the required keys are present: {keys}")


def _to_float32_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=np.float32, copy=True)


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.copy()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 4 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise ValueError(f"image tensor must have three dimensions, got {tuple(value.shape)}")
        if value.shape[0] in {1, 3, 4}:
            value = value.permute(1, 2, 0)
        value = value.numpy()
    array = np.asarray(value)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 1)
        array = (array * 255).round().astype(np.uint8)
    if array.ndim != 3 or array.shape[-1] not in {1, 3, 4}:
        raise ValueError(f"image array must have HWC layout, got {array.shape}")
    if array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(np.array(array, copy=True))


def _scalar(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        value = value.item()
    return int(value)
