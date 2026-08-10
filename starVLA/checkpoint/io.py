"""Strict weights-only loading for Hugging Face and local LaMP .pt checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from starVLA.constants import QWEN3_VL_REVISION


def load_policy(
    model_id_or_path: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: str | torch.dtype = "bfloat16",
    strict: bool = True,
):
    """Load a complete LaMP policy and its normalization statistics.

    Checkpoints are always read with torch.load(weights_only=True) and must
    contain only a tensor state dictionary.
    """
    source = Path(model_id_or_path).expanduser()
    checkpoint_file: Path | None = None
    if source.is_file():
        checkpoint_file = source.resolve()
        artifact_dir = _pt_artifact_directory(checkpoint_file)
    elif source.is_dir():
        artifact_dir = source.resolve()
    else:
        artifact_dir = Path(
            snapshot_download(
                repo_id=str(model_id_or_path),
                allow_patterns=[
                    "*.yaml",
                    "*.json",
                    "*.pt",
                    "README.md",
                ],
            )
        )

    config_path = artifact_dir / "config.yaml"
    statistics_path = artifact_dir / "dataset_statistics.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing LaMP config: {config_path}")
    if not statistics_path.exists():
        raise FileNotFoundError(f"missing dataset statistics: {statistics_path}")
    _verify_manifest(artifact_dir)
    if checkpoint_file is None:
        checkpoint_file = _find_pt_checkpoint(artifact_dir)
    _verify_checkpoint_record(artifact_dir, checkpoint_file)

    config = OmegaConf.load(config_path)
    _validate_config(config)
    base_vlm = Path(str(config.framework.qwenvl.base_vlm)).expanduser()
    if str(config.framework.qwenvl.base_vlm).startswith((".", "/")) and not base_vlm.exists():
        config.framework.qwenvl.base_vlm = "Qwen/Qwen3-VL-4B-Instruct"
    config.framework.qwenvl.revision = QWEN3_VL_REVISION
    from starVLA.model.framework import build_framework

    model = build_framework(config)

    state_dict = _load_pt(checkpoint_file)
    _validate_top_level_keys(state_dict)
    model.load_state_dict(state_dict, strict=strict)

    model.to(device=torch.device(device), dtype=_torch_dtype(dtype))
    with statistics_path.open("r", encoding="utf-8") as handle:
        model.set_dataset_statistics(json.load(handle))
    model.eval()
    return model


def _pt_artifact_directory(checkpoint: Path) -> Path:
    if checkpoint.suffix != ".pt":
        raise ValueError("LaMP checkpoint files must use the .pt format")
    return checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent


def _find_pt_checkpoint(directory: Path) -> Path:
    checkpoints = sorted(directory.glob("*.pt"))
    if len(checkpoints) != 1:
        raise ValueError(f"expected exactly one .pt checkpoint in {directory}, found {len(checkpoints)}")
    return checkpoints[0]


def _load_pt(checkpoint: Path) -> dict[str, torch.Tensor]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError("checkpoint must be a model-only tensor state dictionary")
    return state


def _validate_top_level_keys(state_dict: dict[str, torch.Tensor]) -> None:
    required = {"qwen_vl_interface", "motion_head", "motion_guidance", "action_model"}
    present = {key.split(".", 1)[0] for key in state_dict}
    missing = required - present
    extra = present - required
    if missing:
        raise ValueError(f"checkpoint is missing required LaMP modules: {sorted(missing)}")
    if extra:
        raise ValueError(f"checkpoint contains unsupported top-level modules: {sorted(extra)}")


def _validate_config(config: Any) -> None:
    required_paths = [
        "framework.name",
        "framework.qwenvl.base_vlm",
        "framework.motion_expert",
        "framework.action_model",
        "datasets.vla_data.image_size",
    ]
    missing = [path for path in required_paths if OmegaConf.select(config, path) is None]
    if missing:
        raise ValueError(f"invalid LaMP config; missing: {', '.join(missing)}")
    if config.framework.name != "LaMP":
        raise ValueError("only framework.name=LaMP is supported")


def _verify_manifest(directory: Path) -> None:
    manifest_path = directory / "checkpoint_manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for file_record in manifest.get("files", []):
        path = _artifact_file(directory, file_record["name"])
        if not path.exists():
            raise FileNotFoundError(f"manifest file is missing: {path}")
        expected_size = file_record.get("size")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"size mismatch for {path.name}")
        expected = file_record.get("sha256")
        if expected and _sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch for {path.name}")


def _verify_checkpoint_record(directory: Path, checkpoint: Path) -> None:
    manifest_path = directory / "checkpoint_manifest.json"
    if not manifest_path.exists():
        return
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    record = manifest.get("checkpoint")
    if not record:
        return
    recorded_path = _artifact_file(directory, record["name"])
    if recorded_path.resolve() != checkpoint.resolve():
        raise ValueError(f"manifest checkpoint {recorded_path.name} does not match {checkpoint.name}")
    expected_size = record.get("size")
    if expected_size is not None and checkpoint.stat().st_size != expected_size:
        raise ValueError(f"size mismatch for {checkpoint.name}")
    expected = record.get("sha256")
    if expected and _sha256(checkpoint) != expected:
        raise ValueError(f"SHA256 mismatch for {checkpoint.name}")


def _artifact_file(directory: Path, filename: str) -> Path:
    if not isinstance(filename, str) or Path(filename).name != filename or "\\" in filename:
        raise ValueError(f"invalid checkpoint artifact filename: {filename!r}")
    return directory / filename


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    try:
        result = getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unsupported dtype: {value}") from error
    if not isinstance(result, torch.dtype):
        raise ValueError(f"unsupported dtype: {value}")
    return result
