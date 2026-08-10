"""Prepare full-policy and Motion Expert .pt release artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from starVLA.checkpoint.io import _load_pt, _validate_top_level_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion-output", type=Path)
    parser.add_argument("--full-model-card", type=Path, default=Path("model_cards/lamp-libero.md"))
    parser.add_argument(
        "--motion-model-card",
        type=Path,
        default=Path("model_cards/lamp-motion-expert.md"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for path in (args.checkpoint, args.config, args.statistics, args.full_model_card):
        _require_file(path)
    if args.checkpoint.suffix != ".pt":
        raise ValueError("checkpoint must have a .pt suffix")
    if args.motion_output is not None:
        _require_file(args.motion_model_card)
    if args.source_manifest is not None:
        _require_file(args.source_manifest)
    _require_new_directory(args.output)
    if args.motion_output is not None:
        _require_new_directory(args.motion_output)

    config = OmegaConf.load(args.config)
    _reject_absolute_paths(OmegaConf.to_container(config, resolve=True))
    with args.statistics.open("r", encoding="utf-8") as handle:
        statistics = json.load(handle)
    if not isinstance(statistics, dict) or not statistics:
        raise ValueError("statistics must be a non-empty JSON object")

    source_sha256 = _sha256(args.checkpoint)
    provenance = None
    if args.source_manifest is not None:
        with args.source_manifest.open("r", encoding="utf-8") as handle:
            provenance = json.load(handle)
        _reject_absolute_paths(provenance, "source_manifest")
        _verify_expected_source_hash(provenance, source_sha256)

    state_dict = _load_pt(args.checkpoint)
    _validate_top_level_keys(state_dict)
    _write_full_artifact(
        output=args.output,
        source_checkpoint=args.checkpoint,
        state_dict=state_dict,
        config=args.config,
        statistics=args.statistics,
        model_card=args.full_model_card,
        source_sha256=source_sha256,
        source_provenance=provenance,
    )

    if args.motion_output is not None:
        motion_state = {key: value for key, value in state_dict.items() if key.startswith("motion_head.")}
        if not motion_state or len(motion_state) == len(state_dict):
            raise ValueError("could not isolate a non-empty motion_head.* subset")
        _write_motion_artifact(
            output=args.motion_output,
            state_dict=motion_state,
            config=args.config,
            statistics=args.statistics,
            model_card=args.motion_model_card,
            source_sha256=source_sha256,
            source_provenance=provenance,
        )
    return 0


def _write_full_artifact(
    *,
    output: Path,
    source_checkpoint: Path,
    state_dict: dict[str, torch.Tensor],
    config: Path,
    statistics: Path,
    model_card: Path,
    source_sha256: str,
    source_provenance: dict[str, Any] | None,
) -> None:
    output.mkdir(parents=True)
    checkpoint_path = output / "pytorch_model.pt"
    shutil.copy2(source_checkpoint, checkpoint_path)
    if _sha256(checkpoint_path) != source_sha256:
        raise RuntimeError("copied full-policy checkpoint hash does not match its source")
    _copy_metadata(output, config, statistics, model_card)
    _write_manifest(
        output=output,
        artifact_type="complete_policy",
        checkpoint_path=checkpoint_path,
        state_dict=state_dict,
        source_sha256=source_sha256,
        source_provenance=source_provenance,
        tensor_values_verified_exact=True,
    )


def _write_motion_artifact(
    *,
    output: Path,
    state_dict: dict[str, torch.Tensor],
    config: Path,
    statistics: Path,
    model_card: Path,
    source_sha256: str,
    source_provenance: dict[str, Any] | None,
) -> None:
    output.mkdir(parents=True)
    checkpoint_path = output / "motion_expert.pt"
    torch.save({key: tensor.detach().cpu() for key, tensor in state_dict.items()}, checkpoint_path)
    _verify_state_dict(state_dict, _load_pt(checkpoint_path))
    _copy_metadata(output, config, statistics, model_card)
    _write_manifest(
        output=output,
        artifact_type="motion_expert",
        checkpoint_path=checkpoint_path,
        state_dict=state_dict,
        source_sha256=source_sha256,
        source_provenance=source_provenance,
        tensor_values_verified_exact=True,
    )


def _copy_metadata(output: Path, config: Path, statistics: Path, model_card: Path) -> None:
    shutil.copy2(config, output / "config.yaml")
    shutil.copy2(statistics, output / "dataset_statistics.json")
    shutil.copy2(model_card, output / "README.md")


def _write_manifest(
    *,
    output: Path,
    artifact_type: str,
    checkpoint_path: Path,
    state_dict: dict[str, torch.Tensor],
    source_sha256: str,
    source_provenance: dict[str, Any] | None,
    tensor_values_verified_exact: bool,
) -> None:
    files = [_file_record(path) for path in sorted(output.iterdir())]
    checkpoint_record = next(record for record in files if record["name"] == checkpoint_path.name)
    manifest = {
        "artifact_type": artifact_type,
        "checkpoint": checkpoint_record,
        "checkpoint_sha256": checkpoint_record["sha256"],
        "files": files,
        "format": "pytorch_state_dict",
        "format_version": 1,
        "model": "LaMP",
        "optimizer_state_included": False,
        "scheduler_state_included": False,
        "source_checkpoint_sha256": source_sha256,
        "source_provenance": source_provenance,
        "state_dict": _state_dict_summary(state_dict),
        "tensor_values_verified_exact": tensor_values_verified_exact,
        "torch_load_weights_only": True,
    }
    with (output / "checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _verify_state_dict(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> None:
    if actual.keys() != expected.keys():
        missing = expected.keys() - actual.keys()
        extra = actual.keys() - expected.keys()
        raise RuntimeError(f"checkpoint key mismatch; missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")
    for key, expected_tensor in expected.items():
        actual_tensor = actual[key]
        if actual_tensor.shape != expected_tensor.shape or actual_tensor.dtype != expected_tensor.dtype:
            raise RuntimeError(f"checkpoint metadata mismatch for {key}")
        if not torch.equal(actual_tensor.cpu(), expected_tensor.cpu()):
            raise RuntimeError(f"checkpoint value mismatch for {key}")


def _state_dict_summary(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    dtype_counts = Counter(str(tensor.dtype).removeprefix("torch.") for tensor in state_dict.values())
    module_counts = Counter(key.split(".", 1)[0] for key in state_dict)
    return {
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "module_tensor_counts": dict(sorted(module_counts.items())),
        "parameter_count": sum(tensor.numel() for tensor in state_dict.values()),
        "tensor_count": len(state_dict),
    }


def _file_record(path: Path) -> dict[str, Any]:
    return {"name": path.name, "sha256": _sha256(path), "size": path.stat().st_size}


def _reject_absolute_paths(value: Any, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_absolute_paths(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_absolute_paths(child, f"{path}[{index}]")
    elif isinstance(value, str) and value.startswith(("/", "~")):
        raise ValueError(f"public config contains an absolute or home-relative path at {path}")


def _verify_expected_source_hash(manifest: dict[str, Any], actual: str) -> None:
    expected = manifest.get("checkpoint", {}).get("sha256")
    if expected is not None and expected != actual:
        raise ValueError(f"source checkpoint SHA256 is {actual}, expected {expected}")


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output path: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
