"""Paper Stage 2 training entrypoint for LaMP on the four LIBERO suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf
from tqdm import tqdm

from starVLA.checkpoint.io import _find_pt_checkpoint, _load_pt, _validate_top_level_keys, _verify_manifest
from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework


class JsonlLogger:
    def __init__(self, path: Path, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def freeze_for_stage2(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.motion_guidance, model.action_model):
        for parameter in module.parameters():
            parameter.requires_grad = True

    unexpected = {name.split(".", 1)[0] for name, parameter in model.named_parameters() if parameter.requires_grad} - {
        "motion_guidance",
        "action_model",
    }
    if unexpected:
        raise RuntimeError(f"unexpected trainable modules: {sorted(unexpected)}")


def load_motion_expert(model: torch.nn.Module, source: str | None) -> None:
    if not source:
        return
    path = Path(source).expanduser()
    if not path.exists():
        path = Path(
            snapshot_download(
                repo_id=source,
                allow_patterns=[
                    "*.yaml",
                    "*.json",
                    "*.pt",
                    "README.md",
                ],
            )
        )
    if not path.is_dir():
        raise ValueError("motion_expert_checkpoint must be a .pt artifact directory or Hugging Face repository")
    _verify_manifest(path)
    state_dict = _load_pt(_find_pt_checkpoint(path))
    prefix = "motion_head."
    motion_state = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not motion_state:
        raise ValueError("Motion Expert checkpoint has no motion_head.* tensors")
    model.motion_head.load_state_dict(motion_state, strict=True)


def read_statistics(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_stage2_config(config: Any) -> None:
    required_paths = [
        "output_dir",
        "seed",
        "tracking.backend",
        "tracking.project",
        "framework.name",
        "framework.qwenvl.revision",
        "framework.qwenvl.dtype",
        "framework.action_model.future_action_window_size",
        "framework.action_model.num_inference_timesteps",
        "framework.action_model.noise_beta_alpha",
        "framework.action_model.noise_beta_beta",
        "framework.motion_expert.guidance_init_gate",
        "framework.motion_expert.tau_v_infer",
        "datasets.vla_data.data_root_dir",
        "datasets.vla_data.statistics_path",
        "datasets.vla_data.per_device_batch_size",
        "trainer.max_train_steps",
        "trainer.expected_world_size",
        "trainer.enforce_paper_hardware",
        "trainer.gradient_accumulation_steps",
        "trainer.learning_rate",
        "trainer.optimizer.betas",
        "trainer.optimizer.eps",
        "trainer.optimizer.weight_decay",
        "trainer.num_warmup_steps",
        "trainer.lr_scheduler_type",
        "trainer.min_learning_rate",
        "trainer.repeated_diffusion_steps",
        "trainer.gradient_clipping",
        "trainer.logging_frequency",
        "trainer.save_interval",
        "trainer.motion_expert_checkpoint",
        "trainer.frozen_modules",
        "trainer.trainable_modules",
    ]
    missing = [path for path in required_paths if OmegaConf.select(config, path) is None]
    if missing:
        raise ValueError(f"invalid Stage 2 config; missing: {', '.join(missing)}")
    if config.tracking.backend not in {"jsonl", "wandb"}:
        raise ValueError("tracking.backend must be jsonl or wandb")
    if config.framework.name != "LaMP":
        raise ValueError("Stage 2 only supports framework.name=LaMP")
    if config.framework.qwenvl.revision != "ebb281ec70b05090aa6165b016eac8ec08e71b17":
        raise ValueError("Qwen3-VL revision does not match the release contract")
    if config.framework.qwenvl.dtype != "bfloat16":
        raise ValueError("Stage 2 requires framework.qwenvl.dtype=bfloat16")
    if config.framework.action_model.future_action_window_size + 1 != 10:
        raise ValueError("Stage 2 action horizon must be 10")
    if config.framework.action_model.num_inference_timesteps != 10:
        raise ValueError("Stage 2 action inference must use ten Euler steps")
    if [config.framework.action_model.noise_beta_alpha, config.framework.action_model.noise_beta_beta] != [1.5, 1.0]:
        raise ValueError("Stage 2 action time sampling must use Beta(1.5, 1.0)")
    if config.framework.motion_expert.tau_v_infer != 0.1:
        raise ValueError("Stage 2 partial Motion Expert inference requires tau_v_infer=0.1")
    if config.framework.motion_expert.guidance_init_gate != 0.0:
        raise ValueError("the public Stage 2 recipe requires guidance_init_gate=0.0")
    if list(config.trainer.frozen_modules) != ["qwen_vl_interface", "motion_head"]:
        raise ValueError("Stage 2 must freeze Qwen3-VL and Motion Expert")
    if list(config.trainer.trainable_modules) != ["motion_guidance", "action_model"]:
        raise ValueError("Stage 2 may train only Motion Guidance and Action Expert")
    if config.trainer.num_warmup_steps != 0:
        raise ValueError("the released Stage 2 trainer supports zero warmup only")
    if config.trainer.lr_scheduler_type != "cosine":
        raise ValueError("the released implementation default is a cosine scheduler")
    if config.trainer.repeated_diffusion_steps < 1:
        raise ValueError("trainer.repeated_diffusion_steps must be positive")

    if config.trainer.enforce_paper_hardware:
        paper_values = {
            "datasets.vla_data.per_device_batch_size": 32,
            "trainer.expected_world_size": 16,
            "trainer.gradient_accumulation_steps": 1,
            "trainer.learning_rate": 1e-4,
            "trainer.max_train_steps": 15_000,
            "trainer.optimizer.weight_decay": 0.0,
        }
        mismatches = {
            path: (OmegaConf.select(config, path), expected)
            for path, expected in paper_values.items()
            if OmegaConf.select(config, path) != expected
        }
        if list(config.trainer.optimizer.betas) != [0.9, 0.95]:
            mismatches["trainer.optimizer.betas"] = (list(config.trainer.optimizer.betas), [0.9, 0.95])
        if mismatches:
            raise ValueError(f"paper recipe values were changed while hardware enforcement is enabled: {mismatches}")


def save_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: Any,
    statistics_path: Path,
    completed_steps: int,
    output_dir: Path,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        accelerator.wait_for_everyone()
        return

    checkpoint_dir = output_dir / "checkpoints" / f"step-{completed_steps:06d}"
    if checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint directory: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True)

    state_dict = accelerator.get_state_dict(model)
    _validate_top_level_keys(state_dict)
    state_dict = {key: tensor.detach().cpu() for key, tensor in state_dict.items()}
    checkpoint_path = checkpoint_dir / "pytorch_model.pt"
    torch.save(state_dict, checkpoint_path)
    OmegaConf.save(config, checkpoint_dir / "config.yaml")
    shutil.copy2(statistics_path, checkpoint_dir / "dataset_statistics.json")

    files = []
    for path in sorted(checkpoint_dir.iterdir()):
        if path.name == "checkpoint_manifest.json":
            continue
        files.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    checkpoint_record = next(record for record in files if record["name"] == checkpoint_path.name)
    manifest = {
        "format_version": 1,
        "model": "LaMP",
        "training_stage": 2,
        "optimizer_step": completed_steps,
        "paper_recipe_reference": "configs/lamp_stage2_libero_paper.yaml",
        "paper": "arXiv:2603.25399v2",
        "code_commit": code_commit(),
        "frozen_modules": ["qwen_vl_interface", "motion_head"],
        "trainable_modules": ["motion_guidance", "action_model"],
        "optimizer_state_included": False,
        "scheduler_state_included": False,
        "checkpoint": checkpoint_record,
        "torch_load_weights_only": True,
        "training_metadata": {
            "optimizer": "AdamW",
            "optimizer_betas": list(config.trainer.optimizer.betas),
            "learning_rate": config.trainer.learning_rate,
            "weight_decay": config.trainer.optimizer.weight_decay,
            "warmup_steps": config.trainer.num_warmup_steps,
            "scheduler": config.trainer.lr_scheduler_type,
            "repeated_diffusion_steps": config.trainer.repeated_diffusion_steps,
            "per_device_batch_size": config.datasets.vla_data.per_device_batch_size,
            "world_size": accelerator.num_processes,
            "gradient_accumulation_steps": config.trainer.gradient_accumulation_steps,
            "global_batch_size": (
                config.datasets.vla_data.per_device_batch_size
                * accelerator.num_processes
                * config.trainer.gradient_accumulation_steps
            ),
        },
        "files": files,
    }
    with (checkpoint_dir / "checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    accelerator.print(f"saved checkpoint: {checkpoint_dir}")
    accelerator.wait_for_everyone()


def train(config: Any) -> None:
    validate_stage2_config(config)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="wandb" if config.tracking.backend == "wandb" else None,
    )
    set_seed(config.seed + accelerator.process_index)
    output_dir = Path(config.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    if config.trainer.enforce_paper_hardware and accelerator.num_processes != config.trainer.expected_world_size:
        raise RuntimeError(
            f"paper recipe requires {config.trainer.expected_world_size} processes; "
            "set trainer.enforce_paper_hardware=false only for smoke tests"
        )
    global_batch = (
        config.datasets.vla_data.per_device_batch_size
        * accelerator.num_processes
        * config.trainer.gradient_accumulation_steps
    )
    if config.trainer.enforce_paper_hardware and global_batch != 512:
        raise RuntimeError(f"paper recipe global batch must be 512, got {global_batch}")

    model = build_framework(config)
    statistics_path = Path(config.datasets.vla_data.statistics_path)
    model.set_dataset_statistics(read_statistics(statistics_path))
    load_motion_expert(model, config.trainer.motion_expert_checkpoint)
    freeze_for_stage2(model)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.trainer.learning_rate,
        betas=tuple(config.trainer.optimizer.betas),
        eps=config.trainer.optimizer.eps,
        weight_decay=config.trainer.optimizer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.trainer.max_train_steps,
        eta_min=config.trainer.min_learning_rate,
    )
    dataloader = build_dataloader(config)
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)

    if config.tracking.backend == "wandb":
        accelerator.init_trackers(config.tracking.project, config=OmegaConf.to_container(config, resolve=True))
    logger = JsonlLogger(output_dir / "train.jsonl", accelerator.is_main_process)
    progress = tqdm(
        total=config.trainer.max_train_steps,
        disable=not accelerator.is_local_main_process,
        desc="LaMP Stage 2",
    )
    iterator = iter(dataloader)
    completed_steps = 0
    last_saved_step = 0

    while completed_steps < config.trainer.max_train_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)

        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            losses = model(
                batch,
                repeat_steps=config.trainer.repeated_diffusion_steps,
                normalization_key="libero",
            )
            loss = losses["action_loss"]
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable, config.trainer.gradient_clipping)
            optimizer.step()
            scheduler.step()

        if not accelerator.sync_gradients:
            continue
        completed_steps += 1
        progress.update(1)

        if completed_steps % config.trainer.logging_frequency == 0:
            metrics = {
                "step": completed_steps,
                "action_loss": float(accelerator.gather(loss.detach()).mean().item()),
                "learning_rate": scheduler.get_last_lr()[0],
                "global_batch_size": global_batch,
            }
            logger.log(metrics)
            if config.tracking.backend == "wandb":
                accelerator.log(metrics, step=completed_steps)
            progress.set_postfix(loss=f"{metrics['action_loss']:.4f}")

        if completed_steps % config.trainer.save_interval == 0:
            save_checkpoint(
                accelerator,
                model,
                config,
                statistics_path,
                completed_steps,
                output_dir,
            )
            last_saved_step = completed_steps

    if last_saved_step != completed_steps:
        save_checkpoint(accelerator, model, config, statistics_path, completed_steps, output_dir)
    progress.close()
    if config.tracking.backend == "wandb":
        accelerator.end_training()


def parse_config() -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        default="configs/lamp_stage2_libero_paper.yaml",
        help="OmegaConf YAML followed by optional --path.to.key value overrides",
    )
    arguments, overrides = parser.parse_known_args()
    config = OmegaConf.load(arguments.config_yaml)
    return OmegaConf.merge(config, OmegaConf.from_dotlist(dotlist(overrides)))


def dotlist(arguments: list[str]) -> list[str]:
    normalized = []
    index = 0
    while index < len(arguments):
        token = arguments[index]
        if not token.startswith("--"):
            raise ValueError(f"unexpected override token: {token}")
        token = token[2:]
        if "=" in token:
            normalized.append(token)
            index += 1
            continue
        if index + 1 >= len(arguments) or arguments[index + 1].startswith("--"):
            raise ValueError(f"override --{token} requires a value")
        normalized.append(f"{token}={arguments[index + 1]}")
        index += 2
    return normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_commit() -> str | None:
    value = os.environ.get("GIT_COMMIT")
    if value:
        return value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    train(parse_config())


if __name__ == "__main__":
    main()
