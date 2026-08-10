"""Evaluate a LaMP checkpoint on the four LIBERO suites used by the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from starVLA import load_policy
from starVLA.constants import LIBERO_COMMIT

LIBERO_RESOLUTION = 256
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

LOGGER = logging.getLogger("lamp.libero")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Hugging Face repo, .pt artifact directory, or .pt file")
    parser.add_argument("--suite", choices=(*SUITES, "all"), default="all")
    parser.add_argument("--task-id", type=int, action="append", help="Evaluate only this task; may be repeated")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    parser.add_argument("--max-steps", type=int, help="Override the suite-specific episode limit")
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/libero_episodes.jsonl"))
    parser.add_argument("--append", action="store_true", help="Append to an existing JSONL file instead of refusing")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--unnorm-key", default="libero")
    parser.add_argument("--libero-source", type=Path, help="Path to the pinned LIBERO git checkout")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes_per_task < 1:
        raise ValueError("--episodes-per-task must be positive")
    if args.wait_steps < 0:
        raise ValueError("--wait-steps cannot be negative")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    libero_source = _resolve_libero_source(args.libero_source)
    _require_libero_commit(libero_source)
    checkpoint_source, checkpoint_sha256 = _resolve_checkpoint(args.checkpoint)
    code_commit = _code_commit()
    policy = load_policy(
        checkpoint_source,
        device=args.device,
        dtype=args.dtype,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suites = SUITES if args.suite == "all" else (args.suite,)
    total_episodes = 0
    total_successes = 0
    output_mode = "a" if args.append else "x"
    with args.output.open(output_mode, encoding="utf-8") as output:
        for suite in suites:
            episodes, successes = _evaluate_suite(
                policy=policy,
                suite=suite,
                suite_index=SUITES.index(suite),
                task_ids=args.task_id,
                episodes_per_task=args.episodes_per_task,
                max_steps=args.max_steps or MAX_STEPS[suite],
                wait_steps=args.wait_steps,
                base_seed=args.seed,
                unnorm_key=args.unnorm_key,
                device=args.device,
                checkpoint_sha256=checkpoint_sha256,
                code_commit=code_commit,
                output=output,
            )
            total_episodes += episodes
            total_successes += successes
            LOGGER.info("%s: %d/%d (%.2f%%)", suite, successes, episodes, 100 * successes / episodes)

    LOGGER.info("overall: %d/%d (%.2f%%)", total_successes, total_episodes, 100 * total_successes / total_episodes)
    return 0


def _evaluate_suite(
    *,
    policy: Any,
    suite: str,
    suite_index: int,
    task_ids: list[int] | None,
    episodes_per_task: int,
    max_steps: int,
    wait_steps: int,
    base_seed: int,
    unnorm_key: str,
    device: str,
    checkpoint_sha256: str,
    code_commit: str,
    output: Any,
) -> tuple[int, int]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_suite = benchmark.get_benchmark_dict()[suite]()
    selected_task_ids = list(range(task_suite.n_tasks)) if task_ids is None else task_ids
    invalid = [task_id for task_id in selected_task_ids if not 0 <= task_id < task_suite.n_tasks]
    if invalid:
        raise ValueError(f"invalid task IDs for {suite}: {invalid}")

    completed = 0
    successes = 0
    for task_id in selected_task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=LIBERO_RESOLUTION,
            camera_widths=LIBERO_RESOLUTION,
        )
        try:
            if episodes_per_task > len(initial_states):
                raise ValueError(
                    f"{suite} task {task_id} provides only {len(initial_states)} fixed initial states, "
                    f"but {episodes_per_task} episodes were requested"
                )
            env.seed(base_seed)
            for episode_index in range(episodes_per_task):
                episode_seed = base_seed + suite_index * 100_000 + task_id * 1_000 + episode_index
                started = time.monotonic()
                success = _run_episode(
                    policy=policy,
                    env=env,
                    initial_state=initial_states[episode_index],
                    instruction=str(task.language),
                    seed=episode_seed,
                    max_steps=max_steps,
                    wait_steps=wait_steps,
                    unnorm_key=unnorm_key,
                    device=device,
                )
                record = {
                    "suite": suite,
                    "task": str(task.language),
                    "task_id": task_id,
                    "episode": episode_index,
                    "seed": episode_seed,
                    "environment_seed": base_seed,
                    "success": success,
                    "checkpoint_sha256": checkpoint_sha256,
                    "code_commit": code_commit,
                    "libero_commit": LIBERO_COMMIT,
                    "elapsed_seconds": round(time.monotonic() - started, 6),
                }
                output.write(json.dumps(record, sort_keys=True) + "\n")
                output.flush()
                completed += 1
                successes += int(success)
                LOGGER.info("%s task=%d episode=%d success=%s", suite, task_id, episode_index, success)
        finally:
            env.close()
    return completed, successes


def _run_episode(
    *,
    policy: Any,
    env: Any,
    initial_state: np.ndarray,
    instruction: str,
    seed: int,
    max_steps: int,
    wait_steps: int,
    unnorm_key: str,
    device: str,
) -> bool:
    env.reset()
    observation = env.set_init_state(initial_state)
    for _ in range(wait_steps):
        observation, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
        if done:
            return True

    generator = torch.Generator(device=device).manual_seed(seed)
    steps = 0
    while steps < max_steps:
        example = _observation_to_example(observation, instruction)
        result = policy.predict_action([example], unnorm_key=unnorm_key, generator=generator)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (1, 10, 7):
            raise ValueError(f"policy returned actions with shape {actions.shape}, expected (1, 10, 7)")

        for action in actions[0]:
            libero_action = np.array(action, dtype=np.float32, copy=True)
            libero_action[-1] = 1.0 - 2.0 * float(libero_action[-1] > 0.5)
            observation, _, done, _ = env.step(libero_action.tolist())
            steps += 1
            if done:
                return True
            if steps >= max_steps:
                break
    return False


def _observation_to_example(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    primary = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        [
            np.asarray(observation["robot0_eef_pos"], dtype=np.float32),
            _quat_to_axis_angle(observation["robot0_eef_quat"]),
            np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32),
        ]
    )
    if state.shape != (8,):
        raise ValueError(f"LIBERO observation produced state shape {state.shape}, expected (8,)")
    return {
        "image": [Image.fromarray(primary), Image.fromarray(wrist)],
        "lang": instruction,
        "state": state,
    }


def _quat_to_axis_angle(quaternion: Any) -> np.ndarray:
    quaternion = np.array(quaternion, dtype=np.float64, copy=True)
    quaternion[3] = np.clip(quaternion[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quaternion[3] ** 2))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.acos(float(quaternion[3]))
    return (quaternion[:3] * angle / denominator).astype(np.float32)


def _resolve_libero_source(explicit_source: Path | None) -> Path:
    if explicit_source is not None:
        return explicit_source.expanduser().resolve()
    import libero

    package_path = Path(libero.__file__).resolve()
    for parent in package_path.parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("LIBERO must be installed from the pinned git checkout; pass --libero-source to its repository")


def _require_libero_commit(source: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.strip()
    if actual != LIBERO_COMMIT:
        raise RuntimeError(f"LIBERO checkout is {actual}, expected {LIBERO_COMMIT}")


def _resolve_checkpoint(checkpoint: str) -> tuple[str | Path, str]:
    source = Path(checkpoint).expanduser()
    if source.exists():
        resolved = source.resolve()
    else:
        resolved = Path(
            snapshot_download(
                repo_id=checkpoint,
                allow_patterns=["*.yaml", "*.json", "*.pt", "README.md"],
            )
        )
    return resolved, _checkpoint_digest(resolved)


def _checkpoint_digest(source: Path) -> str:
    if source.is_file():
        return _sha256(source)

    manifest_path = source / "checkpoint_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("checkpoint_sha256"):
            return str(manifest["checkpoint_sha256"])
        checkpoint_record = manifest.get("checkpoint", {})
        if checkpoint_record.get("sha256"):
            return str(checkpoint_record["sha256"])

    weights = sorted(source.glob("*.pt"))
    if len(weights) != 1:
        raise ValueError(f"expected exactly one .pt checkpoint in {source}, found {len(weights)}")
    return _sha256(weights[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _code_commit() -> str:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / ".git").exists():
            result = subprocess.run(
                ["git", "-C", str(parent), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
