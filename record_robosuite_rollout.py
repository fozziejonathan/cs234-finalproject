from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise argparse.ArgumentTypeError("Hidden sizes must contain at least one integer.")
    return sizes


def _coerce_hidden_sizes(value: object, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    if isinstance(value, str):
        return parse_hidden_sizes(value)
    raise ValueError(f"Unsupported hidden size type: {type(value)!r}")


def _extract_frame(obs: object, camera_name: str, flip_vertical: bool) -> np.ndarray:
    if not isinstance(obs, dict):
        raise ValueError("Expected dict observation from Robosuite camera rollout.")

    preferred_key = f"{camera_name}_image"
    if preferred_key in obs:
        frame = np.asarray(obs[preferred_key])
    else:
        image_keys = [key for key in sorted(obs) if key.endswith("_image")]
        if not image_keys:
            raise ValueError(
                "No image keys found in observation. Ensure use_camera_obs=True with has_offscreen_renderer=True."
            )
        frame = np.asarray(obs[image_keys[0]])

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)
    if frame.ndim != 3:
        raise ValueError(f"Unexpected image shape: {frame.shape}")
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] > 3:
        frame = frame[..., :3]

    if frame.dtype.kind == "f":
        frame = np.clip(frame, 0.0, 1.0) * 255.0
    frame = np.clip(frame, 0, 255).astype(np.uint8)

    if flip_vertical:
        frame = np.flipud(frame)
    return frame


def _save_gif(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    duration_ms = max(1, int(round(1000.0 / max(1, fps))))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def _save_mp4(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(max(1, fps)),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    try:
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()


def _save_video(frames: list[np.ndarray], output_path: Path, fps: int) -> None:
    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        _save_gif(frames, output_path, fps=fps)
        return
    if suffix == ".mp4":
        _save_mp4(frames, output_path, fps=fps)
        return
    raise ValueError(f"Unsupported output suffix `{suffix}`. Use .gif or .mp4")


def _episode_output_path(base_output: Path, episode_idx: int, num_episodes: int) -> Path:
    if num_episodes == 1:
        return base_output
    return base_output.with_name(f"{base_output.stem}_ep{episode_idx + 1:03d}{base_output.suffix}")


def _coerce_reward_shaping(config: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.reward_shaping:
        return True
    if args.sparse_reward:
        return False
    if "reward_shaping" in config:
        return bool(config["reward_shaping"])
    if "sparse_reward" in config:
        return not bool(config["sparse_reward"])
    return True


def _extract_done_step(step_out: object) -> tuple[object, float, bool, dict[str, Any]]:
    if not isinstance(step_out, tuple):
        raise ValueError(f"Unexpected step output type: {type(step_out)!r}")
    if len(step_out) == 4:
        obs, reward, done, info = step_out
        return obs, float(reward), bool(done), info if isinstance(info, dict) else {}
    if len(step_out) == 5:
        obs, reward, terminated, truncated, info = step_out
        return obs, float(reward), bool(terminated or truncated), info if isinstance(info, dict) else {}
    raise ValueError(f"Unexpected step output length: {len(step_out)}")


def _extract_success(info: dict[str, Any], env: object) -> bool:
    for key in ("success", "is_success", "task_success"):
        if key in info:
            try:
                return bool(info[key])
            except Exception:
                pass
    check_success = getattr(env, "_check_success", None)
    if callable(check_success):
        try:
            return bool(check_success())
        except Exception:
            pass
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record Robosuite rollouts from a saved RLHF/DPO checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--output", type=str, required=True, help="Video output (.gif or .mp4).")
    parser.add_argument("--metrics-json", type=str, default="", help="Optional metrics JSON output path.")

    parser.add_argument("--env-name", type=str, default="", help="Override environment name from checkpoint config.")
    parser.add_argument("--robot", type=str, default="", help="Override robot name from checkpoint config.")
    parser.add_argument("--seed", type=int, default=-1, help="Seed override. Default uses checkpoint seed.")

    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 uses checkpoint max_steps.")
    parser.add_argument("--horizon", type=int, default=0, help="0 uses checkpoint horizon.")
    parser.add_argument("--control-freq", type=int, default=0, help="0 uses checkpoint control_freq.")
    parser.add_argument("--hard-reset", action="store_true", help="Force hard reset on each episode.")
    parser.add_argument("--no-hard-reset", action="store_true", help="Force disabling hard reset.")
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--sparse-reward", action="store_true")

    parser.add_argument("--camera-name", type=str, default="agentview")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--flip-vertical", dest="flip_vertical", action="store_true")
    parser.add_argument("--no-flip-vertical", dest="flip_vertical", action="store_false")
    parser.set_defaults(flip_vertical=True)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy actions.")
    parser.add_argument("--policy-hidden-sizes", type=parse_hidden_sizes, default=())
    parser.add_argument("--init-log-std", type=float, default=float("nan"))
    parser.add_argument(
        "--robosuite-log-level",
        type=str,
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import robosuite as suite
    from robosuite_pref_learning import (
        GaussianPolicy,
        configure_robosuite_logging,
        flatten_obs,
        set_seed,
    )

    configure_robosuite_logging(args.robosuite_log_level)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    if not isinstance(config, dict):
        config = {}

    benchmark_name = str(config.get("benchmark", "robosuite")).lower()
    if benchmark_name != "robosuite":
        raise SystemExit(
            f"Checkpoint benchmark is `{benchmark_name}`. "
            "This recorder currently supports only Robosuite checkpoints."
        )

    policy_state = checkpoint.get("policy_state_dict")
    if policy_state is None:
        raise SystemExit("Checkpoint does not contain `policy_state_dict`.")

    env_name = args.env_name or str(config.get("env_name", "Lift"))
    robot = args.robot or str(config.get("robot", "Panda"))

    if args.seed >= 0:
        seed = args.seed
    else:
        seed = int(config.get("seed", 0))
    set_seed(seed)

    horizon = int(config.get("horizon", 1000)) if args.horizon <= 0 else args.horizon
    control_freq = int(config.get("control_freq", 20)) if args.control_freq <= 0 else args.control_freq
    default_max_steps = int(config.get("max_steps", 300))
    max_steps = default_max_steps if args.max_steps <= 0 else args.max_steps

    hard_reset = bool(config.get("hard_reset", False))
    if args.hard_reset:
        hard_reset = True
    if args.no_hard_reset:
        hard_reset = False

    reward_shaping = _coerce_reward_shaping(config, args)

    env_kwargs: dict[str, Any] = {
        "env_name": env_name,
        "robots": robot,
        "has_renderer": False,
        "has_offscreen_renderer": True,
        "use_camera_obs": True,
        "camera_names": args.camera_name,
        "camera_heights": args.height,
        "camera_widths": args.width,
        "reward_shaping": reward_shaping,
        "control_freq": control_freq,
        "hard_reset": hard_reset,
    }
    if horizon > 0:
        env_kwargs["horizon"] = horizon

    env = suite.make(**env_kwargs)
    try:
        first_obs = env.reset()
        obs_vec = flatten_obs(first_obs)
        obs_dim = int(obs_vec.shape[0])
        action_low, action_high = env.action_spec
        act_dim = int(action_low.shape[0])

        checkpoint_hidden = _coerce_hidden_sizes(config.get("policy_hidden_sizes"), default=(256, 256))
        hidden_sizes = args.policy_hidden_sizes if args.policy_hidden_sizes else checkpoint_hidden
        checkpoint_log_std = float(config.get("init_log_std", -0.5))
        init_log_std = checkpoint_log_std if np.isnan(args.init_log_std) else float(args.init_log_std)

        device = torch.device(args.device)
        policy = GaussianPolicy(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=hidden_sizes,
            action_low=np.asarray(action_low, dtype=np.float32),
            action_high=np.asarray(action_high, dtype=np.float32),
            init_log_std=init_log_std,
        ).to(device)
        policy.load_state_dict(policy_state)
        policy.eval()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path = (
            Path(args.metrics_json)
            if args.metrics_json
            else output_path.with_name(f"{output_path.stem}_metrics.json")
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        summaries: list[dict[str, Any]] = []
        for episode_idx in range(args.episodes):
            obs = env.reset()
            obs_vec = flatten_obs(obs)
            frames = [_extract_frame(obs, camera_name=args.camera_name, flip_vertical=args.flip_vertical)]

            total_reward = 0.0
            success = False
            num_steps = 0
            for step_idx in range(max_steps):
                action, _ = policy.act(obs_vec, deterministic=not args.stochastic)
                step_out = env.step(action)
                next_obs, reward, done, info = _extract_done_step(step_out)
                total_reward += float(reward)
                success = success or _extract_success(info, env)
                num_steps = step_idx + 1

                frames.append(_extract_frame(next_obs, camera_name=args.camera_name, flip_vertical=args.flip_vertical))
                obs_vec = flatten_obs(next_obs)
                if done:
                    break

            ep_output = _episode_output_path(output_path, episode_idx=episode_idx, num_episodes=args.episodes)
            _save_video(frames, ep_output, fps=args.fps if args.fps > 0 else control_freq)

            episode_summary = {
                "episode": episode_idx + 1,
                "video_path": str(ep_output),
                "num_steps": num_steps,
                "total_reward": total_reward,
                "success": float(success),
            }
            summaries.append(episode_summary)
            print(
                f"[record episode {episode_idx + 1:03d}] steps={num_steps} "
                f"return={total_reward:.3f} success={float(success):.3f} video={ep_output}"
            )

        payload = {
            "checkpoint": str(checkpoint_path),
            "env_name": env_name,
            "robot": robot,
            "camera_name": args.camera_name,
            "episodes": summaries,
        }
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON to {metrics_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
