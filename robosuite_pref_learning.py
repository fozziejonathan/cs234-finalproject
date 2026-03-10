from __future__ import annotations

import math
import random
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import robosuite as suite
except ModuleNotFoundError:
    suite = None


@dataclass
class Trajectory:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    log_probs: np.ndarray
    true_return: float
    success: float


@dataclass
class PPOBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    learned_rewards: torch.Tensor


@dataclass
class BenchmarkEnv:
    env: object
    benchmark: str
    task_name: str
    action_low: np.ndarray
    action_high: np.ndarray
    horizon: int
    success_fn: Callable[[dict[str, Any]], bool] | None = None

    @property
    def action_spec(self) -> tuple[np.ndarray, np.ndarray]:
        return self.action_low, self.action_high

    def reset(self) -> object:
        reset_out = self.env.reset()
        if isinstance(reset_out, tuple):
            return reset_out[0]
        return reset_out

    def step(self, action: np.ndarray) -> tuple[object, float, bool, dict[str, Any]]:
        step_out = self.env.step(action)
        if not isinstance(step_out, tuple):
            raise ValueError(f"Unexpected step output for benchmark `{self.benchmark}`: {type(step_out)!r}")

        if len(step_out) == 4:
            obs, reward, done, info = step_out
            return obs, float(reward), bool(done), _coerce_info_dict(info)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            return obs, float(reward), bool(terminated or truncated), _coerce_info_dict(info)

        raise ValueError(
            f"Unexpected step tuple length for benchmark `{self.benchmark}`: {len(step_out)}"
        )

    def is_success(self, info: dict[str, Any]) -> bool:
        if self.success_fn is not None:
            try:
                return bool(self.success_fn(info))
            except Exception:
                pass

        for key in ("success", "is_success", "task_success"):
            if key in info:
                try:
                    return bool(info[key])
                except Exception:
                    continue

        for method_name in ("_check_success", "check_success"):
            checker = getattr(self.env, method_name, None)
            if callable(checker):
                try:
                    return bool(checker())
                except Exception:
                    continue
        return False

    def close(self) -> None:
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()


def _coerce_info_dict(info: object) -> dict[str, Any]:
    if isinstance(info, dict):
        return info
    return {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module] = nn.Tanh,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.mu_net = build_mlp(obs_dim, act_dim, hidden_sizes)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std, dtype=torch.float32))
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mu_net(obs)
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def sample(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(obs)
        action = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        return dist.log_prob(action).sum(dim=-1)

    def entropy(self, obs: torch.Tensor) -> torch.Tensor:
        dist = self.distribution(obs)
        return dist.entropy().sum(dim=-1)

    def act(self, obs_vec: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float]:
        obs_tensor = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.action_low.device).unsqueeze(0)
        with torch.no_grad():
            dist = self.distribution(obs_tensor)
            sampled_action = dist.mean if deterministic else dist.rsample()
            clipped_action = torch.clamp(sampled_action, min=self.action_low, max=self.action_high)
            log_prob = dist.log_prob(clipped_action).sum(dim=-1)
        action_np = clipped_action.squeeze(0).cpu().numpy()
        return action_np.astype(np.float32), float(log_prob.item())


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.value_net = build_mlp(obs_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value_net(obs).squeeze(-1)


class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.reward_net = build_mlp(obs_dim + act_dim, 1, hidden_sizes)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        features = torch.cat([obs, action], dim=-1)
        return self.reward_net(features).squeeze(-1)

    def trajectory_score(self, obs: torch.Tensor, action: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        per_step = self.forward(obs, action)
        return per_step.mean() if normalize else per_step.sum()


def _to_flat_numeric_array(value: object) -> np.ndarray | None:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
        return None
    return np.asarray(array, dtype=np.float32).reshape(-1)


def flatten_obs(obs: object) -> np.ndarray:
    if isinstance(obs, dict):
        parts: list[np.ndarray] = []
        for key in sorted(obs):
            key_lower = key.lower()
            if "image" in key_lower or "depth" in key_lower or "segmentation" in key_lower:
                continue
            flat = _to_flat_numeric_array(obs[key])
            if flat is None or flat.size == 0:
                continue
            parts.append(flat)
        if not parts:
            raise ValueError("No low-dimensional observations found. Disable camera observations for this pipeline.")
        return np.concatenate(parts, axis=0).astype(np.float32)

    if isinstance(obs, (list, tuple)):
        parts = []
        for value in obs:
            flat = _to_flat_numeric_array(value)
            if flat is not None and flat.size > 0:
                parts.append(flat)
        if not parts:
            raise ValueError("Observation tuple/list does not include numeric values.")
        return np.concatenate(parts, axis=0).astype(np.float32)

    flat = _to_flat_numeric_array(obs)
    if flat is None or flat.size == 0:
        raise ValueError(f"Unsupported observation type: {type(obs)!r}")
    return flat.astype(np.float32)


def _instantiate_with_optional_seed(factory: Callable[..., object], seed: int) -> object:
    try:
        return factory(seed=seed)
    except TypeError:
        return factory()


def _coerce_horizon(horizon: int | None, fallback: int) -> int:
    if horizon is not None and horizon > 0:
        return int(horizon)
    return int(fallback)


def _wrap_env(
    env: object,
    benchmark: str,
    task_name: str,
    horizon: int,
    success_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> BenchmarkEnv:
    action_space = getattr(env, "action_space", None)
    if action_space is None or not hasattr(action_space, "low") or not hasattr(action_space, "high"):
        raise RuntimeError(
            f"`{benchmark}` environment `{task_name}` must expose action_space.low/high for continuous control."
        )

    action_low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if action_low.shape != action_high.shape or action_low.size == 0:
        raise RuntimeError(
            f"Invalid action bounds for `{benchmark}` environment `{task_name}`: {action_low.shape} vs {action_high.shape}"
        )

    return BenchmarkEnv(
        env=env,
        benchmark=benchmark,
        task_name=task_name,
        action_low=action_low,
        action_high=action_high,
        horizon=horizon,
        success_fn=success_fn,
    )


def _metaworld_task_candidates(task_name: str) -> list[str]:
    candidates = [task_name.strip()]
    if task_name.endswith("-v3"):
        candidates.append(f"{task_name[:-3]}-v2")
    elif task_name.endswith("-v2"):
        candidates.append(f"{task_name[:-3]}-v3")
    deduped: list[str] = []
    for name in candidates:
        if name and name not in deduped:
            deduped.append(name)
    return deduped


def _make_robosuite_env(
    env_name: str,
    robot: str,
    reward_shaping: bool,
    control_freq: int,
    horizon: int | None,
    hard_reset: bool,
) -> BenchmarkEnv:
    if suite is None:
        raise ModuleNotFoundError("robosuite")

    kwargs = {
        "env_name": env_name,
        "robots": robot,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "use_camera_obs": False,
        "reward_shaping": reward_shaping,
        "control_freq": control_freq,
        "hard_reset": hard_reset,
    }
    if horizon is not None and horizon > 0:
        kwargs["horizon"] = horizon

    env = suite.make(**kwargs)
    action_low, action_high = env.action_spec
    resolved_horizon = _coerce_horizon(horizon, fallback=int(getattr(env, "horizon", 500)))
    return BenchmarkEnv(
        env=env,
        benchmark="robosuite",
        task_name=env_name,
        action_low=np.asarray(action_low, dtype=np.float32),
        action_high=np.asarray(action_high, dtype=np.float32),
        horizon=resolved_horizon,
    )


def _make_metaworld_env(
    env_name: str,
    horizon: int | None,
    seed: int,
) -> BenchmarkEnv:
    try:
        import metaworld
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("metaworld") from exc

    errors: list[str] = []
    selected_name = env_name
    env = None
    candidates = _metaworld_task_candidates(env_name)

    if hasattr(metaworld, "MT1"):
        for candidate_name in candidates:
            try:
                mt1 = metaworld.MT1(candidate_name, seed=seed)
                train_cls = mt1.train_classes.get(candidate_name)
                if train_cls is None:
                    continue
                env = _instantiate_with_optional_seed(train_cls, seed=seed)
                train_tasks = getattr(mt1, "train_tasks", [])
                if train_tasks and hasattr(env, "set_task"):
                    env.set_task(train_tasks[seed % len(train_tasks)])
                selected_name = candidate_name
                break
            except Exception as exc:
                errors.append(f"MT1({candidate_name}): {exc}")

    if env is None:
        for candidate_name in candidates:
            for registry_name in (
                "ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE",
                "ALL_V3_ENVIRONMENTS_GOAL_HIDDEN",
                "ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE",
                "ALL_V2_ENVIRONMENTS_GOAL_HIDDEN",
            ):
                registry = getattr(metaworld, registry_name, None)
                if not isinstance(registry, dict) or candidate_name not in registry:
                    continue
                try:
                    env = _instantiate_with_optional_seed(registry[candidate_name], seed=seed)
                    selected_name = candidate_name
                    break
                except Exception as exc:
                    errors.append(f"{registry_name}[{candidate_name}]: {exc}")
            if env is not None:
                break

    if env is None:
        error_text = "; ".join(errors) if errors else "No matching task constructor found."
        raise RuntimeError(f"Could not create Meta-World environment `{env_name}`. {error_text}")

    reset_fn = getattr(env, "reset", None)
    if callable(reset_fn):
        try:
            reset_fn(seed=seed)
        except TypeError:
            pass
    seed_fn = getattr(env, "seed", None)
    if callable(seed_fn):
        try:
            seed_fn(seed)
        except Exception:
            pass

    resolved_horizon = _coerce_horizon(horizon, fallback=int(getattr(env, "max_path_length", 500)))
    return _wrap_env(
        env=env,
        benchmark="metaworld",
        task_name=selected_name,
        horizon=resolved_horizon,
        success_fn=lambda info: bool(info.get("success", False)),
    )


def _make_libero_env(
    env_name: str,
    horizon: int | None,
    seed: int,
    libero_suite: str,
    libero_task_id: int,
) -> BenchmarkEnv:
    gym = None
    try:
        import gymnasium as gymnasium_mod

        gym = gymnasium_mod
    except ModuleNotFoundError:
        try:
            import gym as gym_mod

            gym = gym_mod
        except ModuleNotFoundError:
            gym = None

    env = None
    errors: list[str] = []
    candidate_ids = [env_name]

    if gym is not None:
        for candidate in candidate_ids:
            if not candidate:
                continue
            try:
                env = gym.make(candidate)
                break
            except Exception as exc:
                errors.append(f"gym.make({candidate!r}): {exc}")

    if env is None:
        try:
            from libero.libero import benchmark as libero_benchmark
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("libero") from exc

        benchmark_dict_fn = getattr(libero_benchmark, "get_benchmark_dict", None)
        if callable(benchmark_dict_fn):
            benchmark_dict = benchmark_dict_fn()
            if libero_suite in benchmark_dict:
                try:
                    benchmark_instance = benchmark_dict[libero_suite]()
                    task = benchmark_instance.get_task(libero_task_id)
                    for attr in ("env_name", "name", "task_name"):
                        task_name_value = getattr(task, attr, None)
                        if isinstance(task_name_value, str) and task_name_value not in candidate_ids:
                            candidate_ids.append(task_name_value)
                except Exception as exc:
                    errors.append(f"LIBERO benchmark metadata lookup failed: {exc}")

        if gym is not None:
            for candidate in candidate_ids:
                if not candidate:
                    continue
                try:
                    env = gym.make(candidate)
                    break
                except Exception as exc:
                    errors.append(f"gym.make({candidate!r}): {exc}")

    if env is None:
        install_hint = (
            "Install `libero`, and either `gymnasium` or `gym`, then provide a valid `--env-name` "
            "or (`--libero-suite`, `--libero-task-id`) mapping."
        )
        error_text = "; ".join(errors) if errors else "No env factory succeeded."
        raise RuntimeError(f"Could not create LIBERO environment. {install_hint} Details: {error_text}")

    reset_fn = getattr(env, "reset", None)
    if callable(reset_fn):
        try:
            reset_fn(seed=seed)
        except TypeError:
            pass

    fallback_horizon = 500
    spec = getattr(env, "spec", None)
    if spec is not None and getattr(spec, "max_episode_steps", None) is not None:
        fallback_horizon = int(spec.max_episode_steps)
    elif hasattr(env, "max_path_length"):
        fallback_horizon = int(getattr(env, "max_path_length"))

    resolved_horizon = _coerce_horizon(horizon, fallback=fallback_horizon)
    chosen_name = env_name if env_name else f"{libero_suite}:{libero_task_id}"
    return _wrap_env(
        env=env,
        benchmark="libero",
        task_name=chosen_name,
        horizon=resolved_horizon,
        success_fn=lambda info: bool(info.get("success", info.get("task_success", False))),
    )


def make_env(
    benchmark: str,
    env_name: str,
    robot: str,
    reward_shaping: bool,
    control_freq: int,
    horizon: int | None,
    hard_reset: bool,
    seed: int,
    libero_suite: str,
    libero_task_id: int,
) -> BenchmarkEnv:
    benchmark_name = benchmark.lower().strip()
    if benchmark_name == "robosuite":
        return _make_robosuite_env(
            env_name=env_name,
            robot=robot,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            horizon=horizon,
            hard_reset=hard_reset,
        )
    if benchmark_name == "metaworld":
        return _make_metaworld_env(
            env_name=env_name,
            horizon=horizon,
            seed=seed,
        )
    if benchmark_name == "libero":
        return _make_libero_env(
            env_name=env_name,
            horizon=horizon,
            seed=seed,
            libero_suite=libero_suite,
            libero_task_id=libero_task_id,
        )
    raise ValueError(f"Unsupported benchmark `{benchmark}`. Choose from: robosuite, metaworld, libero.")


def configure_robosuite_logging(level_name: str) -> None:
    if suite is None:
        return
    level = getattr(logging, level_name.upper(), logging.WARNING)
    logging.getLogger("robosuite").setLevel(level)
    logging.getLogger("robosuite_logs").setLevel(level)
    try:
        from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

        ROBOSUITE_DEFAULT_LOGGER.setLevel(level)
        for handler in ROBOSUITE_DEFAULT_LOGGER.handlers:
            handler.setLevel(level)
    except Exception:
        pass


def rollout_episode(
    env: BenchmarkEnv,
    policy: GaussianPolicy,
    max_steps: int,
    deterministic: bool,
) -> Trajectory:
    obs = env.reset()
    obs_vec = flatten_obs(obs)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    log_probs: list[float] = []
    success = False

    for _ in range(max_steps):
        action, log_prob = policy.act(obs_vec, deterministic=deterministic)
        next_obs, reward, done, info = env.step(action)

        observations.append(obs_vec)
        actions.append(action)
        rewards.append(float(reward))
        log_probs.append(float(log_prob))

        success = success or env.is_success(info)

        obs_vec = flatten_obs(next_obs)
        if done:
            break

    return Trajectory(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        log_probs=np.asarray(log_probs, dtype=np.float32),
        true_return=float(np.sum(rewards, dtype=np.float32)),
        success=float(success),
    )


def collect_trajectories(
    env: BenchmarkEnv,
    policy: GaussianPolicy,
    num_episodes: int,
    max_steps: int,
    deterministic: bool = False,
) -> list[Trajectory]:
    trajectories: list[Trajectory] = []
    for _ in range(num_episodes):
        trajectories.append(rollout_episode(env, policy, max_steps=max_steps, deterministic=deterministic))
    return trajectories


def sample_preference_pairs(
    trajectories: Sequence[Trajectory],
    num_pairs: int,
    noise_prob: float,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    if len(trajectories) < 2:
        return []

    returns = np.asarray([traj.true_return for traj in trajectories], dtype=np.float32)
    pairs: list[tuple[int, int]] = []
    for _ in range(num_pairs):
        idx_a, idx_b = rng.choice(len(trajectories), size=2, replace=False)
        winner, loser = (idx_a, idx_b) if returns[idx_a] >= returns[idx_b] else (idx_b, idx_a)
        if noise_prob > 0 and rng.random() < noise_prob:
            winner, loser = loser, winner
        pairs.append((int(winner), int(loser)))
    return pairs


def trajectories_to_tensors(
    trajectories: Sequence[Trajectory],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.as_tensor(traj.observations, dtype=torch.float32, device=device),
            torch.as_tensor(traj.actions, dtype=torch.float32, device=device),
        )
        for traj in trajectories
    ]


def train_reward_model(
    reward_model: RewardModel,
    optimizer: torch.optim.Optimizer,
    trajectory_tensors: Sequence[tuple[torch.Tensor, torch.Tensor]],
    preference_pairs: Sequence[tuple[int, int]],
    epochs: int,
    batch_size: int,
) -> float:
    if not preference_pairs:
        return math.nan

    reward_model.train()
    running_losses: list[float] = []

    pair_indices = list(preference_pairs)
    for _ in range(epochs):
        random.shuffle(pair_indices)
        for start in range(0, len(pair_indices), batch_size):
            batch_pairs = pair_indices[start : start + batch_size]
            pair_losses: list[torch.Tensor] = []
            for winner_idx, loser_idx in batch_pairs:
                winner_obs, winner_act = trajectory_tensors[winner_idx]
                loser_obs, loser_act = trajectory_tensors[loser_idx]
                winner_score = reward_model.trajectory_score(winner_obs, winner_act, normalize=True)
                loser_score = reward_model.trajectory_score(loser_obs, loser_act, normalize=True)
                pair_losses.append(-F.logsigmoid(winner_score - loser_score))

            if not pair_losses:
                continue

            loss = torch.stack(pair_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
            optimizer.step()
            running_losses.append(float(loss.item()))

    return float(np.mean(running_losses)) if running_losses else math.nan


def build_ppo_batch(
    trajectories: Sequence[Trajectory],
    reward_model: RewardModel,
    value_net: ValueNetwork,
    gamma: float,
    gae_lambda: float,
    device: torch.device,
) -> PPOBatch | None:
    reward_model.eval()
    value_net.eval()

    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    old_log_probs: list[torch.Tensor] = []
    returns: list[torch.Tensor] = []
    advantages: list[torch.Tensor] = []
    learned_rewards: list[torch.Tensor] = []

    for traj in trajectories:
        if len(traj.observations) == 0:
            continue

        obs_tensor = torch.as_tensor(traj.observations, dtype=torch.float32, device=device)
        act_tensor = torch.as_tensor(traj.actions, dtype=torch.float32, device=device)
        old_log_prob_tensor = torch.as_tensor(traj.log_probs, dtype=torch.float32, device=device)

        with torch.no_grad():
            reward_pred = reward_model(obs_tensor, act_tensor).cpu().numpy()
            value_pred = value_net(obs_tensor).cpu().numpy()

        traj_advantages = np.zeros_like(reward_pred, dtype=np.float32)
        gae = 0.0
        for step in range(len(reward_pred) - 1, -1, -1):
            next_value = value_pred[step + 1] if step + 1 < len(value_pred) else 0.0
            delta = reward_pred[step] + gamma * next_value - value_pred[step]
            gae = delta + gamma * gae_lambda * gae
            traj_advantages[step] = gae

        traj_returns = traj_advantages + value_pred

        observations.append(obs_tensor)
        actions.append(act_tensor)
        old_log_probs.append(old_log_prob_tensor)
        returns.append(torch.as_tensor(traj_returns, dtype=torch.float32, device=device))
        advantages.append(torch.as_tensor(traj_advantages, dtype=torch.float32, device=device))
        learned_rewards.append(torch.as_tensor(reward_pred, dtype=torch.float32, device=device))

    if not observations:
        return None

    adv_tensor = torch.cat(advantages, dim=0)
    adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std(unbiased=False) + 1e-8)

    return PPOBatch(
        observations=torch.cat(observations, dim=0),
        actions=torch.cat(actions, dim=0),
        old_log_probs=torch.cat(old_log_probs, dim=0),
        returns=torch.cat(returns, dim=0),
        advantages=adv_tensor,
        learned_rewards=torch.cat(learned_rewards, dim=0),
    )


def ppo_update(
    policy: GaussianPolicy,
    value_net: ValueNetwork,
    policy_optimizer: torch.optim.Optimizer,
    value_optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    clip_ratio: float,
    ppo_epochs: int,
    minibatch_size: int,
    entropy_coef: float,
) -> dict[str, float]:
    policy.train()
    value_net.train()

    num_samples = batch.observations.shape[0]
    policy_losses: list[float] = []
    value_losses: list[float] = []
    entropies: list[float] = []

    for _ in range(ppo_epochs):
        permutation = torch.randperm(num_samples, device=batch.observations.device)
        for start in range(0, num_samples, minibatch_size):
            idx = permutation[start : start + minibatch_size]
            obs_mb = batch.observations[idx]
            act_mb = batch.actions[idx]
            old_log_prob_mb = batch.old_log_probs[idx]
            adv_mb = batch.advantages[idx]
            ret_mb = batch.returns[idx]

            new_log_prob = policy.log_prob(obs_mb, act_mb)
            ratio = torch.exp(new_log_prob - old_log_prob_mb)
            unclipped = ratio * adv_mb
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_mb
            policy_loss = -torch.min(unclipped, clipped).mean()

            entropy = policy.entropy(obs_mb).mean()
            policy_optimizer.zero_grad()
            (policy_loss - entropy_coef * entropy).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            policy_optimizer.step()

            value_pred = value_net(obs_mb)
            value_loss = F.mse_loss(value_pred, ret_mb)
            value_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            value_optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "learned_reward_mean": float(batch.learned_rewards.mean().item()),
    }


def select_top_trajectories(
    trajectories: Sequence[Trajectory],
    top_fraction: float,
) -> list[Trajectory]:
    if not trajectories:
        return []
    top_count = max(1, int(round(len(trajectories) * top_fraction)))
    sorted_trajectories = sorted(trajectories, key=lambda traj: traj.true_return, reverse=True)
    return sorted_trajectories[:top_count]


def behavior_clone(
    policy: GaussianPolicy,
    optimizer: torch.optim.Optimizer,
    trajectories: Sequence[Trajectory],
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> float:
    if not trajectories:
        return math.nan

    obs_data = np.concatenate([traj.observations for traj in trajectories], axis=0)
    act_data = np.concatenate([traj.actions for traj in trajectories], axis=0)

    obs_tensor = torch.as_tensor(obs_data, dtype=torch.float32, device=device)
    act_tensor = torch.as_tensor(act_data, dtype=torch.float32, device=device)

    policy.train()
    losses: list[float] = []
    for _ in range(epochs):
        permutation = torch.randperm(obs_tensor.shape[0], device=device)
        for start in range(0, obs_tensor.shape[0], batch_size):
            idx = permutation[start : start + batch_size]
            loss = -policy.log_prob(obs_tensor[idx], act_tensor[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else math.nan


def dpo_update(
    policy: GaussianPolicy,
    reference_policy: GaussianPolicy,
    optimizer: torch.optim.Optimizer,
    trajectory_tensors: Sequence[tuple[torch.Tensor, torch.Tensor]],
    preference_pairs: Sequence[tuple[int, int]],
    beta: float,
    epochs: int,
    batch_size: int,
    bc_regularizer: float,
) -> dict[str, float]:
    if not preference_pairs:
        return {"dpo_loss": math.nan, "bc_loss": math.nan}

    policy.train()
    reference_policy.eval()
    dpo_losses: list[float] = []
    bc_losses: list[float] = []
    pair_indices = list(preference_pairs)

    for _ in range(epochs):
        random.shuffle(pair_indices)
        for start in range(0, len(pair_indices), batch_size):
            batch_pairs = pair_indices[start : start + batch_size]
            pair_dpo_losses: list[torch.Tensor] = []
            pair_bc_losses: list[torch.Tensor] = []

            for winner_idx, loser_idx in batch_pairs:
                winner_obs, winner_act = trajectory_tensors[winner_idx]
                loser_obs, loser_act = trajectory_tensors[loser_idx]

                logp_winner = policy.log_prob(winner_obs, winner_act).mean()
                logp_loser = policy.log_prob(loser_obs, loser_act).mean()
                with torch.no_grad():
                    ref_logp_winner = reference_policy.log_prob(winner_obs, winner_act).mean()
                    ref_logp_loser = reference_policy.log_prob(loser_obs, loser_act).mean()

                margin = beta * ((logp_winner - ref_logp_winner) - (logp_loser - ref_logp_loser))
                pair_dpo_losses.append(-F.logsigmoid(margin))
                if bc_regularizer > 0:
                    pair_bc_losses.append(-logp_winner)

            if not pair_dpo_losses:
                continue

            dpo_loss = torch.stack(pair_dpo_losses).mean()
            if pair_bc_losses:
                bc_loss = torch.stack(pair_bc_losses).mean()
            else:
                bc_loss = torch.zeros((), dtype=dpo_loss.dtype, device=dpo_loss.device)

            loss = dpo_loss + bc_regularizer * bc_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            dpo_losses.append(float(dpo_loss.item()))
            bc_losses.append(float(bc_loss.item()))

    return {
        "dpo_loss": float(np.mean(dpo_losses)) if dpo_losses else math.nan,
        "bc_loss": float(np.mean(bc_losses)) if bc_losses else math.nan,
    }


def evaluate_policy(
    env: BenchmarkEnv,
    policy: GaussianPolicy,
    episodes: int,
    max_steps: int,
) -> tuple[float, float]:
    trajectories = collect_trajectories(env, policy, num_episodes=episodes, max_steps=max_steps, deterministic=True)
    mean_return = float(np.mean([traj.true_return for traj in trajectories]))
    success_rate = float(np.mean([traj.success for traj in trajectories]))
    return mean_return, success_rate


def _get_arg(args: object, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def _resolve_best_checkpoint_config(
    args: object,
    output_path: str,
    method: str,
) -> tuple[bool, str, Path | None]:
    save_best = bool(_get_arg(args, "save_best_checkpoint", False))
    best_metric = str(_get_arg(args, "best_metric", "eval_success")).strip().lower()
    if best_metric not in {"eval_success", "eval_return"}:
        raise ValueError(
            f"Unsupported --best-metric `{best_metric}`. Use `eval_success` or `eval_return`."
        )

    if not save_best:
        return False, best_metric, None

    best_path_arg = str(_get_arg(args, "best_checkpoint_path", "")).strip()
    if best_path_arg:
        best_path = Path(best_path_arg)
    elif output_path:
        output = Path(output_path)
        best_path = output.with_name(f"{output.stem}_best{output.suffix}")
    else:
        best_path = Path("checkpoints") / f"{method}_best.pt"

    best_path.parent.mkdir(parents=True, exist_ok=True)
    return True, best_metric, best_path


def run_rlhf(args: object) -> dict[str, Any]:
    seed = int(args.seed)
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)
    configure_robosuite_logging(_get_arg(args, "robosuite_log_level", "WARNING"))

    benchmark_name = str(_get_arg(args, "benchmark", "robosuite")).lower()
    env = make_env(
        benchmark=benchmark_name,
        env_name=args.env_name,
        robot=args.robot,
        reward_shaping=args.reward_shaping,
        control_freq=args.control_freq,
        horizon=args.horizon,
        hard_reset=args.hard_reset,
        seed=seed,
        libero_suite=str(_get_arg(args, "libero_suite", "libero_object")),
        libero_task_id=int(_get_arg(args, "libero_task_id", 0)),
    )

    try:
        initial_obs = flatten_obs(env.reset())
        obs_dim = int(initial_obs.shape[0])
        action_low, action_high = env.action_spec
        action_dim = int(action_low.shape[0])
        max_steps = min(args.max_steps, env.horizon)

        policy = GaussianPolicy(
            obs_dim=obs_dim,
            act_dim=action_dim,
            hidden_sizes=args.policy_hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            init_log_std=args.init_log_std,
        ).to(device)
        value_net = ValueNetwork(obs_dim=obs_dim, hidden_sizes=args.value_hidden_sizes).to(device)
        reward_model = RewardModel(
            obs_dim=obs_dim,
            act_dim=action_dim,
            hidden_sizes=args.reward_hidden_sizes,
        ).to(device)

        policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
        value_optimizer = torch.optim.Adam(value_net.parameters(), lr=args.value_lr)
        reward_optimizer = torch.optim.Adam(reward_model.parameters(), lr=args.reward_lr)

        robot_name = args.robot if benchmark_name == "robosuite" else "N/A"
        print(
            f"RLHF setup: benchmark={benchmark_name}, env={env.task_name}, robot={robot_name}, "
            f"obs_dim={obs_dim}, act_dim={action_dim}, device={device}, max_steps={max_steps}"
        )
        save_best, best_metric, best_checkpoint_path = _resolve_best_checkpoint_config(
            args=args,
            output_path=str(_get_arg(args, "output", "")),
            method="rlhf",
        )

        final_eval_return = math.nan
        final_eval_success = math.nan
        final_train_return = math.nan
        final_train_success = math.nan
        best_eval_return = -math.inf
        best_eval_success = -math.inf
        best_metric_value = -math.inf
        best_iteration = -1
        best_checkpoint: str | None = None
        history: list[dict[str, float]] = []

        for iteration in range(1, args.iterations + 1):
            trajectories = collect_trajectories(
                env,
                policy,
                num_episodes=args.episodes_per_iter,
                max_steps=max_steps,
                deterministic=False,
            )
            preference_pairs = sample_preference_pairs(
                trajectories=trajectories,
                num_pairs=args.pairs_per_iter,
                noise_prob=args.preference_noise,
                rng=rng,
            )
            trajectory_tensors = trajectories_to_tensors(trajectories, device=device)

            reward_loss = train_reward_model(
                reward_model=reward_model,
                optimizer=reward_optimizer,
                trajectory_tensors=trajectory_tensors,
                preference_pairs=preference_pairs,
                epochs=args.reward_epochs,
                batch_size=args.reward_batch_size,
            )

            ppo_batch = build_ppo_batch(
                trajectories=trajectories,
                reward_model=reward_model,
                value_net=value_net,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                device=device,
            )
            if ppo_batch is not None:
                ppo_stats = ppo_update(
                    policy=policy,
                    value_net=value_net,
                    policy_optimizer=policy_optimizer,
                    value_optimizer=value_optimizer,
                    batch=ppo_batch,
                    clip_ratio=args.ppo_clip_ratio,
                    ppo_epochs=args.ppo_epochs,
                    minibatch_size=args.ppo_batch_size,
                    entropy_coef=args.entropy_coef,
                )
            else:
                ppo_stats = {
                    "policy_loss": math.nan,
                    "value_loss": math.nan,
                    "entropy": math.nan,
                    "learned_reward_mean": math.nan,
                }

            eval_return, eval_success = evaluate_policy(env, policy, episodes=args.eval_episodes, max_steps=max_steps)
            train_return = float(np.mean([traj.true_return for traj in trajectories]))
            train_success = float(np.mean([traj.success for traj in trajectories]))

            final_eval_return = eval_return
            final_eval_success = eval_success
            final_train_return = train_return
            final_train_success = train_success
            best_eval_return = max(best_eval_return, eval_return)
            best_eval_success = max(best_eval_success, eval_success)
            history.append(
                {
                    "iteration": float(iteration),
                    "train_return": train_return,
                    "train_success": train_success,
                    "reward_loss": float(reward_loss),
                    "policy_loss": float(ppo_stats["policy_loss"]),
                    "value_loss": float(ppo_stats["value_loss"]),
                    "entropy": float(ppo_stats["entropy"]),
                    "learned_reward_mean": float(ppo_stats["learned_reward_mean"]),
                    "eval_return": eval_return,
                    "eval_success": eval_success,
                }
            )

            metric_value = eval_success if best_metric == "eval_success" else eval_return
            if save_best and best_checkpoint_path is not None and math.isfinite(metric_value):
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_iteration = iteration
                    best_checkpoint = str(best_checkpoint_path)
                    torch.save(
                        {
                            "policy_state_dict": policy.state_dict(),
                            "value_state_dict": value_net.state_dict(),
                            "reward_model_state_dict": reward_model.state_dict(),
                            "config": vars(args),
                            "benchmark": benchmark_name,
                            "env_name": env.task_name,
                            "seed": seed,
                            "best_metric": best_metric,
                            "best_metric_value": float(best_metric_value),
                            "best_iteration": int(best_iteration),
                            "history": history,
                        },
                        best_checkpoint_path,
                    )

            print(
                f"[RLHF {benchmark_name}/{env.task_name} iter {iteration:03d}] "
                f"train_return={train_return:.3f} train_success={train_success:.3f} "
                f"reward_loss={reward_loss:.4f} policy_loss={ppo_stats['policy_loss']:.4f} "
                f"value_loss={ppo_stats['value_loss']:.4f} eval_return={eval_return:.3f} eval_success={eval_success:.3f}"
            )

        summary: dict[str, Any] = {
            "method": "rlhf",
            "benchmark": benchmark_name,
            "env_name": env.task_name,
            "seed": seed,
            "final_train_return": final_train_return,
            "final_train_success": final_train_success,
            "final_eval_return": final_eval_return,
            "final_eval_success": final_eval_success,
            "best_eval_return": best_eval_return if best_eval_return != -math.inf else math.nan,
            "best_eval_success": best_eval_success if best_eval_success != -math.inf else math.nan,
            "best_metric": best_metric,
            "best_metric_value": best_metric_value if best_metric_value != -math.inf else math.nan,
            "best_iteration": int(best_iteration) if best_iteration > 0 else None,
            "best_checkpoint": best_checkpoint,
            "iterations": int(args.iterations),
            "checkpoint": None,
            "history": history,
        }

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "value_state_dict": value_net.state_dict(),
                    "reward_model_state_dict": reward_model.state_dict(),
                    "config": vars(args),
                    "summary": summary,
                },
                output_path,
            )
            summary["checkpoint"] = str(output_path)
            print(f"Saved RLHF checkpoint to {output_path}")
        if summary["best_checkpoint"]:
            print(
                f"Saved RLHF best checkpoint to {summary['best_checkpoint']} "
                f"(metric={best_metric} value={float(summary['best_metric_value']):.4f} "
                f"iter={summary['best_iteration']})"
            )
        return summary
    finally:
        env.close()


def run_dpo(args: object) -> dict[str, Any]:
    seed = int(args.seed)
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    rng = np.random.default_rng(seed)
    configure_robosuite_logging(_get_arg(args, "robosuite_log_level", "WARNING"))

    benchmark_name = str(_get_arg(args, "benchmark", "robosuite")).lower()
    env = make_env(
        benchmark=benchmark_name,
        env_name=args.env_name,
        robot=args.robot,
        reward_shaping=args.reward_shaping,
        control_freq=args.control_freq,
        horizon=args.horizon,
        hard_reset=args.hard_reset,
        seed=seed,
        libero_suite=str(_get_arg(args, "libero_suite", "libero_object")),
        libero_task_id=int(_get_arg(args, "libero_task_id", 0)),
    )

    try:
        initial_obs = flatten_obs(env.reset())
        obs_dim = int(initial_obs.shape[0])
        action_low, action_high = env.action_spec
        action_dim = int(action_low.shape[0])
        max_steps = min(args.max_steps, env.horizon)

        reference_policy = GaussianPolicy(
            obs_dim=obs_dim,
            act_dim=action_dim,
            hidden_sizes=args.policy_hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            init_log_std=args.init_log_std,
        ).to(device)
        policy = GaussianPolicy(
            obs_dim=obs_dim,
            act_dim=action_dim,
            hidden_sizes=args.policy_hidden_sizes,
            action_low=action_low,
            action_high=action_high,
            init_log_std=args.init_log_std,
        ).to(device)

        bc_optimizer = torch.optim.Adam(reference_policy.parameters(), lr=args.bc_lr)
        dpo_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)

        robot_name = args.robot if benchmark_name == "robosuite" else "N/A"
        print(
            f"DPO setup: benchmark={benchmark_name}, env={env.task_name}, robot={robot_name}, "
            f"obs_dim={obs_dim}, act_dim={action_dim}, device={device}, max_steps={max_steps}"
        )
        save_best, best_metric, best_checkpoint_path = _resolve_best_checkpoint_config(
            args=args,
            output_path=str(_get_arg(args, "output", "")),
            method="dpo",
        )

        preference_buffer: list[Trajectory] = []

        bc_pool = collect_trajectories(
            env,
            reference_policy,
            num_episodes=args.bc_pool_episodes,
            max_steps=max_steps,
            deterministic=False,
        )
        bc_demos = select_top_trajectories(bc_pool, top_fraction=args.bc_top_fraction)
        bc_loss = behavior_clone(
            policy=reference_policy,
            optimizer=bc_optimizer,
            trajectories=bc_demos,
            epochs=args.bc_epochs,
            batch_size=args.bc_batch_size,
            device=device,
        )
        policy.load_state_dict(reference_policy.state_dict())
        print(
            f"Reference policy BC pretrain: demos={len(bc_demos)}/{len(bc_pool)}, bc_loss={bc_loss:.4f}, "
            f"mean_pool_return={np.mean([traj.true_return for traj in bc_pool]):.3f}"
        )

        final_eval_return = math.nan
        final_eval_success = math.nan
        final_train_return = math.nan
        final_train_success = math.nan
        best_eval_return = -math.inf
        best_eval_success = -math.inf
        best_metric_value = -math.inf
        best_iteration = -1
        best_checkpoint: str | None = None
        history: list[dict[str, float]] = []

        for iteration in range(1, args.iterations + 1):
            trajectories = collect_trajectories(
                env,
                policy,
                num_episodes=args.episodes_per_iter,
                max_steps=max_steps,
                deterministic=False,
            )
            preference_buffer.extend(trajectories)
            if len(preference_buffer) > args.pref_buffer_size:
                preference_buffer = preference_buffer[-args.pref_buffer_size :]

            preference_pairs = sample_preference_pairs(
                trajectories=preference_buffer,
                num_pairs=args.pairs_per_iter,
                noise_prob=args.preference_noise,
                rng=rng,
            )
            trajectory_tensors = trajectories_to_tensors(preference_buffer, device=device)

            dpo_stats = dpo_update(
                policy=policy,
                reference_policy=reference_policy,
                optimizer=dpo_optimizer,
                trajectory_tensors=trajectory_tensors,
                preference_pairs=preference_pairs,
                beta=args.beta,
                epochs=args.dpo_epochs,
                batch_size=args.dpo_batch_size,
                bc_regularizer=args.bc_regularizer,
            )

            eval_return, eval_success = evaluate_policy(env, policy, episodes=args.eval_episodes, max_steps=max_steps)
            train_return = float(np.mean([traj.true_return for traj in trajectories]))
            train_success = float(np.mean([traj.success for traj in trajectories]))

            final_eval_return = eval_return
            final_eval_success = eval_success
            final_train_return = train_return
            final_train_success = train_success
            best_eval_return = max(best_eval_return, eval_return)
            best_eval_success = max(best_eval_success, eval_success)
            history.append(
                {
                    "iteration": float(iteration),
                    "train_return": train_return,
                    "train_success": train_success,
                    "dpo_loss": float(dpo_stats["dpo_loss"]),
                    "bc_loss": float(dpo_stats["bc_loss"]),
                    "eval_return": eval_return,
                    "eval_success": eval_success,
                    "pref_buffer": float(len(preference_buffer)),
                }
            )

            metric_value = eval_success if best_metric == "eval_success" else eval_return
            if save_best and best_checkpoint_path is not None and math.isfinite(metric_value):
                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_iteration = iteration
                    best_checkpoint = str(best_checkpoint_path)
                    torch.save(
                        {
                            "policy_state_dict": policy.state_dict(),
                            "reference_policy_state_dict": reference_policy.state_dict(),
                            "config": vars(args),
                            "benchmark": benchmark_name,
                            "env_name": env.task_name,
                            "seed": seed,
                            "best_metric": best_metric,
                            "best_metric_value": float(best_metric_value),
                            "best_iteration": int(best_iteration),
                            "history": history,
                        },
                        best_checkpoint_path,
                    )

            print(
                f"[DPO {benchmark_name}/{env.task_name} iter {iteration:03d}] "
                f"train_return={train_return:.3f} train_success={train_success:.3f} "
                f"dpo_loss={dpo_stats['dpo_loss']:.4f} bc_loss={dpo_stats['bc_loss']:.4f} "
                f"eval_return={eval_return:.3f} eval_success={eval_success:.3f} "
                f"pref_buffer={len(preference_buffer)}"
            )

        summary: dict[str, Any] = {
            "method": "dpo",
            "benchmark": benchmark_name,
            "env_name": env.task_name,
            "seed": seed,
            "final_train_return": final_train_return,
            "final_train_success": final_train_success,
            "final_eval_return": final_eval_return,
            "final_eval_success": final_eval_success,
            "best_eval_return": best_eval_return if best_eval_return != -math.inf else math.nan,
            "best_eval_success": best_eval_success if best_eval_success != -math.inf else math.nan,
            "best_metric": best_metric,
            "best_metric_value": best_metric_value if best_metric_value != -math.inf else math.nan,
            "best_iteration": int(best_iteration) if best_iteration > 0 else None,
            "best_checkpoint": best_checkpoint,
            "iterations": int(args.iterations),
            "checkpoint": None,
            "history": history,
        }

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "policy_state_dict": policy.state_dict(),
                    "reference_policy_state_dict": reference_policy.state_dict(),
                    "config": vars(args),
                    "summary": summary,
                },
                output_path,
            )
            summary["checkpoint"] = str(output_path)
            print(f"Saved DPO checkpoint to {output_path}")
        if summary["best_checkpoint"]:
            print(
                f"Saved DPO best checkpoint to {summary['best_checkpoint']} "
                f"(metric={best_metric} value={float(summary['best_metric_value']):.4f} "
                f"iter={summary['best_iteration']})"
            )
        return summary
    finally:
        env.close()
