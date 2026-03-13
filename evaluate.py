"""
evaluate.py — evaluate any saved policy checkpoint in MetaWorld
Usage:
    python evaluate.py --ckpt checkpoints/bc_policy_bin-picking-v3.pt
    python evaluate.py --ckpt checkpoints/dpo_policy_bin-picking-v3.pt --n 50 --n_seeds 12
"""
import argparse
import numpy as np
import torch
import metaworld
from train_bc import GaussianPolicy, ObsNormalizer, DEVICE
from config import ENV_NAME, MAX_EPISODE_STEPS

DEFAULT_OBS_NORM = f"checkpoints/obs_normalizer_{ENV_NAME}.npz"


def evaluate_single_seed(policy, obs_norm, env, tasks, n_episodes, seed):
    np.random.seed(seed)
    returns, successes = [], []

    with torch.no_grad():
        for i in range(n_episodes):
            env.set_task(tasks[i % len(tasks)])
            obs, _ = env.reset()
            ep_ret, success = 0.0, False

            for _ in range(MAX_EPISODE_STEPS):
                obs_n = obs_norm.normalize(obs.astype(np.float32))
                obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(DEVICE)
                action = policy.get_action(obs_t, deterministic=True)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, r, terminated, truncated, info = env.step(action)
                ep_ret += r
                if info.get("success", 0.0):
                    success = True
                if terminated or truncated:
                    break

            returns.append(ep_ret)
            successes.append(float(success))
            if (i + 1) % 20 == 0:
                print(f"  [{i + 1:>3}/{n_episodes}]  "
                      f"success so far: {np.mean(successes):.2%}")

    return returns, successes


def evaluate(ckpt_path, obs_norm_path=DEFAULT_OBS_NORM,
             n_episodes=50, n_seeds=24, base_seed=1):
    obs_norm = ObsNormalizer()
    obs_norm.load(obs_norm_path)

    policy = GaussianPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()

    print(f"\n{'=' * 60}")
    print(f"  Evaluating : {ckpt_path}")
    print(f"  Episodes   : {n_episodes} × {n_seeds} seeds = {n_episodes * n_seeds} total")
    print(f"{'=' * 60}")

    seed_success_rates, seed_mean_returns = [], []

    for s in range(n_seeds):
        seed = base_seed + s
        print(f"\n  Seed {seed}:")

        mt1 = metaworld.MT1(ENV_NAME, seed=seed)
        env = mt1.train_classes[ENV_NAME]()
        tasks = mt1.train_tasks

        returns, successes = evaluate_single_seed(
            policy, obs_norm, env, tasks, n_episodes, seed
        )
        sr = np.mean(successes)
        mr = np.mean(returns)
        seed_success_rates.append(sr)
        seed_mean_returns.append(mr)
        print(f"  → success={sr:.2%}  mean_return={mr:.2f} ± {np.std(returns):.2f}")

    mean_sr  = np.mean(seed_success_rates)
    std_sr   = np.std(seed_success_rates)
    mean_ret = np.mean(seed_mean_returns)
    std_ret  = np.std(seed_mean_returns)
    ci_sr    = 1.96 * np.sqrt(mean_sr * (1 - mean_sr) / (n_episodes * n_seeds))

    print(f"\n  {'─' * 40}")
    print(f"  Success rate : {mean_sr:.2%} ± {std_sr:.2%}  (95% CI ±{ci_sr:.2%})")
    print(f"  Mean return  : {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"  {'─' * 40}\n")

    return {
        "success_rate": mean_sr,
        "success_std":  std_sr,
        "success_ci":   ci_sr,
        "mean_return":  mean_ret,
        "return_std":   std_ret,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True,
                        help="Path to policy checkpoint .pt")
    parser.add_argument("--obs_norm", default=DEFAULT_OBS_NORM,
                        help="Path to obs normalizer .npz")
    parser.add_argument("--n",        type=int, default=50,
                        help="Episodes per seed")
    parser.add_argument("--n_seeds",  type=int, default=24,
                        help="Number of random seeds to average over")
    parser.add_argument("--seed",     type=int, default=1,
                        help="Base random seed")
    args = parser.parse_args()

    evaluate(args.ckpt, args.obs_norm,
             n_episodes=args.n,
             n_seeds=args.n_seeds,
             base_seed=args.seed)