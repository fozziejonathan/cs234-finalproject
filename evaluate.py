"""
evaluate.py — evaluate any saved policy checkpoint in MetaWorld reach-v3
Usage:
    python evaluate.py --ckpt checkpoints/dpo_policy.pt
    python evaluate.py --ckpt checkpoints/bc_policy.pt
"""
import argparse
import numpy as np
import torch
import metaworld
from train_dpo import GaussianPolicy, ObsNormalizer, DEVICE, OBS_NORM_PATH
from config import ENV_NAME, MAX_EPISODE_STEPS

def evaluate(ckpt_path, n_episodes=100):
    obs_norm = ObsNormalizer()
    obs_norm.load(OBS_NORM_PATH)

    policy = GaussianPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()

    ml1   = metaworld.ML1(ENV_NAME)
    env   = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks

    returns, successes = [], []
    with torch.no_grad():
        for i in range(n_episodes):
            env.set_task(tasks[i % len(tasks)])
            obs, _ = env.reset()
            ep_ret, success = 0.0, False
            for _ in range(MAX_EPISODE_STEPS):
                obs_n  = obs_norm.normalize(obs.astype(np.float32))
                obs_t  = torch.FloatTensor(obs_n).unsqueeze(0).to(DEVICE)
                action = policy.get_action(obs_t, deterministic=True)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, r, terminated, truncated, info = env.step(action)
                ep_ret += r
                if info.get("success", 0.0): success = True
                if terminated or truncated: break
            returns.append(ep_ret)
            successes.append(float(success))
            if (i + 1) % 20 == 0:
                print(f"  [{i+1:>3}/{n_episodes}]  "
                      f"success so far: {np.mean(successes):.2%}")

    print(f"\n=== {ckpt_path} ===")
    print(f"  Success rate: {np.mean(successes):.2%}")
    print(f"  Mean return:  {np.mean(returns):.2f} ± {np.std(returns):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--n",    type=int, default=100)
    args = parser.parse_args()
    evaluate(args.ckpt, args.n)