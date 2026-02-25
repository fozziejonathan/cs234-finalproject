"""
collect_data.py
---------------
Phase 1: Trajectory Collection

Runs the MetaWorld scripted expert policy with very low noise (σ=0.02) so
nearly every episode succeeds. Success flag is stored per-trajectory so
downstream BC can filter to genuinely successful demonstrations only.

Output
------
data/raw_trajectories.h5
    trajectory_0/
        observations  (T, 39)  float32
        actions       (T,  4)  float32
        rewards       (T,)     float32
        dones         (T,)     bool
        attrs:
            success   bool     — whether episode hit the success threshold
            ep_return float    — undiscounted sum of rewards
"""

import argparse
import os
import numpy as np
import h5py
import metaworld
import metaworld.policies as mw_policies
from config import ENV_NAME, MAX_EPISODE_STEPS, RAW_DATA_PATH

# ── Policy lookup ─────────────────────────────────────────────────────────────
_POLICY_FRAGMENTS = {
    "reach-v3":                  "Reach",
    "push-v3":                   "Push",
    "pick-place-v3":             "PickPlace",
    "door-open-v3":              "DoorOpen",
    "drawer-open-v3":            "DrawerOpen",
    "drawer-close-v3":           "DrawerClose",
    "button-press-topdown-v3":   "ButtonPressTopdown",
    "peg-insert-side-v3":        "PegInsertionSide",
    "window-open-v3":            "WindowOpen",
    "window-close-v3":           "WindowClose",
}

def _get_scripted_policy(env_name):
    all_names = [n for n in dir(mw_policies) if "Policy" in n]
    frag = _POLICY_FRAGMENTS.get(env_name)
    if frag is None:
        print(f"[WARN] No scripted policy for '{env_name}' — using random actions.")
        return None
    matches = [n for n in all_names if frag in n]
    if not matches:
        print(f"[WARN] No policy matching '{frag}' found — using random actions.")
        return None
    cls = getattr(mw_policies, matches[0])
    print(f"[INFO] Using scripted policy: {cls.__name__}")
    return cls()


def collect(env_name=ENV_NAME, n=5000, noise=0.02, out=RAW_DATA_PATH):
    ml1    = metaworld.ML1(env_name)
    env    = ml1.train_classes[env_name]()
    tasks  = ml1.train_tasks
    policy = _get_scripted_policy(env_name)

    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    print(f"[INFO] Collecting {n} trajectories  env={env_name}  noise={noise}")
    print(f"[INFO] Saving → {out}")

    returns, successes = [], []

    with h5py.File(out, "w") as f:
        for i in range(n):
            env.set_task(tasks[i % len(tasks)])
            obs, _ = env.reset()

            obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
            ep_success = False

            for _ in range(MAX_EPISODE_STEPS):
                if policy is not None:
                    base = policy.get_action(obs)
                else:
                    base = env.action_space.sample()
                action = np.clip(
                    base + np.random.normal(0.0, noise, size=base.shape),
                    env.action_space.low, env.action_space.high
                ).astype(np.float32)

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                obs_buf.append(obs.astype(np.float32))
                act_buf.append(action)
                rew_buf.append(float(reward))
                done_buf.append(bool(done))

                if info.get("success", False):
                    ep_success = True

                obs = next_obs
                if done:
                    break

            ep_return = float(np.sum(rew_buf))
            returns.append(ep_return)
            successes.append(float(ep_success))

            grp = f.create_group(f"trajectory_{i}")
            grp.create_dataset("observations", data=np.array(obs_buf))
            grp.create_dataset("actions",      data=np.array(act_buf))
            grp.create_dataset("rewards",      data=np.array(rew_buf, dtype=np.float32))
            grp.create_dataset("dones",        data=np.array(done_buf))
            grp.attrs["success"]   = ep_success
            grp.attrs["ep_return"] = ep_return

            if (i + 1) % 500 == 0:
                print(f"  [{i+1:>5}/{n}]  "
                      f"mean_return: {np.mean(returns[-500:]):.2f}  "
                      f"success_rate: {np.mean(successes[-500:]):.2%}")

    overall_success = np.mean(successes)
    n_success = int(sum(successes))
    print(f"\n[INFO] Done.")
    print(f"[INFO] Return — mean: {np.mean(returns):.2f}  "
          f"std: {np.std(returns):.2f}  "
          f"range: [{np.min(returns):.2f}, {np.max(returns):.2f}]")
    print(f"[INFO] Success rate: {overall_success:.2%}  ({n_success}/{n} episodes)")
    print(f"[INFO] Successful demos available for BC: {n_success}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",   default=ENV_NAME)
    parser.add_argument("--n",     type=int,   default=5000)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--out",   default=RAW_DATA_PATH)
    args = parser.parse_args()
    collect(args.env, args.n, args.noise, args.out)