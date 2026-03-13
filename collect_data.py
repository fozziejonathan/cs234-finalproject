"""
collect_data.py
---------------
Phase 1: Trajectory Collection

Supports two collection modes:
  1. Scripted expert (default) — deterministic, no noise.
  2. BC policy (--bc_ckpt) — stochastic rollouts from a trained GaussianPolicy
     checkpoint. Use this to generate on-policy preference data for DPO
     fine-tuning. Stochastic rollouts are essential so that the BC policy
     produces both successes and failures, giving DPO a clean preference signal.

For BC rollouts, use --n_rollouts_per_task to run each task configuration
multiple times. Each run produces a different trajectory due to stochasticity,
giving you natural variance (successes and failures) on the same goal/object
placement. This is the cleanest possible preference signal for DPO since the
only thing varying between paired trajectories is the policy's behavior, not
the environment setup.

Output filenames:
  Scripted : data/raw_trajectories_{env}_{n}seeds.h5
  BC policy: data/raw_trajectories_{env}_{n}seeds_{r}rpt_bc.h5
             where rpt = rollouts per task
"""

import argparse
import os
import warnings
import numpy as np
import h5py
import torch
import metaworld
import metaworld.policies as mw_policies
from config import ENV_NAME, MAX_EPISODE_STEPS, RAW_DATA_PATH

SUCCESS_CONSEC_STEPS = 5

DEFAULT_SEED_START = 9500
DEFAULT_SEED_MAX   = 10000  # exclusive upper bound
DEFAULT_SEEDS = list(range(DEFAULT_SEED_START, DEFAULT_SEED_MAX))

# ── Scripted policy lookup ─────────────────────────────────────────────────────
_POLICY_FRAGMENTS = {
    "reach-v3": "Reach",
    "push-v3": "Push",
    "pick-place-v3": "PickPlace",
    "door-open-v3": "DoorOpen",
    "drawer-open-v3": "DrawerOpen",
    "drawer-close-v3": "DrawerClose",
    "button-press-topdown-v3": "ButtonPressTopdown",
    "peg-insert-side-v3": "PegInsertionSide",
    "window-open-v3": "WindowOpen",
    "window-close-v3": "WindowClose",
    "assembly-v3": "Assembly",
    "door-unlock-v3": "DoorUnlock",
    "bin-picking-v3": "BinPicking"
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


# ── BC policy loader ───────────────────────────────────────────────────────────

def _load_bc_policy(ckpt_path, obs_norm_path):
    from train_bc import GaussianPolicy, ObsNormalizer, DEVICE

    obs_norm = ObsNormalizer()
    obs_norm.load(obs_norm_path)

    policy = GaussianPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()

    print(f"[INFO] Loaded BC policy from       {ckpt_path}")
    print(f"[INFO] Loaded obs normalizer from  {obs_norm_path}")
    return policy, obs_norm


class BCPolicyWrapper:
    """
    Thin wrapper so BC policy has the same .get_action(obs) interface
    as the MetaWorld scripted policies.

    Always samples stochastically (deterministic=False) so that repeated
    rollouts on the same task produce different trajectories — some successes,
    some failures — which is exactly what DPO needs.
    """

    def __init__(self, policy, obs_norm):
        from train_bc import DEVICE
        self.policy = policy
        self.obs_norm = obs_norm
        self.device = DEVICE

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        obs_n = self.obs_norm.normalize(obs.astype(np.float32))
        obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(self.device)
        action = self.policy.get_action(obs_t, deterministic=False)
        return action.astype(np.float32)


# ── Main collection loop ───────────────────────────────────────────────────────

def collect(env_name=ENV_NAME, seeds=None, out=RAW_DATA_PATH,
            bc_ckpt=None, obs_norm_path=None, n_rollouts_per_task=1):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    # ── Output path ───────────────────────────────────────────────────────────
    out = out.format(env=env_name, n=f"{len(seeds)}seeds")
    if bc_ckpt is not None:
        # e.g. data/raw_trajectories_peg-insert-side-v3_20seeds_3rpt_bc.h5
        out = out.replace(".h5", f"_{n_rollouts_per_task}rpt_bc.h5")

    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)

    # ── Load policy ───────────────────────────────────────────────────────────
    if bc_ckpt is not None:
        if obs_norm_path is None:
            obs_norm_path = f"checkpoints/obs_normalizer_{env_name}.npz"
            print(f"[INFO] obs_norm not specified, defaulting to {obs_norm_path}")
        bc_policy, obs_norm = _load_bc_policy(bc_ckpt, obs_norm_path)
        policy = BCPolicyWrapper(bc_policy, obs_norm)
        mode = f"BC policy (stochastic, {n_rollouts_per_task} rollouts/task)"
    else:
        policy = _get_scripted_policy(env_name)
        mode = "Scripted expert (deterministic)"
        if n_rollouts_per_task > 1:
            print(f"[WARN] --n_rollouts_per_task={n_rollouts_per_task} has no effect "
                  f"with the scripted expert (deterministic policy).")
        n_rollouts_per_task = 1  # repeating a deterministic policy is wasteful

    n_total = len(seeds) * 50 * n_rollouts_per_task
    print(f"[INFO] Mode: {mode}")
    print(f"[INFO] Collecting {len(seeds)} seeds × 50 tasks "
          f"× {n_rollouts_per_task} rollouts = {n_total} trajectories")
    print(f"[INFO] env={env_name}")
    print(f"[INFO] Success requires {SUCCESS_CONSEC_STEPS} consecutive success flags")
    print(f"[INFO] Saving → {out}")

    returns, successes = [], []
    traj_idx = 0

    with h5py.File(out, "w") as f:
        for seed in seeds:
            mt1 = metaworld.MT1(env_name, seed=seed)
            env = mt1.train_classes[env_name]()
            tasks = mt1.train_tasks
            seed_returns, seed_successes = [], []

            for task_idx in range(len(tasks)):
                env.set_task(tasks[task_idx])

                for rollout in range(n_rollouts_per_task):
                    obs, _ = env.reset()

                    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
                    ep_success = False
                    success_step = -1
                    consec_count = 0

                    for step in range(MAX_EPISODE_STEPS):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            action = policy.get_action(obs)
                            action = np.clip(action, env.action_space.low,
                                            env.action_space.high)

                        next_obs, reward, terminated, truncated, info = env.step(action)
                        done = terminated or truncated

                        obs_buf.append(obs.astype(np.float32))
                        act_buf.append(action)
                        rew_buf.append(float(reward))
                        done_buf.append(bool(done))

                        if info.get("success", False):
                            consec_count += 1
                        else:
                            consec_count = 0

                        if not ep_success and consec_count >= SUCCESS_CONSEC_STEPS:
                            ep_success = True
                            success_step = step - (SUCCESS_CONSEC_STEPS - 1)

                        obs = next_obs
                        if done:
                            break

                    ep_return = float(np.sum(rew_buf))
                    returns.append(ep_return)
                    successes.append(float(ep_success))
                    seed_returns.append(ep_return)
                    seed_successes.append(float(ep_success))

                    grp = f.create_group(f"trajectory_{traj_idx}")
                    grp.create_dataset("observations", data=np.array(obs_buf))
                    grp.create_dataset("actions", data=np.array(act_buf))
                    grp.create_dataset("rewards", data=np.array(rew_buf, dtype=np.float32))
                    grp.create_dataset("dones", data=np.array(done_buf))
                    grp.attrs["success"] = ep_success
                    grp.attrs["ep_return"] = ep_return
                    grp.attrs["success_step"] = success_step
                    grp.attrs["task_idx"] = task_idx
                    grp.attrs["mt1_seed"] = seed
                    grp.attrs["rollout_idx"] = rollout  # which repeat this was

                    traj_idx += 1

            print(f"  [seed {seed:>5}]  "
                  f"mean_return: {np.mean(seed_returns):.2f}  "
                  f"success_rate: {np.mean(seed_successes):.2%}  "
                  f"({int(sum(seed_successes))}/{len(seed_returns)})")

    overall_success = np.mean(successes)
    n_success = int(sum(successes))
    print(f"\n[INFO] Done.  Total trajectories: {traj_idx}")
    print(f"[INFO] Return — mean: {np.mean(returns):.2f}  "
          f"std: {np.std(returns):.2f}  "
          f"range: [{np.min(returns):.2f}, {np.max(returns):.2f}]")
    print(f"[INFO] Success rate: {overall_success:.2%}  ({n_success}/{traj_idx})")
    print(f"[INFO] Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=ENV_NAME)
    parser.add_argument("--n_seeds", type=int, default=len(DEFAULT_SEEDS),
                        help="Number of MT1 seeds (each gives 50 unique tasks)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Explicit seed list (overrides --n_seeds)")
    parser.add_argument("--bc_ckpt", type=str, default=None,
                        help="Path to BC policy checkpoint (.pt). "
                             "If omitted, uses the scripted expert.")
    parser.add_argument("--obs_norm", type=str, default=None,
                        help="Path to obs normalizer (.npz). "
                             "Defaults to checkpoints/obs_normalizer_{env}.npz")
    parser.add_argument("--n_rollouts_per_task", type=int, default=3,
                        help="Number of stochastic rollouts per task config "
                             "(BC mode only). Default 3 → 3000 trajectories "
                             "with 20 seeds.")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds is not None else DEFAULT_SEEDS[:args.n_seeds]

    collect(
        env_name=args.env,
        seeds=seeds,
        bc_ckpt=args.bc_ckpt,
        obs_norm_path=args.obs_norm,
        n_rollouts_per_task=args.n_rollouts_per_task,
    )