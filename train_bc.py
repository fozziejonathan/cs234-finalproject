"""
train_bc.py
-----------
Stage 0: Behavioral Cloning (BC) on successful expert demonstrations.

Loss:   MSE( μ_θ(s), a_expert )
Data:   Successful trajectories loaded directly from the raw h5 file
        produced by collect_data.py. Training on the h5 directly avoids
        the duplication introduced by the preference pairing process, where
        the same trajectory can appear as 'chosen' in multiple pairs.

Architecture note — log_std as nn.Parameter (not a frozen buffer):
    MSE has no gradient signal for log_std, so it stays near its
    initialization (-1.0 → std ≈ 0.37) throughout BC. However, making it
    a proper trainable parameter means DPO and RLHF can both unfreeze it
    during fine-tuning without any architectural mismatch. This follows the
    PPO/TRPO convention of a state-independent learned log_std.
    See: OpenAI SpinningUp docs; Schulman et al. (PPO, 2017).

Outputs (saved as a paired unit — never load one without the other):
    checkpoints/bc_policy.pt        — policy weights
    checkpoints/obs_normalizer.npz  — per-dim mean/std fit on training obs
"""

import os
import h5py
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import metaworld

from config import (
    ENV_NAME, OBS_DIM, ACT_DIM, HIDDEN_DIM,
    MAX_EPISODE_STEPS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BC_SAVE_PATH = f"checkpoints/bc_policy_{ENV_NAME}.pt"
OBS_NORM_PATH = f"checkpoints/obs_normalizer_{ENV_NAME}.npz"

# ── Hyperparameters ───────────────────────────────────────────────────────────
BC_LR = 3e-4
BC_EPOCHS = 60
BC_BATCH = 256
BC_EVAL_EVERY = 1  # run env eval every N epochs
BC_EVAL_EPS = 100  # episodes per env eval

# Initial log_std value: std = exp(-1) ≈ 0.37.
# Not updated by MSE, but kept as a parameter so downstream fine-tuning
# (DPO, RLHF/PPO) can learn it without any checkpoint surgery.
LOG_STD_INIT = -1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# Observation Normalizer
# ══════════════════════════════════════════════════════════════════════════════

class ObsNormalizer:
    """
    Fits per-dimension mean and std from training observations,
    then normalizes to zero mean and unit variance.

    Must be saved alongside the policy checkpoint — the network weights
    are only valid for inputs normalized with these exact statistics.
    """

    def __init__(self):
        self.mean = self.std = None

    def fit(self, obs: np.ndarray):
        """obs: (N, obs_dim) array of raw observations."""
        self.mean = obs.mean(axis=0).astype(np.float32)
        self.std = np.clip(obs.std(axis=0), 1e-8, None).astype(np.float32)
        print(f"[INFO] ObsNormalizer fit on {len(obs):,} observations.")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)
        print(f"[INFO] ObsNormalizer saved → {path}")

    def load(self, path: str):
        d = np.load(path)
        self.mean = d["mean"].astype(np.float32)
        self.std = d["std"].astype(np.float32)
        print(f"[INFO] ObsNormalizer loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Policy — shared architecture for BC, DPO, and RLHF
# ══════════════════════════════════════════════════════════════════════════════

class GaussianPolicy(nn.Module):
    """
    π(a|s) = N( μ_θ(s), σ²I )

    μ_θ: two-hidden-layer MLP (39 → 256 → 256 → 4).
    σ:   global, state-independent, learned scalar per action dim.
         Initialized to LOG_STD_INIT; MSE training leaves it unchanged.
         DPO and RLHF fine-tuning are free to update it.

    Architecture choices:
        - LayerNorm after the first linear layer: stabilizes activations
          for MetaWorld's heterogeneous 39-dim obs space (XYZ, quaternions,
          zeroed object dims). Acts as a second line of defense after
          obs normalization. See Ba et al. (2016); Kostrikov et al. (2021).
        - Orthogonal initialization: standard for on-policy RL (PPO paper).
        - Small gain (0.01) on the output head: keeps initial actions near
          zero, matching the scripted policy's typical starting magnitude.
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)

        # Learned, state-independent log_std — not a buffer.
        # MSE leaves it at init; fine-tuning stages can update it freely.
        self.log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))

        # Orthogonal init — standard for policy gradient / BC networks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)

    def forward(self, obs):
        """Returns (μ, σ) for a batch of observations."""
        mu = self.mu_head(self.trunk(obs))
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_action(self, obs_t: torch.Tensor, deterministic: bool = True) -> np.ndarray:
        """Single-observation action for env interaction."""
        with torch.no_grad():
            mu, std = self.forward(obs_t)
            action = mu if deterministic else Normal(mu, std).sample()
        return action.squeeze(0).cpu().numpy()

    def traj_log_prob(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """
        Sum of per-step log-probs normalized by episode length.
        Used by DPO fine-tuning; defined here so the checkpoint is
        self-contained.

        log π(τ) = Σ_t log π(a_t | s_t) / T
        """
        mu, std = self.forward(obs_seq)
        step_lp = Normal(mu, std).log_prob(act_seq).sum(dim=-1)  # (T,)
        T = obs_seq.shape[0]
        return step_lp.sum() / float(T)


# ══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_bc_data_from_h5(h5_path: str, obs_norm: ObsNormalizer, val_frac: float = 0.1):
    """
    Load successful trajectories directly from raw h5 — no duplication.
    Splits into train/val by trajectory (not by transition) to avoid leakage.
    """
    all_obs, all_acts = [], []
    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        n_success = 0
        for key in keys:
            grp = f[key]
            if not grp.attrs.get("success", False):
                continue
            obs = grp["observations"][:]
            acts = np.clip(grp["actions"][:], -1.0, 1.0).astype(np.float32)
            obs_n = obs_norm.normalize(obs.astype(np.float32))
            all_obs.append(obs_n)
            all_acts.append(acts)
            n_success += 1

    print(f"[BC] {n_success}/{len(keys)} successful trajectories "
          f"({n_success / len(keys):.1%})")

    # Train/val split at trajectory level to avoid leakage
    split = int(len(all_obs) * (1 - val_frac))
    idx = np.random.permutation(len(all_obs))
    tr_idx, val_idx = idx[:split], idx[split:]

    def to_tensors(indices):
        obs_cat = np.concatenate([all_obs[i] for i in indices])
        acts_cat = np.concatenate([all_acts[i] for i in indices])
        return (torch.FloatTensor(obs_cat).to(DEVICE),
                torch.FloatTensor(acts_cat).to(DEVICE))

    obs_tr, acts_tr = to_tensors(tr_idx)
    obs_val, acts_val = to_tensors(val_idx)
    print(f"[BC] Train: {len(obs_tr):,} transitions | "
          f"Val: {len(obs_val):,} transitions (no duplicates)")
    return obs_tr, acts_tr, obs_val, acts_val


# ══════════════════════════════════════════════════════════════════════════════
# Environment evaluation
# ══════════════════════════════════════════════════════════════════════════════

EVAL_SEEDS     = [119, 120, 121]  # disjoint from collect_data (9500+) and evaluate.py (1-24)
EVAL_EPS_PER_SEED = 50


def env_eval(policy: GaussianPolicy, obs_norm: ObsNormalizer,
             n_episodes: int = None,
             deterministic: bool = True):
    """
    Roll out the policy across EVAL_SEEDS × EVAL_EPS_PER_SEED episodes and
    return (mean_return, success_rate).  Each seed produces a different set of
    50 task configurations, so 3 seeds × 50 episodes = 150 distinct tasks.

    n_episodes is ignored (kept for API compatibility) — episode count is
    always EVAL_SEEDS × EVAL_EPS_PER_SEED.
    """
    policy.eval()
    returns, successes = [], []

    for seed in EVAL_SEEDS:
        mt1 = metaworld.MT1(ENV_NAME, seed=seed)
        env = mt1.train_classes[ENV_NAME]()
        tasks = mt1.train_tasks

        with torch.no_grad():
            for i in range(EVAL_EPS_PER_SEED):
                env.set_task(tasks[i % len(tasks)])
                obs, _ = env.reset()
                ep_ret, success = 0.0, False
                consec_count = 0

                for _ in range(MAX_EPISODE_STEPS):
                    obs_n = obs_norm.normalize(obs.astype(np.float32))
                    obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(DEVICE)
                    action = policy.get_action(obs_t, deterministic=deterministic)
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    obs, reward, terminated, truncated, info = env.step(action)
                    ep_ret += reward

                    if info.get("success", False):
                        consec_count += 1
                    else:
                        consec_count = 0
                    if not success and consec_count >= 5:
                        success = True

                    if terminated or truncated:
                        break

                returns.append(ep_ret)
                successes.append(float(success))

        env.close()

    policy.train()
    return float(np.mean(returns)), float(np.mean(successes))


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_bc(policy: GaussianPolicy, h5_path: str, obs_norm: ObsNormalizer,
             epochs: int = BC_EPOCHS, lr: float = BC_LR, batch: int = BC_BATCH):
    """
    Minimize MSE( μ_θ(s), a_expert ) on successful demonstrations.

    Checkpointing: saves the weights that achieve the best deterministic
    success rate during training (not just the final weights).
    """
    obs_t, acts_t, obs_val, acts_val = load_bc_data_from_h5(h5_path, obs_norm)
    N = len(obs_t)

    # log_std has no gradient signal under MSE, but including it in the
    # optimizer keeps the interface consistent — downstream fine-tuning
    # doesn't need to reconstruct the optimizer from scratch.
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'=' * 60}")
    print(f"  Behavioral Cloning")
    print(f"  Transitions : {N:,}  |  Epochs: {epochs}  |  Batch: {batch}")
    print(f"  Loss        : MSE( μ_θ(s), a_expert )")
    print(f"  LR schedule : cosine annealing  (init lr={lr})")
    print(f"  Eval every  : {BC_EVAL_EVERY} epochs  |  {BC_EVAL_EPS} episodes")
    print(f"  Checkpoint  : best deterministic success rate")
    print(f"{'=' * 60}\n")

    best_success = -1.0
    best_return = -float('inf')
    os.makedirs(os.path.dirname(BC_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        policy.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch):
            idx = perm[start: start + batch]
            obs_b = obs_t[idx]
            acts_b = acts_t[idx]

            mu, _ = policy.forward(obs_b)
            loss = F.mse_loss(mu, acts_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation MSE — cheap forward pass, no grad
        policy.eval()
        with torch.no_grad():
            mu_val, _ = policy.forward(obs_val)
            val_loss = F.mse_loss(mu_val, acts_val).item()
        policy.train()

        if (epoch + 1) % BC_EVAL_EVERY == 0 or epoch == 0:
            ret_det, succ_det = env_eval(policy, obs_norm, BC_EVAL_EPS, deterministic=True)

            improved = (succ_det > best_success) or \
                       (succ_det == best_success and ret_det > best_return)
            if improved:
                best_success = succ_det
                best_return = ret_det
                torch.save(policy.state_dict(), BC_SAVE_PATH)

            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| train MSE: {epoch_loss / n_batches:.5f}  "
                  f"| val MSE: {val_loss:.5f}  "
                  f"| success det: {succ_det:.2%}  "
                  f"| return det: {ret_det:.1f}"
                  + ("  ✓" if improved else ""))

        else:
            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| train MSE: {epoch_loss / n_batches:.5f}  "
                  f"| val MSE: {val_loss:.5f}")

    # Fallback: if env eval was never triggered, save final weights
    if best_success < 0:
        torch.save(policy.state_dict(), BC_SAVE_PATH)

    print(f"\n  Best BC checkpoint: {best_success:.2%} success → {BC_SAVE_PATH}")

    # Reload best checkpoint so the returned policy matches what was saved
    policy.load_state_dict(torch.load(BC_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True,
                        help="Path to raw trajectory h5 from collect_data.py")
    parser.add_argument("--epochs", type=int, default=BC_EPOCHS)
    parser.add_argument("--lr", type=float, default=BC_LR)
    parser.add_argument("--batch", type=int, default=BC_BATCH)
    args = parser.parse_args()

    # ── Fit obs normalizer on ALL h5 observations (success + fail) ────────────
    # Covers the full distribution the policy will see during fine-tuning.
    print(f"[INFO] Fitting obs normalizer from {args.h5}")
    with h5py.File(args.h5, "r") as f:
        all_obs = np.concatenate(
            [f[k]["observations"][:] for k in f.keys()], axis=0
        )
    obs_norm = ObsNormalizer()
    obs_norm.fit(all_obs)
    obs_norm.save(OBS_NORM_PATH)

    # ── Build and train policy ────────────────────────────────────────────────
    policy = GaussianPolicy().to(DEVICE)
    policy = train_bc(policy, args.h5, obs_norm,
                      epochs=args.epochs, lr=args.lr, batch=args.batch)

    # ── Final held-out evaluation ─────────────────────────────────────────────
    print(f"\n--- BC final evaluation (50 episodes) ---")
    ret_det, succ_det = env_eval(policy, obs_norm, n_episodes=50, deterministic=True)
    print(f"  Deterministic : return {ret_det:.2f}  success {succ_det:.2%}")
    print(f"\n[INFO] Saved checkpoint pair:")
    print(f"         Policy      → {BC_SAVE_PATH}")
    print(f"         Obs norm    → {OBS_NORM_PATH}")