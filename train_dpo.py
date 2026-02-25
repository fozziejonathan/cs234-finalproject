"""
train_dpo.py
------------
Stage 0: Behavioral Cloning (BC) on successful demonstrations
Stage 1: DPO fine-tuning against the frozen BC reference

BC design: simple MSE on action mean, trained only on successful trajectories.
No NLL, no state-dependent std — those caused std collapse and instability.
A global log_std is kept as a fixed parameter (not trained) to keep the policy
as a proper Gaussian for DPO log-prob computation.

DPO design: standard contrastive loss using preference pairs from actual
environment returns — clean signal, no synthetic noise model.
"""

import argparse
import os
import copy
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import metaworld

from config import (
    ENV_NAME, OBS_DIM, ACT_DIM, MAX_EPISODE_STEPS, HIDDEN_DIM,
    PREF_DATASET_PATH, DPO_LR, DPO_EPOCHS, DPO_BETA, DPO_SAVE_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

OBS_NORM_PATH = "checkpoints/obs_normalizer.npz"
BC_SAVE_PATH  = "checkpoints/bc_policy.pt"

# ── BC hyperparameters ────────────────────────────────────────────────────────
BC_LR         = 3e-4
BC_EPOCHS     = 60
BC_BATCH      = 256
BC_EVAL_EVERY = 5    # env eval every N epochs
BC_EVAL_EPS   = 20  # episodes per eval

# ── DPO hyperparameters ───────────────────────────────────────────────────────
DPO_BATCH      = 64
DPO_EVAL_EVERY = 2
DPO_EVAL_EPS   = 20

# Fixed log_std value: std = exp(-1) ≈ 0.37 — reasonable action spread,
# not trained by MSE BC (which has no std gradient), stable for DPO log-probs.
LOG_STD_INIT = -1.0

# Trajectory log-prob normalisation: sum over steps divided by nominal horizon
# so β stays meaningful regardless of actual episode length.
T_NORM = float(MAX_EPISODE_STEPS)


# ══════════════════════════════════════════════════════════════════════════════
# Observation Normalizer
# ══════════════════════════════════════════════════════════════════════════════

class ObsNormalizer:
    def __init__(self):
        self.mean = self.std = None

    def fit(self, all_obs: np.ndarray):
        self.mean = all_obs.mean(axis=0).astype(np.float32)
        self.std  = np.clip(all_obs.std(axis=0), 1e-8, None).astype(np.float32)
        print(f"[INFO] ObsNormalizer fit on {len(all_obs):,} observations.")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path):
        d = np.load(path)
        self.mean, self.std = d["mean"].astype(np.float32), d["std"].astype(np.float32)
        print(f"[INFO] ObsNormalizer loaded from {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Policy
# ══════════════════════════════════════════════════════════════════════════════

class GaussianPolicy(nn.Module):
    """
    π(a|s) = N(μ_θ(s), σ²I)

    μ is learned. log_std is a fixed scalar parameter — not trained during BC
    (MSE loss has no std gradient), kept stable for DPO log-prob computation.
    """
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        # Fixed log_std — not in optimizer, just used for DPO log-prob
        self.register_buffer("log_std",
                             torch.full((act_dim,), LOG_STD_INIT))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)

    def forward(self, obs):
        mu  = self.mu_head(self.trunk(obs))
        std = self.log_std.exp().expand_as(mu)
        return mu, std

    def get_action(self, obs, deterministic=True):
        with torch.no_grad():
            mu, std = self.forward(obs)
            action  = mu if deterministic else Normal(mu, std).sample()
        return action.squeeze(0).cpu().numpy()

    def traj_log_prob(self, obs_seq, act_seq):
        """log π(τ) = Σ_t log π(a_t|s_t) / T_NORM"""
        mu, std = self.forward(obs_seq)
        step_lp = Normal(mu, std).log_prob(act_seq).sum(dim=-1)  # (T,)
        return step_lp.sum() / T_NORM                             # scalar


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0: Behavioral Cloning
# ══════════════════════════════════════════════════════════════════════════════

def build_bc_data(train_pairs, obs_norm):
    """
    Use ONLY successful chosen trajectories.
    These are demonstrations where the robot actually reached the goal —
    the exact behaviors we want BC to imitate.
    """
    successful = [p for p in train_pairs if p["chosen"]["success"]]
    total = len(train_pairs)
    print(f"[BC] Successful chosen trajs: {len(successful)}/{total} "
          f"({len(successful)/total:.1%})")

    if len(successful) == 0:
        print("[BC] WARNING: No successful trajectories found! "
              "Using all chosen trajectories as fallback. "
              "Re-collect data with noise=0.02.")
        successful = train_pairs

    all_obs, all_acts = [], []
    for p in successful:
        obs_n = obs_norm.normalize(p["chosen"]["observations"].astype(np.float32))
        all_obs.append(obs_n)
        all_acts.append(p["chosen"]["actions"].astype(np.float32))

    obs_t  = torch.FloatTensor(np.concatenate(all_obs)).to(DEVICE)
    acts_t = torch.FloatTensor(np.concatenate(all_acts)).to(DEVICE)
    print(f"[BC] Dataset: {len(obs_t):,} transitions from "
          f"{len(successful)} successful episodes")
    return obs_t, acts_t


def env_eval(policy, obs_norm, n_episodes=BC_EVAL_EPS, deterministic=True):
    """Run policy in MetaWorld, return (mean_return, success_rate)."""
    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks
    policy.eval()
    returns, successes = [], []
    with torch.no_grad():
        for i in range(n_episodes):
            env.set_task(tasks[i % len(tasks)])
            obs, _ = env.reset()
            ep_ret, success = 0.0, False
            for _ in range(MAX_EPISODE_STEPS):
                obs_n  = obs_norm.normalize(obs.astype(np.float32))
                obs_t  = torch.FloatTensor(obs_n).unsqueeze(0).to(DEVICE)
                action = policy.get_action(obs_t, deterministic=deterministic)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += reward
                if info.get("success", False):
                    success = True
                if terminated or truncated:
                    break
            returns.append(ep_ret)
            successes.append(float(success))
    policy.train()
    return float(np.mean(returns)), float(np.mean(successes))


def train_bc(policy, train_pairs, obs_norm):
    obs_t, acts_t = build_bc_data(train_pairs, obs_norm)
    N = len(obs_t)

    # Only optimize trunk + mu_head — log_std is a buffer (not a parameter)
    optimizer = optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=BC_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=BC_EPOCHS)

    print(f"\n{'='*60}")
    print(f"  Stage 0: Behavioral Cloning")
    print(f"  Transitions: {N:,}  |  Epochs: {BC_EPOCHS}  |  Batch: {BC_BATCH}")
    print(f"  Loss: MSE(μ_θ(s), a_expert)  |  LR: {BC_LR}")
    print(f"  Eval every {BC_EVAL_EVERY} epochs  |  Saving by success rate")
    print(f"{'='*60}")

    best_success = -1.0
    os.makedirs(os.path.dirname(BC_SAVE_PATH), exist_ok=True)

    for epoch in range(BC_EPOCHS):
        policy.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches  = 0

        for start in range(0, N, BC_BATCH):
            idx    = perm[start:start + BC_BATCH]
            obs_b  = obs_t[idx]
            acts_b = acts_t[idx]

            mu, _ = policy.forward(obs_b)
            loss  = F.mse_loss(mu, acts_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        if (epoch + 1) % BC_EVAL_EVERY == 0 or epoch == 0:
            ret_d, succ_d = env_eval(policy, obs_norm, BC_EVAL_EPS,
                                     deterministic=True)
            ret_s, succ_s = env_eval(policy, obs_norm, BC_EVAL_EPS,
                                     deterministic=False)
            improved = succ_d > best_success
            if improved:
                best_success = succ_d
                torch.save(policy.state_dict(), BC_SAVE_PATH)
            print(f"  Epoch {epoch+1:>3}/{BC_EPOCHS}  "
                  f"| MSE: {epoch_loss/n_batches:.5f}  "
                  f"| success det/stoch: {succ_d:.2%}/{succ_s:.2%}  "
                  f"| return det/stoch: {ret_d:.1f}/{ret_s:.1f}"
                  + (" ✓" if improved else ""))

    if best_success < 0:
        torch.save(policy.state_dict(), BC_SAVE_PATH)
    print(f"\n  Best BC checkpoint: {best_success:.2%} success → {BC_SAVE_PATH}\n")
    policy.load_state_dict(torch.load(BC_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: DPO Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

def dpo_loss(policy, ref_policy, obs_w, act_w, obs_l, act_l, beta):
    log_pi_w  = policy.traj_log_prob(obs_w, act_w)
    log_pi_l  = policy.traj_log_prob(obs_l, act_l)
    with torch.no_grad():
        log_ref_w = ref_policy.traj_log_prob(obs_w, act_w)
        log_ref_l = ref_policy.traj_log_prob(obs_l, act_l)

    rw = beta * (log_pi_w - log_ref_w)
    rl = beta * (log_pi_l - log_ref_l)
    loss = -F.logsigmoid(rw - rl)
    acc  = float((rw > rl).item())
    return loss, {"reward_w": rw.item(), "reward_l": rl.item(),
                  "margin": (rw - rl).item(), "accuracy": acc}


def train_dpo(policy, ref_policy, train_data, val_data, obs_norm,
              epochs=DPO_EPOCHS, lr=DPO_LR, beta=DPO_BETA):

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'='*60}")
    print(f"  Stage 1: DPO Fine-tuning")
    print(f"  Pairs — train: {len(train_data)}  val: {len(val_data)}")
    print(f"  β={beta}  lr={lr}  epochs={epochs}")
    print(f"  Reference: frozen BC checkpoint")
    print(f"{'='*60}")

    best_success = -1.0
    os.makedirs(os.path.dirname(DPO_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        policy.train()
        np.random.shuffle(train_data)

        epoch_loss = epoch_acc = 0.0
        n_batches  = 0

        for start in range(0, len(train_data), DPO_BATCH):
            batch      = train_data[start:start + DPO_BATCH]
            batch_loss = 0.0
            batch_acc  = 0.0

            for pair in batch:
                obs_w = torch.FloatTensor(
                    obs_norm.normalize(pair["chosen"]["observations"])).to(DEVICE)
                act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
                obs_l = torch.FloatTensor(
                    obs_norm.normalize(pair["rejected"]["observations"])).to(DEVICE)
                act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)

                loss, m  = dpo_loss(policy, ref_policy,
                                    obs_w, act_w, obs_l, act_l, beta)
                batch_loss += loss
                batch_acc  += m["accuracy"]

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_acc  += batch_acc / len(batch)
            n_batches  += 1

        scheduler.step()

        # ── Val preference accuracy (every epoch, free) ───────────────────────
        policy.eval()
        val_acc = 0.0
        with torch.no_grad():
            for pair in val_data:
                obs_w = torch.FloatTensor(
                    obs_norm.normalize(pair["chosen"]["observations"])).to(DEVICE)
                act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
                obs_l = torch.FloatTensor(
                    obs_norm.normalize(pair["rejected"]["observations"])).to(DEVICE)
                act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)
                _, m  = dpo_loss(policy, ref_policy,
                                 obs_w, act_w, obs_l, act_l, beta)
                val_acc += m["accuracy"]
        val_acc /= len(val_data)

        # ── Env eval (every DPO_EVAL_EVERY epochs) ────────────────────────────
        if (epoch + 1) % DPO_EVAL_EVERY == 0 or epoch == epochs - 1:
            ret_d, succ_d = env_eval(policy, obs_norm, DPO_EVAL_EPS,
                                     deterministic=True)
            ret_s, succ_s = env_eval(policy, obs_norm, DPO_EVAL_EPS,
                                     deterministic=False)
            improved = succ_d > best_success
            if improved:
                best_success = succ_d
                torch.save(policy.state_dict(), DPO_SAVE_PATH)
            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"| loss: {epoch_loss/n_batches:.4f}  "
                  f"pref_acc: {epoch_acc/n_batches:.2%}  "
                  f"val_acc: {val_acc:.2%}  "
                  f"| success det/stoch: {succ_d:.2%}/{succ_s:.2%}  "
                  f"| return: {ret_d:.1f}/{ret_s:.1f}"
                  + (" ✓" if improved else ""))
        else:
            print(f"  Epoch {epoch+1:>3}/{epochs}  "
                  f"| loss: {epoch_loss/n_batches:.4f}  "
                  f"pref_acc: {epoch_acc/n_batches:.2%}  "
                  f"val_acc: {val_acc:.2%}")

    if best_success < 0:
        torch.save(policy.state_dict(), DPO_SAVE_PATH)
    print(f"\n  Best DPO checkpoint: {best_success:.2%} success → {DPO_SAVE_PATH}\n")
    policy.load_state_dict(torch.load(DPO_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default=PREF_DATASET_PATH)
    parser.add_argument("--bc_epochs", type=int,   default=BC_EPOCHS)
    parser.add_argument("--dpo_epochs",type=int,   default=DPO_EPOCHS)
    parser.add_argument("--lr",        type=float, default=DPO_LR)
    parser.add_argument("--beta",      type=float, default=DPO_BETA)
    parser.add_argument("--skip_bc",   action="store_true")
    args = parser.parse_args()

    # Load dataset
    with open(args.data, "rb") as f:
        ds = pickle.load(f)
    train_data, val_data = ds["train"], ds["val"]

    # Fit obs normalizer on all training observations
    obs_norm = ObsNormalizer()
    all_obs = np.concatenate(
        [p["chosen"]["observations"] for p in train_data] +
        [p["rejected"]["observations"] for p in train_data], axis=0)
    obs_norm.fit(all_obs)
    obs_norm.save(OBS_NORM_PATH)

    # Build policy
    policy = GaussianPolicy().to(DEVICE)

    # Stage 0: BC
    if args.skip_bc and os.path.exists(BC_SAVE_PATH):
        print(f"[INFO] Loading BC policy from {BC_SAVE_PATH}")
        policy.load_state_dict(torch.load(BC_SAVE_PATH, map_location=DEVICE))
    else:
        policy = train_bc(policy, train_data, obs_norm)

    print(f"\n--- BC final evaluation ---")
    ret_d, succ_d = env_eval(policy, obs_norm, n_episodes=50, deterministic=True)
    ret_s, succ_s = env_eval(policy, obs_norm, n_episodes=50, deterministic=False)
    print(f"  Deterministic:  return {ret_d:.2f}  success {succ_d:.2%}")
    print(f"  Stochastic:     return {ret_s:.2f}  success {succ_s:.2%}")

    # Stage 1: DPO
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    policy = train_dpo(policy, ref_policy, train_data, val_data, obs_norm,
                       epochs=args.dpo_epochs, lr=args.lr, beta=args.beta)

    print(f"\n--- DPO final evaluation ---")
    ret_d, succ_d = env_eval(policy, obs_norm, n_episodes=50, deterministic=True)
    ret_s, succ_s = env_eval(policy, obs_norm, n_episodes=50, deterministic=False)
    print(f"  Deterministic:  return {ret_d:.2f}  success {succ_d:.2%}")
    print(f"  Stochastic:     return {ret_s:.2f}  success {succ_s:.2%}")