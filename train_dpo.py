"""
train_dpo.py
------------
Phase 4: Direct Preference Optimization (DPO) for Continuous Robotic Control

Pipeline (mirrors LLM DPO):
    Stage 0: Behavioral Cloning (BC) on chosen trajectories  ← NEW
    Stage 1: DPO fine-tuning against the frozen BC reference

Why Stage 0 matters:
    DPO's implicit reward is  β(log π_θ - log π_ref).  If π_ref is random,
    this quantity has nothing to do with task competence — the policy just
    learns to drift away from random noise.  We need π_ref to be a policy
    that already knows how to do the task so that the DPO loss has a
    meaningful signal to contrastively refine.
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
    ENV_NAME, OBS_DIM, ACT_DIM, MAX_EPISODE_STEPS,
    HIDDEN_DIM, PREF_DATASET_PATH,
    DPO_LR, DPO_EPOCHS, DPO_BETA, DPO_SAVE_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_NORM_PATH = "checkpoints/obs_normalizer.npz"
DPO_BATCH_SIZE = 64
BC_SAVE_PATH = "checkpoints/bc_policy.pt"  # BC (SFT) checkpoint used as π_ref

# ── Hyperparameters ──────────────────────────────────────────────────────────
BC_LR = 3e-4
BC_EPOCHS = 30  # Enough to get mean-action BC to converge
BC_BATCH_SIZE = 256  # Standard supervised learning batch size

print(f"[INFO] Running on device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# Obs Normalizer (same as RLHF version — must share stats)
# ══════════════════════════════════════════════════════════════════════════════

class ObsNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def load(self, path: str):
        data = np.load(path)
        self.mean = data["mean"].astype(np.float32)
        self.std = data["std"].astype(np.float32)
        print(f"[INFO] ObsNormalizer loaded from {path}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std

    def normalize_tensor(self, obs_t: torch.Tensor) -> torch.Tensor:
        mean_t = torch.FloatTensor(self.mean).to(obs_t.device)
        std_t = torch.FloatTensor(self.std).to(obs_t.device)
        return (obs_t - mean_t) / std_t


# ══════════════════════════════════════════════════════════════════════════════
# Policy (identical architecture to RLHF for fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousDPOPolicy(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        x = self.trunk(obs)
        mu = self.mu_head(x)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 2.0)).expand_as(mu)
        return mu, std

    def get_trajectory_log_prob(self, obs_seq: torch.Tensor,
                                act_seq: torch.Tensor) -> torch.Tensor:
        """log π_θ(τ) = mean_t [ log π_θ(a_t | s_t) ]"""
        mu, std = self.forward(obs_seq)
        step_lp = Normal(mu, std).log_prob(act_seq).sum(dim=-1)  # (T,)
        return step_lp.mean()  # scalar

    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            mu, _ = self.forward(obs)
        return mu.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0: Behavioral Cloning
# ══════════════════════════════════════════════════════════════════════════════

def build_bc_dataset(train_data: list, obs_normalizer: ObsNormalizer):
    """
    Flatten all chosen trajectories into (obs, action) pairs for BC.
    We only use CHOSEN trajectories — these are the high-return demonstrations
    we want the policy to imitate.
    """
    all_obs, all_acts = [], []
    for pair in train_data:
        obs_raw = pair["chosen"]["observations"]  # (T, 39)
        acts = pair["chosen"]["actions"]  # (T, 4)
        obs_norm = obs_normalizer.normalize(obs_raw.astype(np.float32))
        all_obs.append(obs_norm)
        all_acts.append(acts.astype(np.float32))
    all_obs = np.concatenate(all_obs, axis=0)  # (N*T, 39)
    all_acts = np.concatenate(all_acts, axis=0)  # (N*T, 4)
    print(f"[BC] Dataset: {len(all_obs):,} (obs, action) pairs from "
          f"{len(train_data)} chosen trajectories")
    return all_obs, all_acts


def pretrain_bc(policy: ContinuousDPOPolicy, train_data: list,
                obs_normalizer: ObsNormalizer,
                epochs: int = BC_EPOCHS, lr: float = BC_LR,
                batch_size: int = BC_BATCH_SIZE) -> ContinuousDPOPolicy:
    """
    Stage 0 — Supervised Behavioral Cloning on chosen trajectories.

    Loss = MSE( μ_θ(s), a_expert )
    We use MSE on the mean (not NLL) because:
      - It's simpler and converges faster
      - The log_std parameter will be optimised jointly via NLL in DPO stage
      - Matches standard BC practice for continuous control

    After this stage the policy will already achieve a non-trivial success rate,
    and π_ref = deepcopy(policy) will be a meaningful anchor for DPO.
    """
    all_obs, all_acts = build_bc_dataset(train_data, obs_normalizer)
    obs_t = torch.FloatTensor(all_obs).to(DEVICE)
    acts_t = torch.FloatTensor(all_acts).to(DEVICE)
    N = len(obs_t)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'=' * 60}")
    print(f"  Stage 0: Behavioral Cloning  ({N:,} transitions, {epochs} epochs)")
    print(f"  Batch size: {batch_size}  |  LR: {lr}")
    print(f"{'=' * 60}")

    best_loss = float("inf")
    for epoch in range(epochs):
        policy.train()
        perm = torch.randperm(N)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start: start + batch_size]
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
        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  BC Epoch {epoch + 1:>3}/{epochs}  | loss: {avg_loss:.5f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(os.path.dirname(BC_SAVE_PATH), exist_ok=True)
            torch.save(policy.state_dict(), BC_SAVE_PATH)

    print(f"  Best BC policy saved → {BC_SAVE_PATH}  (loss: {best_loss:.5f})\n")
    policy.load_state_dict(torch.load(BC_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# DPO Loss
# ══════════════════════════════════════════════════════════════════════════════

def compute_dpo_loss(
        policy: ContinuousDPOPolicy,
        ref_policy: ContinuousDPOPolicy,
        obs_w: torch.Tensor, act_w: torch.Tensor,
        obs_l: torch.Tensor, act_l: torch.Tensor,
        beta: float = DPO_BETA,
        margin: float = 0.0,  # optional margin-weighted loss
) -> tuple:
    log_pi_w = policy.get_trajectory_log_prob(obs_w, act_w)
    log_pi_l = policy.get_trajectory_log_prob(obs_l, act_l)

    with torch.no_grad():
        log_ref_w = ref_policy.get_trajectory_log_prob(obs_w, act_w)
        log_ref_l = ref_policy.get_trajectory_log_prob(obs_l, act_l)

    implicit_rw = beta * (log_pi_w - log_ref_w)
    implicit_rl = beta * (log_pi_l - log_ref_l)

    # Standard DPO loss
    loss = -F.logsigmoid(implicit_rw - implicit_rl)

    with torch.no_grad():
        delta = (implicit_rw - implicit_rl).item()
        accuracy = float(implicit_rw > implicit_rl)

    metrics = {
        "reward_w": implicit_rw.item(),
        "reward_l": implicit_rl.item(),
        "margin": delta,
        "accuracy": accuracy,
        "kl_chosen": (log_pi_w - log_ref_w).item() / beta,
    }
    return loss, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: DPO Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════

def run_dpo_alignment(
        data_path: str = PREF_DATASET_PATH,
        epochs: int = DPO_EPOCHS,
        lr: float = DPO_LR,
        beta: float = DPO_BETA,
        bc_epochs: int = BC_EPOCHS,
        skip_bc: bool = False,
) -> ContinuousDPOPolicy:
    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(data_path, "rb") as f:
        dataset_dict = pickle.load(f)
    train_data = dataset_dict["train"]
    val_data = dataset_dict["val"]

    # ── Load obs normalizer ───────────────────────────────────────────────────
    obs_normalizer = ObsNormalizer()
    obs_normalizer.load(OBS_NORM_PATH)

    # ── Build policy ──────────────────────────────────────────────────────────
    policy = ContinuousDPOPolicy().to(DEVICE)

    # ── Stage 0: BC pretraining ───────────────────────────────────────────────
    if skip_bc and os.path.exists(BC_SAVE_PATH):
        print(f"[INFO] Loading existing BC policy from {BC_SAVE_PATH}")
        policy.load_state_dict(torch.load(BC_SAVE_PATH, map_location=DEVICE))
    else:
        policy = pretrain_bc(policy, train_data, obs_normalizer, epochs=bc_epochs)

    # ── Freeze reference policy at BC checkpoint ──────────────────────────────
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False

    # ── Stage 1: DPO fine-tuning ──────────────────────────────────────────────
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'=' * 60}")
    print(f"  Stage 1: DPO Fine-tuning  (β={beta}, lr={lr}, {epochs} epochs)")
    print(f"  Train pairs: {len(train_data)}  |  Val pairs: {len(val_data)}")
    print(f"  Reference policy: BC checkpoint (NOT random init)")
    print(f"{'=' * 60}")

    best_val_acc = -1.0
    os.makedirs(os.path.dirname(DPO_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        policy.train()
        np.random.shuffle(train_data)

        epoch_loss = epoch_acc = epoch_margin = epoch_kl = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_data), DPO_BATCH_SIZE):
            batch = train_data[batch_start: batch_start + DPO_BATCH_SIZE]
            batch_loss = 0.0
            batch_acc = batch_margin = batch_kl = 0.0

            for pair in batch:
                obs_w = torch.FloatTensor(
                    obs_normalizer.normalize(pair["chosen"]["observations"])).to(DEVICE)
                act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
                obs_l = torch.FloatTensor(
                    obs_normalizer.normalize(pair["rejected"]["observations"])).to(DEVICE)
                act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)

                loss, metrics = compute_dpo_loss(
                    policy, ref_policy, obs_w, act_w, obs_l, act_l,
                    beta=beta, margin=pair["margin"])

                batch_loss += loss
                batch_acc += metrics["accuracy"]
                batch_margin += metrics["margin"]
                batch_kl += metrics["kl_chosen"]

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc / len(batch)
            epoch_margin += batch_margin / len(batch)
            epoch_kl += batch_kl / len(batch)
            n_batches += 1

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        policy.eval()
        val_loss = val_acc = 0.0
        with torch.no_grad():
            for pair in val_data:
                obs_w = torch.FloatTensor(
                    obs_normalizer.normalize(pair["chosen"]["observations"])).to(DEVICE)
                act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
                obs_l = torch.FloatTensor(
                    obs_normalizer.normalize(pair["rejected"]["observations"])).to(DEVICE)
                act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)
                loss, metrics = compute_dpo_loss(
                    policy, ref_policy, obs_w, act_w, obs_l, act_l, beta=beta)
                val_loss += loss.item()
                val_acc += metrics["accuracy"]

        n_v = len(val_data)
        print(f"  Epoch {epoch + 1:>3}/{epochs}  "
              f"| train loss: {epoch_loss / n_batches:.4f}  "
              f"acc: {epoch_acc / n_batches:.2%}  "
              f"margin: {epoch_margin / n_batches:.3f}  "
              f"KL: {epoch_kl / n_batches:.3f}  "
              f"| val loss: {val_loss / n_v:.4f}  "
              f"acc: {val_acc / n_v:.2%}")

        if val_acc / n_v > best_val_acc:
            best_val_acc = val_acc / n_v
            torch.save(policy.state_dict(), DPO_SAVE_PATH)

    print(f"\n  Best DPO policy saved → {DPO_SAVE_PATH}  "
          f"(val acc: {best_val_acc:.2%})\n")
    policy.load_state_dict(torch.load(DPO_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dpo_policy(policy: ContinuousDPOPolicy, n_episodes: int = 50) -> dict:
    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks

    obs_normalizer = ObsNormalizer()
    obs_normalizer.load(OBS_NORM_PATH)
    policy.eval()

    returns, successes = [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            task = tasks[np.random.randint(len(tasks))]
            env.set_task(task)
            obs, _ = env.reset()
            ep_ret, success = 0.0, False

            for _ in range(MAX_EPISODE_STEPS):
                obs_norm = obs_normalizer.normalize(obs.astype(np.float32))
                obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(DEVICE)
                action = policy.get_action(obs_t)
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += reward
                if info.get("success", False):
                    success = True
                if terminated or truncated:
                    break

            returns.append(ep_ret)
            successes.append(float(success))

    results = {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
    }
    print(f"[DPO Eval]  Return: {results['mean_return']:.2f} ± "
          f"{results['std_return']:.2f}  "
          f"|  Success rate: {results['success_rate']:.2%}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO for continuous control.")
    parser.add_argument("--data", default=PREF_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=DPO_EPOCHS)
    parser.add_argument("--bc_epochs", type=int, default=BC_EPOCHS,
                        help="Epochs for BC pretraining stage")
    parser.add_argument("--lr", type=float, default=DPO_LR)
    parser.add_argument("--beta", type=float, default=DPO_BETA,
                        help="KL regularisation strength (0.05–0.2 for robotics)")
    parser.add_argument("--skip_bc", action="store_true",
                        help="Skip BC pretraining and load existing BC checkpoint")
    parser.add_argument("--eval", action="store_true",
                        help="Run eval after training")
    args = parser.parse_args()

    policy = run_dpo_alignment(
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        bc_epochs=args.bc_epochs,
        skip_bc=args.skip_bc,
    )

    if args.eval:
        # First evaluate BC alone (sanity check)
        print("\n--- BC policy performance (before DPO) ---")
        bc_policy = ContinuousDPOPolicy().to(DEVICE)
        bc_policy.load_state_dict(torch.load(BC_SAVE_PATH, map_location=DEVICE))
        evaluate_dpo_policy(bc_policy, n_episodes=50)

        print("\n--- DPO policy performance (after DPO) ---")
        evaluate_dpo_policy(policy, n_episodes=50)
