"""
train_dpo.py
------------
Phase 4: Direct Preference Optimization (DPO) for Continuous Robotic Control

DPO bypasses explicit reward modelling entirely.  The key mathematical insight
(Rafailov et al. 2024) is that the implicit reward of a state-action pair can
be expressed as a log-probability ratio between the active policy and a frozen
reference policy:

    r_implicit(s, a) = β · [ log π_θ(a|s) - log π_ref(a|s) ]

For continuous trajectories the log-probability ratio telescopes to a temporal
sum (environment dynamics cancel because they are policy-independent):

    log π_θ(τ) / π_ref(τ) = Σ_t [ log π_θ(a_t|s_t) - log π_ref(a_t|s_t) ]

The DPO loss is then a binary cross-entropy over these trajectory-level
implicit reward margins:

    L_DPO(θ, π_ref) = -E_{(τ_w,τ_l)~D}[
        log σ( β · ( Σ_t log π_θ(a_t^w|s_t^w) - Σ_t log π_ref(a_t^w|s_t^w)
                   - Σ_t log π_θ(a_t^l|s_t^l) + Σ_t log π_ref(a_t^l|s_t^l) ) )
    ]

Key design choices for continuous domains:
    • Policy outputs a diagonal Gaussian: log π_θ(a|s) = log N(a; μ_θ(s), σ²_θ)
    • Trajectory log-prob = Σ_t Σ_d log N(a_{t,d}; μ_{t,d}, σ²_{t,d})
    • β controls KL divergence against reference — tune carefully (0.05–0.2)
    • No environment interaction required — fully offline

Usage
-----
    python train_dpo.py
    python train_dpo.py --beta 0.05 --epochs 30 --lr 5e-6
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
print(f"[INFO] Running on device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# Policy Architecture (identical structure to RLHF policy for fair comparison)
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousDPOPolicy(nn.Module):
    """
    π_θ(a|s) = N( μ_θ(s), diag(σ²_θ) )

    Identical architecture to the RLHF Gaussian policy so that capacity
    differences don't confound the RLHF vs DPO comparison.
    """

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

        # Orthogonal initialisation (standard for continuous RL)
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

    def get_trajectory_log_prob(
            self, obs_seq: torch.Tensor, act_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log π_θ(τ) = Σ_t log π_θ(a_t | s_t)

        Each time-step contributes the sum of marginal log-probabilities
        over action dimensions (equivalent to a diagonal-covariance Gaussian).

        Parameters
        ----------
        obs_seq : (T, obs_dim)
        act_seq : (T, act_dim)

        Returns
        -------
        Scalar tensor (trajectory log-probability).
        """
        mu, std = self.forward(obs_seq)  # (T, act_dim)
        dist = Normal(mu, std)
        step_lp = dist.log_prob(act_seq).sum(dim=-1)  # (T,)  sum over act dims
        return step_lp.sum()  # scalar: sum over time

    def get_action(self, obs: torch.Tensor) -> np.ndarray:
        """Deterministic action for evaluation (use mean of Gaussian)."""
        with torch.no_grad():
            mu, _ = self.forward(obs)
        return mu.squeeze(0).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# DPO Loss
# ══════════════════════════════════════════════════════════════════════════════

def compute_dpo_loss(
        policy: ContinuousDPOPolicy,
        ref_policy: ContinuousDPOPolicy,
        obs_w: torch.Tensor,  # chosen   observations  (T_w, obs_dim)
        act_w: torch.Tensor,  # chosen   actions       (T_w, act_dim)
        obs_l: torch.Tensor,  # rejected observations  (T_l, obs_dim)
        act_l: torch.Tensor,  # rejected actions       (T_l, act_dim)
        beta: float = DPO_BETA,
) -> tuple[torch.Tensor, dict]:
    """
    Full DPO contrastive loss for a single preference pair.

    Returns
    -------
    loss    : scalar tensor (to be .backward()'d)
    metrics : dict with reward_w, reward_l, margin, accuracy
    """
    # ── Active policy log-probs (gradients flow through these) ────────────────
    log_pi_w = policy.get_trajectory_log_prob(obs_w, act_w)
    log_pi_l = policy.get_trajectory_log_prob(obs_l, act_l)

    # ── Reference policy log-probs (frozen — no gradients) ───────────────────
    with torch.no_grad():
        log_ref_w = ref_policy.get_trajectory_log_prob(obs_w, act_w)
        log_ref_l = ref_policy.get_trajectory_log_prob(obs_l, act_l)

    # ── Implicit reward margins ───────────────────────────────────────────────
    # r_implicit(τ) = β · [log π_θ(τ) - log π_ref(τ)]
    implicit_reward_w = beta * (log_pi_w - log_ref_w)
    implicit_reward_l = beta * (log_pi_l - log_ref_l)

    # ── DPO contrastive loss ──────────────────────────────────────────────────
    loss = -F.logsigmoid(implicit_reward_w - implicit_reward_l)

    # Diagnostics
    with torch.no_grad():
        margin = (implicit_reward_w - implicit_reward_l).item()
        accuracy = float(implicit_reward_w > implicit_reward_l)
        kl_w = (log_pi_w - log_ref_w).item() / beta  # unnormalised KL contribution

    metrics = {
        "reward_w": implicit_reward_w.item(),
        "reward_l": implicit_reward_l.item(),
        "margin": margin,
        "accuracy": accuracy,
        "kl_chosen": kl_w,
    }
    return loss, metrics


# ══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ══════════════════════════════════════════════════════════════════════════════

def run_dpo_alignment(
        data_path: str = PREF_DATASET_PATH,
        epochs: int = DPO_EPOCHS,
        lr: float = DPO_LR,
        beta: float = DPO_BETA,
) -> ContinuousDPOPolicy:
    """
    Full DPO training loop.

    Because DPO is off-policy and uses no environment interaction,
    this is much simpler and cheaper than PPO.  The entire training
    runs on the static preference dataset.
    """
    # ── Load dataset ──────────────────────────────────────────────────────────
    with open(data_path, "rb") as f:
        dataset_dict = pickle.load(f)
    train_data = dataset_dict["train"]
    val_data = dataset_dict["val"]

    # ── Build policy and frozen reference ────────────────────────────────────
    policy = ContinuousDPOPolicy().to(DEVICE)

    # Reference policy = deep copy, completely frozen
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for param in ref_policy.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'=' * 60}")
    print(f"  DPO Training  (β={beta}, lr={lr}, {epochs} epochs)")
    print(f"  Train pairs: {len(train_data)}  |  Val pairs: {len(val_data)}")
    print(f"{'=' * 60}")

    best_val_acc = -1.0
    os.makedirs(os.path.dirname(DPO_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        policy.train()
        np.random.shuffle(train_data)

        epoch_loss = epoch_acc = epoch_margin = epoch_kl = 0.0

        for pair in train_data:
            obs_w = torch.FloatTensor(pair["chosen"]["observations"]).to(DEVICE)
            act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
            obs_l = torch.FloatTensor(pair["rejected"]["observations"]).to(DEVICE)
            act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)

            loss, metrics = compute_dpo_loss(
                policy, ref_policy, obs_w, act_w, obs_l, act_l, beta=beta
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += metrics["accuracy"]
            epoch_margin += metrics["margin"]
            epoch_kl += metrics["kl_chosen"]

        scheduler.step()

        # ── Validation ───────────────────────────────────────────────────────
        policy.eval()
        val_loss = val_acc = 0.0
        with torch.no_grad():
            for pair in val_data:
                obs_w = torch.FloatTensor(pair["chosen"]["observations"]).to(DEVICE)
                act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
                obs_l = torch.FloatTensor(pair["rejected"]["observations"]).to(DEVICE)
                act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)
                loss, metrics = compute_dpo_loss(
                    policy, ref_policy, obs_w, act_w, obs_l, act_l, beta=beta
                )
                val_loss += loss.item()
                val_acc += metrics["accuracy"]

        n_tr, n_v = len(train_data), len(val_data)
        print(f"  Epoch {epoch + 1:>3}/{epochs}  "
              f"| train loss: {epoch_loss / n_tr:.4f}  acc: {epoch_acc / n_tr:.2%}  "
              f"margin: {epoch_margin / n_tr:.3f}  "
              f"KL(chosen): {epoch_kl / n_tr:.3f}  "
              f"| val loss: {val_loss / n_v:.4f}  acc: {val_acc / n_v:.2%}")

        if val_acc / n_v > best_val_acc:
            best_val_acc = val_acc / n_v
            torch.save(policy.state_dict(), DPO_SAVE_PATH)

    print(f"\n  Best DPO policy saved → {DPO_SAVE_PATH}  "
          f"(val acc: {best_val_acc:.2%})\n")

    policy.load_state_dict(torch.load(DPO_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Quick In-Env Evaluation Helper
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dpo_policy(policy: ContinuousDPOPolicy, n_episodes: int = 20) -> dict:
    """Run the DPO policy in MetaWorld and return task success / return stats."""
    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks

    policy.eval()
    returns, successes = [], []

    with torch.no_grad():
        for _ in range(n_episodes):
            task = tasks[np.random.randint(len(tasks))]
            env.set_task(task)
            obs, _ = env.reset()
            ep_ret, success = 0.0, False

            for _ in range(MAX_EPISODE_STEPS):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
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
    print(f"[DPO Eval]  Return: {results['mean_return']:.2f} ± {results['std_return']:.2f}  "
          f"|  Success rate: {results['success_rate']:.2%}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO for continuous control.")
    parser.add_argument("--data", default=PREF_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=DPO_EPOCHS)
    parser.add_argument("--lr", type=float, default=DPO_LR)
    parser.add_argument("--beta", type=float, default=DPO_BETA,
                        help="KL regularisation strength (0.05–0.2 for robotics)")
    parser.add_argument("--eval", action="store_true", help="Run eval after training")
    args = parser.parse_args()

    policy = run_dpo_alignment(
        data_path=args.data,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
    )

    if args.eval:
        evaluate_dpo_policy(policy, n_episodes=50)
