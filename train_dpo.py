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
BC_SAVE_PATH = "checkpoints/bc_policy.pt"   # BC (SFT) checkpoint used as π_ref

# ── Hyperparameters ──────────────────────────────────────────────────────────
BC_LR = 3e-4
BC_EPOCHS = 80          # NLL converges slower than MSE — give it time
BC_BATCH_SIZE = 256
BC_TOP_FRAC = 0.30      # Only imitate top 30% of chosen trajectories by return
                         # "Chosen" just means better of two similar trajs —
                         # many are still mediocre. Filter to genuinely good ones.
BC_EVAL_EVERY = 10      # Evaluate in environment every N epochs during BC
BC_EVAL_EPISODES = 20   # Episodes per BC eval (fast)
BC_STD_MIN_START = 0.30 # std floor at start of BC — prevents collapse before μ converges
BC_STD_MIN_END   = 0.05 # std floor at end of BC — allows final precision
                         # Linearly annealed epoch 0 → BC_EPOCHS
DPO_EVAL_EVERY = 2      # Evaluate in environment every N epochs during DPO
DPO_EVAL_EPISODES = 20  # Episodes per DPO eval

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
# Policy
# ══════════════════════════════════════════════════════════════════════════════

# Trajectory log-prob normalisation constant.
# We use sum() over timesteps (mathematically correct: log π(τ) = Σ_t log π(a_t|s_t))
# but divide by T_NORM so the scale stays stable regardless of actual episode length
# and β remains meaningful at its configured value.
# Using raw sum() without normalisation would inflate log probs ~150x vs mean(),
# requiring β to be rescaled by the same factor. Dividing by the fixed nominal
# horizon gives us sum semantics (correct gradient direction) with mean-like scale.
T_NORM = float(MAX_EPISODE_STEPS)  # 150 for reach-v3


class ContinuousDPOPolicy(nn.Module):
    """
    π_θ(a|s) = N( μ_θ(s), diag(σ²_θ(s)) )

    Key design: σ is STATE-DEPENDENT (heteroscedastic).
    A global log_std scalar forces the same variance everywhere — but in reach-v3
    the policy should be precise (low σ) near the goal and loose (high σ) in free
    space. A log_std_head conditioned on the observation learns this automatically
    via NLL, producing more committed actions near the goal.
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
        self.mu_head      = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # Small gain on output heads — keeps initial actions and log_stds near 0
        nn.init.orthogonal_(self.mu_head.weight,      gain=0.01)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        x       = self.trunk(obs)
        mu      = self.mu_head(x)
        # Clamp tighter on the lower end (-5) so std never collapses to ~0
        # which would cause NLL to blow up early in training
        log_std = self.log_std_head(x)
        std     = torch.exp(torch.clamp(log_std, -5.0, 2.0))
        return mu, std

    def get_trajectory_log_prob(self, obs_seq: torch.Tensor,
                                act_seq: torch.Tensor) -> torch.Tensor:
        """
        Compute log π_θ(τ) normalised by trajectory length.

        Mathematically: log π(τ) = Σ_t log π(a_t|s_t)   [sum is correct]
        We divide by T_NORM (fixed nominal horizon = 150) so the magnitude
        stays consistent with β's configured value regardless of actual T.
        This gives correct gradient direction while avoiding sigmoid saturation.
        """
        mu, std   = self.forward(obs_seq)                          # (T, act_dim)
        step_lp   = Normal(mu, std).log_prob(act_seq).sum(dim=-1)  # (T,) — sum over action dims
        return step_lp.sum() / T_NORM                              # scalar — sum over time, normalised

    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> np.ndarray:
        """
        deterministic=True  → return μ (standard eval, DPO inference)
        deterministic=False → sample from N(μ, σ²) (BC training eval,
                              catches near-successes when μ is slightly off-target)
        """
        with torch.no_grad():
            mu, std = self.forward(obs)
            action  = mu if deterministic else Normal(mu, std).sample()
        return action.squeeze(0).cpu().numpy()

    def mean_std(self, obs: torch.Tensor) -> float:
        """Diagnostic: mean of per-dim std across a batch of observations."""
        with torch.no_grad():
            _, std = self.forward(obs)
        return std.mean().item()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0: Behavioral Cloning
# ══════════════════════════════════════════════════════════════════════════════

def build_bc_dataset(train_data: list, obs_normalizer: ObsNormalizer,
                     top_frac: float = BC_TOP_FRAC):
    """
    Flatten the TOP top_frac of chosen trajectories (by true_return) into
    (obs, action) pairs for BC.

    Why filter? "Chosen" only means better of two similar trajectories — many
    chosen trajs still have mediocre returns. Imitating the bottom half of
    "chosen" trajectories teaches the policy sub-optimal behavior and muddies
    the BC signal. Filtering to the top 30% gives cleaner, more decisive demos.
    """
    # Sort by return, keep top fraction
    sorted_pairs = sorted(train_data,
                          key=lambda p: p["chosen"]["true_return"],
                          reverse=True)
    cutoff = max(1, int(len(sorted_pairs) * top_frac))
    top_pairs = sorted_pairs[:cutoff]

    ret_threshold = top_pairs[-1]["chosen"]["true_return"]
    all_returns   = [p["chosen"]["true_return"] for p in top_pairs]
    print(f"[BC] Filtering to top {top_frac:.0%} of chosen trajs "
          f"({cutoff}/{len(train_data)})  "
          f"return threshold: {ret_threshold:.2f}  "
          f"mean: {np.mean(all_returns):.2f}")

    all_obs, all_acts = [], []
    for pair in top_pairs:
        obs_norm = obs_normalizer.normalize(
            pair["chosen"]["observations"].astype(np.float32))
        all_obs.append(obs_norm)
        all_acts.append(pair["chosen"]["actions"].astype(np.float32))

    all_obs  = np.concatenate(all_obs,  axis=0)
    all_acts = np.concatenate(all_acts, axis=0)
    print(f"[BC] Dataset: {len(all_obs):,} (obs, action) pairs")
    return all_obs, all_acts


def _bc_env_eval(policy: ContinuousDPOPolicy, obs_normalizer: ObsNormalizer,
                 n_episodes: int = BC_EVAL_EPISODES,
                 deterministic: bool = False) -> tuple:
    """
    Quick in-environment eval during BC training.
    Uses stochastic actions by default — with NLL-trained std the policy will
    naturally concentrate probability mass on good actions, and stochastic eval
    catches near-successes that the mean action alone would miss.
    Returns (mean_return, success_rate).
    """
    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks
    policy.eval()

    returns, successes = [], []
    with torch.no_grad():
        for i in range(n_episodes):
            task = tasks[i % len(tasks)]
            env.set_task(task)
            obs, _ = env.reset()
            ep_ret, success = 0.0, False
            for _ in range(MAX_EPISODE_STEPS):
                obs_norm = obs_normalizer.normalize(obs.astype(np.float32))
                obs_t    = torch.FloatTensor(obs_norm).unsqueeze(0).to(DEVICE)
                action   = policy.get_action(obs_t, deterministic=deterministic)
                action   = np.clip(action, env.action_space.low,
                                   env.action_space.high)
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


def pretrain_bc(policy: ContinuousDPOPolicy, train_data: list,
                obs_normalizer: ObsNormalizer,
                epochs: int = BC_EPOCHS, lr: float = BC_LR,
                batch_size: int = BC_BATCH_SIZE) -> ContinuousDPOPolicy:
    """
    Stage 0 — Supervised Behavioral Cloning on top chosen trajectories.

    Three key improvements over naive BC:

    1. NLL loss instead of MSE.
       Loss = -E[log π_θ(a|s)] = -E[Σ_d log N(a_d; μ_d, σ_d)]
       MSE only trains μ and ignores log_std entirely, leaving std≈1.0
       throughout training. The policy learns the right mean but with huge
       variance, so actions are imprecise. NLL jointly trains μ and σ,
       causing σ to shrink toward the actual behavioral variance in the data.
       A policy with low σ produces committed, precise reaching motions
       instead of diffuse, stop-short behavior.

    2. Top-K trajectory filtering.
       Only imitate the top BC_TOP_FRAC of chosen trajectories by return.

    3. Periodic environment evaluation with save-by-success-rate.
       Loss is not a reliable proxy for task success. We evaluate in the
       real environment every BC_EVAL_EVERY epochs and save the checkpoint
       with the highest success rate rather than the lowest NLL.
    """
    all_obs, all_acts = build_bc_dataset(train_data, obs_normalizer)
    obs_t  = torch.FloatTensor(all_obs).to(DEVICE)
    acts_t = torch.FloatTensor(all_acts).to(DEVICE)
    N = len(obs_t)

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    # Warmup for 5 epochs then cosine decay — prevents NLL from diverging early
    # when log_std is still at 0 (std=1) and gradients are large
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n{'=' * 60}")
    print(f"  Stage 0: Behavioral Cloning  ({N:,} transitions, {epochs} epochs)")
    print(f"  Loss: NLL  |  Batch: {batch_size}  |  LR: {lr}")
    print(f"  Top frac: {BC_TOP_FRAC:.0%}  |  Eval every: {BC_EVAL_EVERY} epochs")
    print(f"  Saving by: success rate (stochastic eval)")
    print(f"{'=' * 60}")

    best_success = -1.0
    os.makedirs(os.path.dirname(BC_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        policy.train()
        perm = torch.randperm(N)
        epoch_nll = 0.0
        n_batches  = 0

        # Linearly anneal std floor: 0.30 → 0.05 over training.
        # This prevents std from collapsing before μ has converged —
        # the critical failure mode where NLL greedily shrinks σ in epoch 1
        # and then μ can never escape the resulting near-deterministic regime.
        frac     = epoch / max(1, epochs - 1)
        std_min  = BC_STD_MIN_START + frac * (BC_STD_MIN_END - BC_STD_MIN_START)

        for start in range(0, N, batch_size):
            idx    = perm[start: start + batch_size]
            obs_b  = obs_t[idx]
            acts_b = acts_t[idx]

            mu, std = policy.forward(obs_b)
            # Apply floor — keeps variance high enough for μ to stay trainable.
            # clamp(min=std_min) only during BC; DPO uses raw forward() unchanged.
            std_floored = std.clamp(min=std_min)
            nll = -Normal(mu, std_floored).log_prob(acts_b).sum(dim=-1).mean()

            optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_nll  += nll.item()
            n_batches  += 1

        scheduler.step()

        # ── Periodic environment eval ─────────────────────────────────────────
        if (epoch + 1) % BC_EVAL_EVERY == 0 or epoch == 0:
            # Track BOTH modes — a big gap (stochastic succeeds, deterministic fails)
            # means μ is off-target and the policy relies on noise to stumble in
            ret_stoch,  succ_stoch  = _bc_env_eval(policy, obs_normalizer,
                                                    deterministic=False)
            ret_det,    succ_det    = _bc_env_eval(policy, obs_normalizer,
                                                    deterministic=True)

            # Mean std across a sample of training obs (state-dependent — varies)
            sample_obs = obs_t[:512]
            current_std = policy.mean_std(sample_obs)

            improved = succ_stoch > best_success
            print(f"  BC Epoch {epoch + 1:>3}/{epochs}  "
                  f"| NLL: {epoch_nll / n_batches:.4f}  "
                  f"| floor: {std_min:.3f}  actual_std: {current_std:.3f}  "
                  f"| ret: {ret_stoch:.1f}/{ret_det:.1f} (s/d)  "
                  f"| success: {succ_stoch:.2%}/{succ_det:.2%} (s/d)"
                  + (" ← best" if improved else ""))

            if improved:
                best_success = succ_stoch
                torch.save(policy.state_dict(), BC_SAVE_PATH)

    # If we never got any success (shouldn't happen on reach-v3 but just in case),
    # fall back to saving the final checkpoint
    if best_success < 0:
        torch.save(policy.state_dict(), BC_SAVE_PATH)

    print(f"\n  Best BC checkpoint: success rate {best_success:.2%} "
          f"→ {BC_SAVE_PATH}\n")
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
        margin: float = 0.0,   # optional margin-weighted loss
) -> tuple:
    log_pi_w  = policy.get_trajectory_log_prob(obs_w, act_w)
    log_pi_l  = policy.get_trajectory_log_prob(obs_l, act_l)

    with torch.no_grad():
        log_ref_w = ref_policy.get_trajectory_log_prob(obs_w, act_w)
        log_ref_l = ref_policy.get_trajectory_log_prob(obs_l, act_l)

    implicit_rw = beta * (log_pi_w - log_ref_w)
    implicit_rl = beta * (log_pi_l - log_ref_l)

    # Standard DPO loss
    loss = -F.logsigmoid(implicit_rw - implicit_rl)

    with torch.no_grad():
        delta  = (implicit_rw - implicit_rl).item()
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
    val_data   = dataset_dict["val"]

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

    best_val_acc  = -1.0
    best_success  = -1.0
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

                batch_loss   += loss
                batch_acc    += metrics["accuracy"]
                batch_margin += metrics["margin"]
                batch_kl     += metrics["kl_chosen"]

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss   += batch_loss.item()
            epoch_acc    += batch_acc    / len(batch)
            epoch_margin += batch_margin / len(batch)
            epoch_kl     += batch_kl     / len(batch)
            n_batches    += 1

        scheduler.step()

        # ── Preference validation (every epoch — cheap, no env) ───────────────
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
                val_acc  += metrics["accuracy"]
        n_v = len(val_data)

        # ── Environment eval (every DPO_EVAL_EVERY epochs — costs env rollouts) ─
        if (epoch + 1) % DPO_EVAL_EVERY == 0 or epoch == epochs - 1:
            ret_stoch, succ_stoch = _bc_env_eval(policy, obs_normalizer,
                                                  n_episodes=DPO_EVAL_EPISODES,
                                                  deterministic=False)
            ret_det,   succ_det   = _bc_env_eval(policy, obs_normalizer,
                                                  n_episodes=DPO_EVAL_EPISODES,
                                                  deterministic=True)
            improved = succ_stoch > best_success
            if improved:
                best_success = succ_stoch
                best_val_acc = val_acc / n_v
                torch.save(policy.state_dict(), DPO_SAVE_PATH)

            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| loss: {epoch_loss / n_batches:.4f}  "
                  f"pref_acc: {epoch_acc / n_batches:.2%}  "
                  f"KL: {epoch_kl / n_batches:.3f}  "
                  f"| val_acc: {val_acc / n_v:.2%}  "
                  f"| ret: {ret_stoch:.1f}/{ret_det:.1f} (s/d)  "
                  f"| success: {succ_stoch:.2%}/{succ_det:.2%} (s/d)"
                  + (" ✓" if improved else ""))
        else:
            # Non-eval epochs: print preference metrics only (no env cost)
            if val_acc / n_v > best_val_acc:
                best_val_acc = val_acc / n_v
            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| loss: {epoch_loss / n_batches:.4f}  "
                  f"pref_acc: {epoch_acc / n_batches:.2%}  "
                  f"KL: {epoch_kl / n_batches:.3f}  "
                  f"| val_acc: {val_acc / n_v:.2%}")

    print(f"\n  Best DPO policy saved → {DPO_SAVE_PATH}  "
          f"(success: {best_success:.2%}  val_acc: {best_val_acc:.2%})\n")
    policy.load_state_dict(torch.load(DPO_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_dpo_policy(policy: ContinuousDPOPolicy, n_episodes: int = 50,
                        deterministic: bool = True) -> dict:
    """
    Evaluate policy in MetaWorld.
    deterministic=True  → use μ only (clean comparison, standard eval)
    deterministic=False → sample from N(μ,σ²) (catches near-successes)
    Reports both so you can see the gap.
    """
    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks

    obs_normalizer = ObsNormalizer()
    obs_normalizer.load(OBS_NORM_PATH)
    policy.eval()

    returns, successes = [], []
    with torch.no_grad():
        for i in range(n_episodes):
            task = tasks[i % len(tasks)]
            env.set_task(task)
            obs, _ = env.reset()
            ep_ret, success = 0.0, False

            for _ in range(MAX_EPISODE_STEPS):
                obs_norm = obs_normalizer.normalize(obs.astype(np.float32))
                obs_t    = torch.FloatTensor(obs_norm).unsqueeze(0).to(DEVICE)
                action   = policy.get_action(obs_t, deterministic=deterministic)
                action   = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_ret += reward
                if info.get("success", False):
                    success = True
                if terminated or truncated:
                    break

            returns.append(ep_ret)
            successes.append(float(success))

    mode = "deterministic" if deterministic else "stochastic"
    results = {
        "mean_return":  float(np.mean(returns)),
        "std_return":   float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
    }
    print(f"[Eval/{mode}]  Return: {results['mean_return']:.2f} ± "
          f"{results['std_return']:.2f}  "
          f"|  Success rate: {results['success_rate']:.2%}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DPO for continuous control.")
    parser.add_argument("--data",      default=PREF_DATASET_PATH)
    parser.add_argument("--epochs",    type=int,   default=DPO_EPOCHS)
    parser.add_argument("--bc_epochs", type=int,   default=BC_EPOCHS,
                        help="Epochs for BC pretraining stage")
    parser.add_argument("--lr",        type=float, default=DPO_LR)
    parser.add_argument("--beta",      type=float, default=DPO_BETA,
                        help="KL regularisation strength (0.05–0.2 for robotics)")
    parser.add_argument("--skip_bc",   action="store_true",
                        help="Skip BC pretraining and load existing BC checkpoint")
    parser.add_argument("--eval",      action="store_true",
                        help="Run eval after training")
    args = parser.parse_args()

    policy = run_dpo_alignment(
        data_path  = args.data,
        epochs     = args.epochs,
        lr         = args.lr,
        beta       = args.beta,
        bc_epochs  = args.bc_epochs,
        skip_bc    = args.skip_bc,
    )

    if args.eval:
        print("\n--- BC policy performance (before DPO) ---")
        bc_policy = ContinuousDPOPolicy().to(DEVICE)
        bc_policy.load_state_dict(torch.load(BC_SAVE_PATH, map_location=DEVICE))
        evaluate_dpo_policy(bc_policy, n_episodes=50, deterministic=False)
        evaluate_dpo_policy(bc_policy, n_episodes=50, deterministic=True)

        print("\n--- DPO policy performance (after DPO) ---")
        evaluate_dpo_policy(policy, n_episodes=50, deterministic=False)
        evaluate_dpo_policy(policy, n_episodes=50, deterministic=True)