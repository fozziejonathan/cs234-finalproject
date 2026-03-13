"""
train_rlhf.py
-------------
RLHF fine-tuning of a pre-trained BC policy.

  Stage 1 — Bradley-Terry Reward Model
  Stage 2 — Reward-Weighted Regression (RWR) policy fine-tuning

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Why RLHF fine-tunes instead of trains from scratch
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The KL-regularized RL objective is:

    max_π  E_π[R(τ)]  −  β · KL( π ‖ π_ref )

Peters & Schaal (2007) show this has a closed-form optimal solution:

    π*(a|s)  ∝  π_ref(a|s) · exp( r(s,a) / β )

When π_ref = π_BC, the BC policy is the reference, and the optimum is
just a re-weighted version of BC. You do not need to search the entire
policy space from scratch — you only need to nudge π_BC toward
trajectories that receive high reward from the learned reward model.

RWR implements this by treating the weights exp(R(τ)/β) as fixed
scalars and solving the resulting weighted maximum-likelihood problem:

    θ* = argmax_θ  Σ_τ  exp(R(τ)/β) · log π_θ(τ)

This is pure supervised learning (no RL loops, no value functions,
no on-policy rollouts) with the BC policy as the warm start.

The β temperature controls how aggressively the policy departs from BC:
  β → ∞ : all weights → 1 (equivalent to plain BC, no change)
  β → 0 : winner-take-all (only the single best trajectory survives)
A finite β keeps the policy close to BC while preferring high-reward
trajectories, which is exactly what you want for fine-tuning.

References
----------
Christiano et al. (2017)  — Deep RL from Human Preferences (Bradley-Terry RM)
Peters & Schaal (2007)    — Reinforcement Learning by Reward-Weighted Regression
Peng et al. (2019) / AWR  — Advantage-Weighted Regression (extends RWR offline)
Dong et al. (2023)        — Aligning LMs with Offline Learning from Human Feedback
                            (arXiv 2308.12050) — reward normalization before weighting

Usage
-----
    # Train both RM and RWR from scratch:
    python train_rlhf.py --data data/preferences_bin-picking-v3_20seeds_10000.pkl

    # Skip RM training (reuse saved checkpoint):
    python train_rlhf.py --data data/preferences_... --skip_rm

Outputs
-------
    checkpoints/rm_{ENV_NAME}.pt          — frozen reward model
    checkpoints/rlhf_policy_{ENV_NAME}.pt — best RWR-fine-tuned policy
"""

import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from config import (
    ENV_NAME, OBS_DIM, ACT_DIM, HIDDEN_DIM,
)
from train_bc import GaussianPolicy, ObsNormalizer, env_eval, DEVICE

# ── Paths ──────────────────────────────────────────────────────────────────────
BC_CKPT       = f"checkpoints/bc_policy_{ENV_NAME}.pt"
OBS_NORM_PATH = f"checkpoints/obs_normalizer_{ENV_NAME}.npz"
RM_SAVE_PATH  = f"checkpoints/rm_{ENV_NAME}.pt"
RWR_SAVE_PATH = f"checkpoints/rlhf_policy_{ENV_NAME}.pt"

# ── Hyperparameters ────────────────────────────────────────────────────────────
RM_EPOCHS  = 20
RM_LR      = 3e-4
RM_BATCH   = 32    # preference pairs per gradient step

RWR_EPOCHS = 20
RWR_LR     = 1e-4
RWR_BATCH  = 32    # trajectories per gradient step
RWR_BETA   = 1.0   # temperature β: lower = more aggressive re-weighting

EVAL_EVERY = 1
EVAL_EPS   = 100


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Bradley-Terry Reward Model
# ══════════════════════════════════════════════════════════════════════════════

class RewardModel(nn.Module):
    """
    r_φ(s, a) → scalar reward.

    Architecture: two-hidden-layer MLP, same hidden dim as the BC policy.
    Input : (obs, action) concatenated — shape (T, obs_dim + act_dim)
    Output: scalar — mean of per-step rewards over the trajectory

    Mean-pooling over timesteps (not sum) prevents longer trajectories
    from receiving artificially high/low scores relative to short ones,
    which matters in MetaWorld where episodes terminate early on success.

    Training objective (Bradley-Terry model, Christiano et al. 2017):
        L = -log σ( R(τ_chosen) − R(τ_rejected) )
    where R(τ) = (1/T) Σ_t r_φ(s_t, a_t).
    """

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """
        obs : (T, obs_dim)   act : (T, act_dim)
        returns scalar — mean reward across the T timesteps.
        """
        x = torch.cat([obs, act], dim=-1)       # (T, obs_dim + act_dim)
        return self.net(x).squeeze(-1).mean()   # scalar


def _to_tensors(pair_side, obs_norm):
    """Normalize obs and move obs/acts to DEVICE for one trajectory."""
    obs_n = obs_norm.normalize(pair_side["observations"].astype(np.float32))
    obs_t = torch.FloatTensor(obs_n).to(DEVICE)
    act_t = torch.FloatTensor(pair_side["actions"].astype(np.float32)).to(DEVICE)
    return obs_t, act_t


def train_reward_model(rm, train_data, val_data, obs_norm,
                       epochs=RM_EPOCHS, lr=RM_LR, batch_size=RM_BATCH):
    """
    Train the reward model with the Bradley-Terry objective.

    Only hard-label pairs (label=1.0) are used. Soft-label pairs
    (label=0.5, both trajectories failed) are discarded because they
    carry no preference signal — using them would add noise to the
    gradient and could push the RM to assign arbitrary ordering to
    pairs where neither trajectory is preferred.

    Reference: Christiano et al. (2017), Section 2.2.
    """
    train_hard = [p for p in train_data if p["label"] == 1.0]
    val_hard   = [p for p in val_data   if p["label"] == 1.0]

    optimizer = optim.Adam(rm.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'=' * 60}")
    print(f"  Stage 1 — Bradley-Terry Reward Model")
    print(f"  Hard pairs  : {len(train_hard)} train  |  {len(val_hard)} val")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    print(f"  Loss  : -log σ( R(τ_chosen) - R(τ_rejected) )")
    print(f"  Pool  : mean over timesteps (not sum)")
    print(f"{'=' * 60}\n")

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(RM_SAVE_PATH), exist_ok=True)

    for epoch in range(epochs):
        rm.train()
        np.random.shuffle(train_hard)
        epoch_loss = epoch_acc = 0.0
        n_batches = 0

        for start in range(0, len(train_hard), batch_size):
            batch = train_hard[start: start + batch_size]
            batch_loss = torch.tensor(0.0, device=DEVICE)
            batch_acc  = 0.0

            for pair in batch:
                obs_w, act_w = _to_tensors(pair["chosen"],   obs_norm)
                obs_l, act_l = _to_tensors(pair["rejected"], obs_norm)

                r_w = rm(obs_w, act_w)   # scalar
                r_l = rm(obs_l, act_l)   # scalar

                # Bradley-Terry: preferred trajectory should have higher reward
                batch_loss = batch_loss + (-F.logsigmoid(r_w - r_l))
                batch_acc  += float((r_w > r_l).item())

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_acc  += batch_acc / len(batch)
            n_batches  += 1

        scheduler.step()

        # Validation accuracy — no env rollouts needed, just RM forward passes
        rm.eval()
        val_acc = 0.0
        with torch.no_grad():
            for pair in val_hard:
                obs_w, act_w = _to_tensors(pair["chosen"],   obs_norm)
                obs_l, act_l = _to_tensors(pair["rejected"], obs_norm)
                val_acc += float((rm(obs_w, act_w) > rm(obs_l, act_l)).item())
        val_acc /= len(val_hard)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            torch.save(rm.state_dict(), RM_SAVE_PATH)

        print(f"  Epoch {epoch + 1:>3}/{epochs}  "
              f"| train loss: {epoch_loss / n_batches:.4f}  "
              f"acc: {epoch_acc / n_batches:.2%}  "
              f"| val acc: {val_acc:.2%}"
              + ("  ✓" if improved else ""))

    print(f"\n  Best RM val acc: {best_val_acc:.2%} → {RM_SAVE_PATH}")

    # Reload best checkpoint and freeze
    rm.load_state_dict(torch.load(RM_SAVE_PATH, map_location=DEVICE))
    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False
    print(f"  Reward model frozen — no further gradient updates.\n")
    return rm


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: Reward-Weighted Regression (RWR)
# ══════════════════════════════════════════════════════════════════════════════

def _score_all_trajectories(rm, train_data, obs_norm):
    """
    Score every trajectory that appears in the hard-label training pairs
    using the frozen reward model, and compute normalized RWR weights.

    Both chosen and rejected sides of each pair are scored. This is correct:
    the RWR update should see the full distribution of behaviors that the
    BC policy produced, not just the winners, so that low-reward trajectories
    are down-weighted rather than silently ignored.

    Reward normalization (subtract mean, divide std) is applied before
    computing exp weights. This decouples the weight scale from the reward
    model's absolute output range so that the temperature β has a consistent
    interpretation regardless of the RM's scale.
    Reference: Dong et al. (2023), arXiv 2308.12050, Section 3.

    Returns
    -------
    list of dicts, each containing:
        obs_n   : (T, obs_dim) float32 ndarray, already normalized
        acts    : (T, act_dim) float32 ndarray
        reward  : float, raw RM score
        reward_norm : float, normalized RM score
        success : bool
    """
    hard_pairs = [p for p in train_data if p["label"] == 1.0]
    trajs = []

    rm.eval()
    with torch.no_grad():
        for pair in hard_pairs:
            for side in ("chosen", "rejected"):
                t = pair[side]
                obs_n = obs_norm.normalize(t["observations"].astype(np.float32))
                obs_t = torch.FloatTensor(obs_n).to(DEVICE)
                act_t = torch.FloatTensor(t["actions"].astype(np.float32)).to(DEVICE)
                r = rm(obs_t, act_t).item()   # scalar
                trajs.append({
                    "obs_n":   obs_n,
                    "acts":    t["actions"].astype(np.float32),
                    "reward":  r,
                    "success": t["success"],
                })

    raw = np.array([t["reward"] for t in trajs])
    r_mean, r_std = raw.mean(), max(raw.std(), 1e-8)
    for t in trajs:
        t["reward_norm"] = (t["reward"] - r_mean) / r_std

    print(f"[RWR] Scored {len(trajs)} trajectories "
          f"({sum(t['success'] for t in trajs)} successful)")
    print(f"[RWR] Raw reward   — mean: {r_mean:.4f}  std: {r_std:.4f}  "
          f"range: [{raw.min():.4f}, {raw.max():.4f}]")
    norm = np.array([t["reward_norm"] for t in trajs])
    print(f"[RWR] Norm reward  — mean: {norm.mean():.4f}  std: {norm.std():.4f}  "
          f"range: [{norm.min():.4f}, {norm.max():.4f}]")
    return trajs


def train_rwr(policy, rm, train_data, val_data, obs_norm,
              epochs=RWR_EPOCHS, lr=RWR_LR, beta=RWR_BETA):
    """
    Fine-tune the BC policy with Reward-Weighted Regression (RWR).

    Algorithm (Peters & Schaal, 2007)
    ----------------------------------
    Given fixed weights  w_i = exp( R_norm(τ_i) / β ):

        θ* = argmax_θ  Σ_i  w_i · log π_θ(τ_i)

    This is exactly the weighted MLE problem, which is standard
    supervised learning with per-sample importance weights.  The RM
    is frozen — there are no on-policy rollouts, no value functions,
    and no RL update rules. The entire fine-tuning stage is a
    weighted regression over the existing trajectory dataset.

    Connection to KL-regularized RL (why this is principled fine-tuning)
    ----------------------------------------------------------------------
    The above objective is the exact solution to:

        max_π  E_π[R(τ)]  −  β · KL( π ‖ π_BC )

    (Peters & Schaal 2007; Peng et al. 2019).  Setting π_ref = π_BC means:
      (a) We never stray far from the behavior the BC policy already learned.
      (b) The only parts of the policy that change are those where the RM
          provides a clear signal (high vs. low reward trajectories differ).
      (c) The warm-start from BC weights means the weighted MLE converges
          quickly — we are already in a good region of parameter space.

    This is fundamentally different from training RLHF from scratch, where
    π_ref would be a random policy and the KL penalty would have nothing
    useful to anchor to.
    """
    trajs = _score_all_trajectories(rm, train_data, obs_norm)

    # Compute and normalize RWR weights
    weights = np.array([np.exp(t["reward_norm"] / beta) for t in trajs],
                       dtype=np.float64)
    weights /= weights.sum()   # probability distribution over trajectories

    eff_n = 1.0 / (weights ** 2).sum()
    print(f"[RWR] β = {beta}")
    print(f"[RWR] Weight stats — max: {weights.max():.5f}  "
          f"min: {weights.min():.7f}  "
          f"effective N: {eff_n:.1f} / {len(trajs)}")

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'=' * 60}")
    print(f"  Stage 2 — Reward-Weighted Regression (RWR)")
    print(f"  Trajectories : {len(trajs)}")
    print(f"  β (temperature) : {beta}")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch: {RWR_BATCH}")
    print(f"  Init   : BC policy checkpoint")
    print(f"  Ckpt   : best deterministic success rate")
    print(f"{'=' * 60}\n")

    best_success = -1.0
    best_return  = -float("inf")
    os.makedirs(os.path.dirname(RWR_SAVE_PATH), exist_ok=True)
    indices = np.arange(len(trajs))

    for epoch in range(epochs):
        policy.train()
        np.random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(trajs), RWR_BATCH):
            batch_idx = indices[start: start + RWR_BATCH]
            batch_loss  = torch.tensor(0.0, device=DEVICE)
            batch_w_sum = 0.0

            for i in batch_idx:
                t   = trajs[i]
                w_i = float(weights[i])

                obs_t = torch.FloatTensor(t["obs_n"]).to(DEVICE)
                act_t = torch.FloatTensor(t["acts"]).to(DEVICE)

                # log π_θ(τ) = (1/T) Σ_t log π_θ(a_t | s_t)
                # Mean over steps so trajectory length doesn't bias the loss.
                mu, std = policy.forward(obs_t)
                log_pi_tau = Normal(mu, std).log_prob(act_t).sum(dim=-1).mean()

                batch_loss  = batch_loss + (-w_i * log_pi_tau)
                batch_w_sum += w_i

            # Normalize by the total weight mass in this batch so that a
            # batch dominated by one high-weight trajectory doesn't produce
            # an outsized gradient step.
            if batch_w_sum > 0:
                batch_loss = batch_loss / batch_w_sum

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            epoch_loss += batch_loss.item()
            n_batches  += 1

        scheduler.step()

        if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
            ret_d, succ_d = env_eval(policy, obs_norm, EVAL_EPS, deterministic=True)
            # Primary criterion: success rate. Tiebreaker: mean return.
            # Among equally successful policies, higher return means the task
            # is completed faster / more efficiently (Yu et al. 2021, MetaWorld).
            improved = (succ_d > best_success) or \
                       (succ_d == best_success and ret_d > best_return)
            if improved:
                best_success = succ_d
                best_return  = ret_d
                torch.save(policy.state_dict(), RWR_SAVE_PATH)
            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| loss: {epoch_loss / n_batches:.4f}  "
                  f"| success: {succ_d:.2%}  return: {ret_d:.1f}"
                  + ("  ✓" if improved else ""))
        else:
            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| loss: {epoch_loss / n_batches:.4f}")

    print(f"\n  Best RWR checkpoint: {best_success:.2%} success → {RWR_SAVE_PATH}")
    policy.load_state_dict(torch.load(RWR_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,
                        help="Preference dataset pkl from label_data.py")
    parser.add_argument("--bc",         default=BC_CKPT,
                        help="BC policy checkpoint from train_bc.py")
    parser.add_argument("--obs_norm",   default=OBS_NORM_PATH,
                        help="Obs normalizer npz from train_bc.py")
    parser.add_argument("--rm_epochs",  type=int,   default=RM_EPOCHS)
    parser.add_argument("--rwr_epochs", type=int,   default=RWR_EPOCHS)
    parser.add_argument("--rm_lr",      type=float, default=RM_LR)
    parser.add_argument("--rwr_lr",     type=float, default=RWR_LR)
    parser.add_argument("--beta",       type=float, default=RWR_BETA,
                        help="RWR temperature β. Lower = more aggressive "
                             "re-weighting away from BC.")
    parser.add_argument("--skip_rm",    action="store_true",
                        help="Load existing RM checkpoint instead of retraining")
    args = parser.parse_args()

    # ── Load preference dataset ───────────────────────────────────────────────
    with open(args.data, "rb") as f:
        ds = pickle.load(f)
    train_data, val_data = ds["train"], ds["val"]
    print(f"[INFO] Loaded {len(train_data)} train / {len(val_data)} val pairs "
          f"from {args.data}")

    # ── Load obs normalizer (fitted on BC data, reused here) ──────────────────
    obs_norm = ObsNormalizer()
    obs_norm.load(args.obs_norm)

    # ── Stage 1: Bradley-Terry Reward Model ───────────────────────────────────
    rm = RewardModel().to(DEVICE)

    if args.skip_rm and os.path.exists(RM_SAVE_PATH):
        print(f"[INFO] Loading existing RM from {RM_SAVE_PATH}")
        rm.load_state_dict(torch.load(RM_SAVE_PATH, map_location=DEVICE))
        rm.eval()
        for p in rm.parameters():
            p.requires_grad = False
        print(f"[INFO] Reward model frozen.")
    else:
        rm = train_reward_model(rm, train_data, val_data, obs_norm,
                                epochs=args.rm_epochs, lr=args.rm_lr)

    # ── Stage 2: RWR fine-tuning of BC policy ─────────────────────────────────
    policy = GaussianPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(args.bc, map_location=DEVICE))
    print(f"[INFO] Loaded BC policy from {args.bc}")

    policy = train_rwr(policy, rm, train_data, val_data, obs_norm,
                       epochs=args.rwr_epochs, lr=args.rwr_lr, beta=args.beta)

    # ── Final evaluation ──────────────────────────────────────────────────────
    print(f"\n--- RLHF (RWR) final evaluation ---")
    ret_d, succ_d = env_eval(policy, obs_norm, n_episodes=50, deterministic=True)
    print(f"  Deterministic : return {ret_d:.2f}  success {succ_d:.2%}")
    print(f"\n[INFO] Saved RLHF policy → {RWR_SAVE_PATH}")