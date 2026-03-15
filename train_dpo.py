"""
train_dpo.py
------------
DPO fine-tuning of a BC policy.

Fine-tunes against a frozen copy of the BC policy as reference.
Hard-label pairs only (label=1.0, success vs. failure).

Loss: -log σ( β*(log π(τ_w) - log π_ref(τ_w)) - β*(log π(τ_l) - log π_ref(τ_l)) )

Outputs:
    checkpoints/dpo_policy_{ENV_NAME}.pt   (or dpop_policy with --dpop)
"""

import argparse
import copy
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import (
    ENV_NAME, MAX_EPISODE_STEPS,
    DPO_LR, DPO_EPOCHS, DPO_BETA,
)
from train_bc import GaussianPolicy, ObsNormalizer, env_eval, DEVICE, EVAL_SEEDS, EVAL_EPS_PER_SEED

# ── Paths ──────────────────────────────────────────────────────────────────────
BC_SAVE_PATH  = f"checkpoints/bc_policy_{ENV_NAME}.pt"
OBS_NORM_PATH = f"checkpoints/obs_normalizer_{ENV_NAME}.npz"
DPO_SAVE_PATH = f"checkpoints/dpop_policy_{ENV_NAME}.pt"

# ── DPO hyperparameters ────────────────────────────────────────────────────────
DPO_BATCH      = 32
DPO_EVAL_EVERY = 1
DPO_EVAL_EPS   = len(EVAL_SEEDS) * EVAL_EPS_PER_SEED  # 3 seeds × 50 = 150

# Trajectory log-prob normalisation: sum over steps / nominal horizon
T_NORM = float(MAX_EPISODE_STEPS)


# ══════════════════════════════════════════════════════════════════════════════
# DPO Loss
# ══════════════════════════════════════════════════════════════════════════════

def dpo_loss(policy, ref_policy, obs_w, act_w, obs_l, act_l, beta, dpop_lambda=0.0):
    log_pi_w = policy.traj_log_prob(obs_w, act_w)
    log_pi_l = policy.traj_log_prob(obs_l, act_l)
    with torch.no_grad():
        log_ref_w = ref_policy.traj_log_prob(obs_w, act_w)
        log_ref_l = ref_policy.traj_log_prob(obs_l, act_l)

    rw   = beta * (log_pi_w - log_ref_w)
    rl   = beta * (log_pi_l - log_ref_l)
    loss = -F.logsigmoid(rw - rl)

    if dpop_lambda > 0.0:
        dpop_penalty = torch.clamp(log_ref_w - log_pi_w, min=0.0)
        loss = loss + dpop_lambda * dpop_penalty

    acc  = float((rw > rl).item())
    return loss, {"reward_w": rw.item(), "reward_l": rl.item(),
                  "margin": (rw - rl).item(), "accuracy": acc}

# ══════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(policy, ref_policy, bc_policy_init, batch, obs_norm, beta):
    policy.eval()
    stats = {"kl_div": 0.0, "log_pi_w": 0.0, "log_pi_l": 0.0,
             "rw": 0.0, "rl": 0.0, "margin": 0.0, "mu_drift": 0.0}
    n = 0
    with torch.no_grad():
        for pair in batch:
            obs_w = torch.FloatTensor(
                obs_norm.normalize(pair["chosen"]["observations"].astype(np.float32))
            ).to(DEVICE)
            act_w = torch.FloatTensor(pair["chosen"]["actions"].astype(np.float32)).to(DEVICE)
            obs_l = torch.FloatTensor(
                obs_norm.normalize(pair["rejected"]["observations"].astype(np.float32))
            ).to(DEVICE)
            act_l = torch.FloatTensor(pair["rejected"]["actions"].astype(np.float32)).to(DEVICE)

            lp_w  = policy.traj_log_prob(obs_w, act_w).item()
            lp_l  = policy.traj_log_prob(obs_l, act_l).item()
            lp_ref_w = ref_policy.traj_log_prob(obs_w, act_w).item()
            lp_ref_l = ref_policy.traj_log_prob(obs_l, act_l).item()

            rw = beta * (lp_w - lp_ref_w)
            rl = beta * (lp_l - lp_ref_l)

            mu_curr, _ = policy.forward(obs_w)
            mu_init, _ = bc_policy_init.forward(obs_w)
            drift = (mu_curr - mu_init).pow(2).mean().item()

            stats["kl_div"] += 0.5 * ((lp_w - lp_ref_w) + (lp_l - lp_ref_l))
            stats["log_pi_w"] += lp_w
            stats["log_pi_l"] += lp_l
            stats["rw"] += rw
            stats["rl"] += rl
            stats["margin"] += rw - rl
            stats["mu_drift"] += drift
            n += 1
        policy.train()
        return {k: v / n for k, v in stats.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_dpo(policy, ref_policy, train_data, val_data, obs_norm,
              epochs=DPO_EPOCHS, lr=DPO_LR, beta=DPO_BETA, dpop_lambda=0.0):
    # Only success-failure, success-success is just noise (Hejna and Sadigh)
    train_hard = [p for p in train_data
                  if p["label"] == 1.0
                  and p["chosen"]["success"]
                  and not p["rejected"]["success"]]
    val_hard   = [p for p in val_data   if p["label"] == 1.0]
    print(f"[DPO] Hard-label pairs — train: {len(train_hard)}/{len(train_data)}  "
          f"val: {len(val_hard)}/{len(val_data)}")

    policy.log_std.requires_grad = False
    optimizer = optim.Adam(
        [p for p in policy.parameters() if p.requires_grad], lr=lr
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"\n{'=' * 60}")
    print(f"  {'DPOP' if dpop_lambda > 0 else 'DPO'} Fine-tuning")
    print(f"  Pairs — train: {len(train_hard)}  val: {len(val_hard)}")
    print(f"  β={beta}  lr={lr}  epochs={epochs}  batch={DPO_BATCH}"
          + (f"  dpop_λ={dpop_lambda}" if dpop_lambda > 0 else ""))
    print(f"  Reference: frozen BC checkpoint")
    print(f"  Eval every  : {DPO_EVAL_EVERY} epochs  |  {DPO_EVAL_EPS} episodes")
    print(f"  Checkpoint  : best deterministic success rate")
    print(f"{'=' * 60}\n")

    best_success = -1.0
    os.makedirs(os.path.dirname(DPO_SAVE_PATH), exist_ok=True)

    # Frozen copy of BC at epoch 0 — measures cumulative μ drift
    bc_policy_init = copy.deepcopy(policy)
    bc_policy_init.eval()
    for p in bc_policy_init.parameters():
        p.requires_grad = False

    ret_d, succ_d = env_eval(policy, obs_norm, DPO_EVAL_EPS, deterministic=True)
    best_success = succ_d
    torch.save(policy.state_dict(), DPO_SAVE_PATH)

    diag0 = run_diagnostics(policy, ref_policy, bc_policy_init,
                            train_hard[:50], obs_norm, beta)
    print(f"  [DIAG] kl={diag0['kl_div']:.4f}  "
          f"log_π(w)={diag0['log_pi_w']:.3f}  "
          f"log_π(l)={diag0['log_pi_l']:.3f}  "
          f"rw={diag0['rw']:.4f}  rl={diag0['rl']:.4f}  "
          f"margin={diag0['margin']:.4f}  "
          f"μ_drift={diag0['mu_drift']:.6f}  "
          f"grad_norm=0.0000")

    policy.eval()
    val_acc0 = 0.0
    with torch.no_grad():
        for pair in val_hard:
            obs_w = torch.FloatTensor(
                obs_norm.normalize(pair["chosen"]["observations"].astype(np.float32))
            ).to(DEVICE)
            act_w = torch.FloatTensor(
                pair["chosen"]["actions"].astype(np.float32)
            ).to(DEVICE)
            obs_l = torch.FloatTensor(
                obs_norm.normalize(pair["rejected"]["observations"].astype(np.float32))
            ).to(DEVICE)
            act_l = torch.FloatTensor(
                pair["rejected"]["actions"].astype(np.float32)
            ).to(DEVICE)
            _, m = dpo_loss(policy, ref_policy, obs_w, act_w, obs_l, act_l, beta)
            val_acc0 += m["accuracy"]
    val_acc0 /= len(val_hard)

    print(f"  Epoch   0/{epochs}  "
          f"| loss: n/a        "
          f"pref_acc:   n/a  "
          f"val_acc: {val_acc0:.2%}  "
          f"| success: {succ_d:.2%}  return: {ret_d:.1f}")

    for epoch in range(epochs):
        policy.train()
        np.random.shuffle(train_hard)

        epoch_loss = epoch_acc = 0.0
        n_batches = 0

        for start in range(0, len(train_hard), DPO_BATCH):
            batch      = train_hard[start:start + DPO_BATCH]
            batch_loss = 0.0
            batch_acc  = 0.0

            for pair in batch:
                obs_w = torch.FloatTensor(
                    obs_norm.normalize(pair["chosen"]["observations"].astype(np.float32))
                ).to(DEVICE)
                act_w = torch.FloatTensor(
                    pair["chosen"]["actions"].astype(np.float32)
                ).to(DEVICE)
                obs_l = torch.FloatTensor(
                    obs_norm.normalize(pair["rejected"]["observations"].astype(np.float32))
                ).to(DEVICE)
                act_l = torch.FloatTensor(
                    pair["rejected"]["actions"].astype(np.float32)
                ).to(DEVICE)

                loss, m = dpo_loss(policy, ref_policy, obs_w, act_w, obs_l, act_l, beta, dpop_lambda)
                batch_loss += loss
                batch_acc  += m["accuracy"]

            batch_loss = batch_loss / len(batch)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            total_grad_norm = sum(
                p.grad.norm().item() ** 2
                for p in policy.parameters()
                if p.grad is not None
            ) ** 0.5
            optimizer.step()

            epoch_loss += batch_loss.item()
            epoch_acc  += batch_acc / len(batch)
            n_batches  += 1

        scheduler.step()

        diag = run_diagnostics(policy, ref_policy, bc_policy_init,
                               train_hard[:50], obs_norm, beta)
        print(f"  [DIAG] kl={diag['kl_div']:.4f}  "
              f"log_π(w)={diag['log_pi_w']:.3f}  "
              f"log_π(l)={diag['log_pi_l']:.3f}  "
              f"rw={diag['rw']:.4f}  rl={diag['rl']:.4f}  "
              f"margin={diag['margin']:.4f}  "
              f"μ_drift={diag['mu_drift']:.6f}  "
              f"grad_norm={total_grad_norm:.4f}")

        # Val preference accuracy (every epoch, no env rollouts)

        # Val preference accuracy (every epoch, no env rollouts)
        policy.eval()
        val_acc = 0.0
        with torch.no_grad():
            for pair in val_hard:
                obs_w = torch.FloatTensor(
                    obs_norm.normalize(pair["chosen"]["observations"].astype(np.float32))
                ).to(DEVICE)
                act_w = torch.FloatTensor(
                    pair["chosen"]["actions"].astype(np.float32)
                ).to(DEVICE)
                obs_l = torch.FloatTensor(
                    obs_norm.normalize(pair["rejected"]["observations"].astype(np.float32))
                ).to(DEVICE)
                act_l = torch.FloatTensor(
                    pair["rejected"]["actions"].astype(np.float32)
                ).to(DEVICE)
                _, m = dpo_loss(policy, ref_policy, obs_w, act_w, obs_l, act_l, beta)
                val_acc += m["accuracy"]
        val_acc /= len(val_hard)

        # Env eval (every DPO_EVAL_EVERY epochs)
        if (epoch + 1) % DPO_EVAL_EVERY == 0 or epoch == epochs - 1:
            ret_d, succ_d = env_eval(policy, obs_norm, deterministic=True)
            improved = succ_d > best_success
            if improved:
                best_success = succ_d
                torch.save(policy.state_dict(), DPO_SAVE_PATH)
            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| loss: {epoch_loss / n_batches:.4f}  "
                  f"pref_acc: {epoch_acc / n_batches:.2%}  "
                  f"val_acc: {val_acc:.2%}  "
                  f"| success: {succ_d:.2%}  return: {ret_d:.1f}"
                  + ("  ✓" if improved else ""))
        else:
            print(f"  Epoch {epoch + 1:>3}/{epochs}  "
                  f"| loss: {epoch_loss / n_batches:.4f}  "
                  f"pref_acc: {epoch_acc / n_batches:.2%}  "
                  f"val_acc: {val_acc:.2%}")

    print(f"\n  Best DPO checkpoint: {best_success:.2%} success → {DPO_SAVE_PATH}")
    policy.load_state_dict(torch.load(DPO_SAVE_PATH, map_location=DEVICE))
    return policy


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     required=True,
                        help="Preference dataset pkl from label_data.py")
    parser.add_argument("--bc",       default=BC_SAVE_PATH,
                        help="BC policy checkpoint from train_bc.py")
    parser.add_argument("--obs_norm", default=OBS_NORM_PATH,
                        help="Obs normalizer npz from train_bc.py")
    parser.add_argument("--epochs",   type=int,   default=DPO_EPOCHS)
    parser.add_argument("--lr",       type=float, default=DPO_LR)
    parser.add_argument("--beta",        type=float, default=DPO_BETA)
    parser.add_argument("--dpop",        action="store_true",
                        help="Use DPOP loss (adds penalty to keep chosen log-prob >= ref)")
    parser.add_argument("--dpop_lambda", type=float, default=0.5,
                        help="DPOP penalty weight (only used if --dpop is set, default: 0.5)")
    args = parser.parse_args()

    dpop_lambda = args.dpop_lambda if args.dpop else 0.0
    DPO_SAVE_PATH = (f"checkpoints/dpop_policy_{ENV_NAME}.pt" if args.dpop
                     else f"checkpoints/dpo_policy_{ENV_NAME}.pt")

    # Load preference dataset
    with open(args.data, "rb") as f:
        ds = pickle.load(f)
    train_data, val_data = ds["train"], ds["val"]
    print(f"[INFO] Loaded {len(train_data)} train / {len(val_data)} val pairs "
          f"from {args.data}")

    # Load obs normalizer from BC stage
    obs_norm = ObsNormalizer()
    obs_norm.load(args.obs_norm)

    # Load BC policy
    policy = GaussianPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(args.bc, map_location=DEVICE))
    print(f"[INFO] Loaded BC policy from {args.bc}")

    # Frozen reference policy (deep copy of BC)
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # DPO fine-tuning
    policy = train_dpo(policy, ref_policy, train_data, val_data, obs_norm,
                       epochs=args.epochs, lr=args.lr, beta=args.beta,
                       dpop_lambda=dpop_lambda)

    # Final evaluation
    print(f"\n--- DPO final evaluation (50 episodes) ---")
    ret_d, succ_d = env_eval(policy, obs_norm, deterministic=True)
    print(f"  Deterministic : return {ret_d:.2f}  success {succ_d:.2%}")
    print(f"\n[INFO] Saved DPO policy → {DPO_SAVE_PATH}")