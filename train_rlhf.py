"""
train_rlhf.py
-------------
Phase 3: Reinforcement Learning from Human Feedback (RLHF)

Fixes applied (v3):
    1. Global observation normalization — compute mean/std from dataset,
       normalize all obs before feeding to RM and policy. Fixes dying ReLU
       caused by heterogeneous 39-dim MetaWorld obs space.
    2. Mean pooling in Bradley-Terry loss — .mean() instead of .sum().
       Prevents sigmoid saturation (logit 7.5 → gradient 0.0006).
    3. Mini-batch DataLoader for RM training — batch_size=64 instead of 1.
       Stabilizes Adam gradients and restores LayerNorm validity.
    4. Separate optimizers for policy/value (carried over from v2).
    5. Value loss clipping (carried over from v2).
    6. Running reward normalization (carried over from v2).
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
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
import metaworld

from config import (
    ENV_NAME, OBS_DIM, ACT_DIM, MAX_EPISODE_STEPS,
    HIDDEN_DIM, PREF_DATASET_PATH,
    RM_LR, RM_EPOCHS, RM_MIN_VAL_ACC, RM_WEIGHT_DECAY, RM_SAVE_PATH,
    PPO_LR, PPO_VALUE_LR, PPO_EPOCHS, PPO_STEPS, PPO_MINI_BATCH,
    PPO_OPT_EPOCHS, PPO_CLIP_EPS, PPO_VALUE_COEF, PPO_ENTROPY_COEF,
    PPO_VALUE_CLIP, PPO_GAE_LAMBDA, PPO_GAMMA, PPO_KL_COEF,
    PPO_MAX_GRAD_NORM, PPO_REWARD_NORM, RLHF_POLICY_SAVE_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on device: {DEVICE}")

RM_BATCH_SIZE = 64  # FIX 3: mini-batch size for RM training


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: Observation Normalizer
# Computes per-dimension mean and std from the full dataset and normalizes
# all observations before they enter any neural network.
# Must be fit on training data and then frozen for PPO rollouts.
# ══════════════════════════════════════════════════════════════════════════════

class ObsNormalizer:
    """
    Fits per-dimension (mean, std) statistics from the preference dataset,
    then normalizes observations to zero mean and unit variance.

    Why this matters: MetaWorld's 39-dim obs mixes XYZ coords (~0.5 scale),
    quaternions ([-1,1]), and permanently zeroed dims for unused objects.
    Without normalization the first linear layer is dominated by high-magnitude
    dims, causing dying ReLU with orthogonal initialization.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset: list):
        """Compute stats from all observations in the preference dataset."""
        all_obs = []
        for pair in dataset:
            all_obs.append(pair["chosen"]["observations"])
            all_obs.append(pair["rejected"]["observations"])
        all_obs = np.concatenate(all_obs, axis=0)  # (N*T, 39)
        self.mean = all_obs.mean(axis=0).astype(np.float32)
        self.std = all_obs.std(axis=0).astype(np.float32)
        self.std = np.clip(self.std, 1e-8, None)  # prevent division by zero
        print(f"[INFO] ObsNormalizer fit on {len(all_obs):,} observations.")
        print(f"[INFO] Obs mean range: [{self.mean.min():.3f}, {self.mean.max():.3f}]")
        print(f"[INFO] Obs std range:  [{self.std.min():.3f},  {self.std.max():.3f}]")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std

    def normalize_tensor(self, obs_t: torch.Tensor) -> torch.Tensor:
        mean_t = torch.FloatTensor(self.mean).to(obs_t.device)
        std_t = torch.FloatTensor(self.std).to(obs_t.device)
        return (obs_t - mean_t) / std_t

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, mean=self.mean, std=self.std)

    def load(self, path: str):
        data = np.load(path)
        self.mean = data["mean"].astype(np.float32)
        self.std = data["std"].astype(np.float32)


OBS_NORM_PATH = "checkpoints/obs_normalizer.npz"
RESUME_CKPT_PATH = "checkpoints/ppo_resume.pt"  # full resume checkpoint (every 10 iters)


# ══════════════════════════════════════════════════════════════════════════════
# Neural Network Architectures
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousRewardModel(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class ContinuousGaussianPolicy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden_dim=HIDDEN_DIM):
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

    def forward(self, obs):
        features = self.trunk(obs)
        mu = self.mu_head(features)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 2.0)).expand_as(mu)
        return mu, std

    def get_action_and_logprob(self, obs, action=None):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        if action is None:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def get_logprob(self, obs, action):
        mu, std = self.forward(obs)
        return Normal(mu, std).log_prob(action).sum(dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 3: PyTorch Dataset for mini-batch RM training
# ══════════════════════════════════════════════════════════════════════════════

class PreferenceDataset(Dataset):
    """Wraps preference pairs for use with PyTorch DataLoader."""

    def __init__(self, pairs: list, obs_normalizer: ObsNormalizer):
        self.pairs = pairs
        self.obs_normalizer = obs_normalizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        obs_w = self.obs_normalizer.normalize(
            pair["chosen"]["observations"].astype(np.float32))
        act_w = pair["chosen"]["actions"].astype(np.float32)
        obs_l = self.obs_normalizer.normalize(
            pair["rejected"]["observations"].astype(np.float32))
        act_l = pair["rejected"]["actions"].astype(np.float32)
        return (torch.FloatTensor(obs_w), torch.FloatTensor(act_w),
                torch.FloatTensor(obs_l), torch.FloatTensor(act_l))


def collate_preference_batch(batch):
    """Stack variable-length trajectory tensors into padded batches."""
    obs_w_list, act_w_list, obs_l_list, act_l_list = zip(*batch)
    return obs_w_list, act_w_list, obs_l_list, act_l_list


# ══════════════════════════════════════════════════════════════════════════════
# Running Reward Normaliser
# ══════════════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.var = m2 / tot_count
        self.count = tot_count

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Reward Model Training
# ══════════════════════════════════════════════════════════════════════════════

def train_reward_model(rm, obs_normalizer, data_path=PREF_DATASET_PATH,
                       epochs=RM_EPOCHS, lr=RM_LR):
    with open(data_path, "rb") as f:
        dataset_dict = pickle.load(f)
    train_data = dataset_dict["train"]
    val_data = dataset_dict["val"]

    # FIX 1: Fit obs normalizer on training data
    obs_normalizer.fit(train_data)
    obs_normalizer.save(OBS_NORM_PATH)

    # FIX 3: DataLoader with batch_size=64
    train_dataset = PreferenceDataset(train_data, obs_normalizer)
    val_dataset = PreferenceDataset(val_data, obs_normalizer)
    train_loader = DataLoader(train_dataset, batch_size=RM_BATCH_SIZE,
                              shuffle=True, collate_fn=collate_preference_batch)
    val_loader = DataLoader(val_dataset, batch_size=RM_BATCH_SIZE,
                            shuffle=False, collate_fn=collate_preference_batch)

    optimizer = optim.Adam(rm.parameters(), lr=lr, weight_decay=RM_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rm.train()

    print(f"\n{'=' * 60}")
    print(f"  Stage 1: Reward Model Training  ({len(train_data)} pairs)")
    print(f"  Batch size: {RM_BATCH_SIZE}  |  LR: {lr}")
    print(f"{'=' * 60}")

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(epochs):
        total_loss, total_acc, n_batches = 0.0, 0.0, 0

        for obs_w_list, act_w_list, obs_l_list, act_l_list in train_loader:
            batch_loss, batch_acc = 0.0, 0.0

            for obs_w, act_w, obs_l, act_l in zip(
                    obs_w_list, act_w_list, obs_l_list, act_l_list):
                obs_w = obs_w.to(DEVICE)
                act_w = act_w.to(DEVICE)
                obs_l = obs_l.to(DEVICE)
                act_l = act_l.to(DEVICE)

                # FIX 2: .mean() instead of .sum() — prevents sigmoid saturation
                reward_w = rm(obs_w, act_w).mean()
                reward_l = rm(obs_l, act_l).mean()
                loss = -F.logsigmoid(reward_w - reward_l)

                batch_loss += loss
                batch_acc += float((reward_w > reward_l).item())

            # Average over batch, then step
            batch_loss = batch_loss / len(obs_w_list)
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss.item()
            total_acc += batch_acc / len(obs_w_list)
            n_batches += 1

        scheduler.step()

        # Validation
        rm.eval()
        val_loss, val_acc, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for obs_w_list, act_w_list, obs_l_list, act_l_list in val_loader:
                for obs_w, act_w, obs_l, act_l in zip(
                        obs_w_list, act_w_list, obs_l_list, act_l_list):
                    obs_w = obs_w.to(DEVICE)
                    act_w = act_w.to(DEVICE)
                    obs_l = obs_l.to(DEVICE)
                    act_l = act_l.to(DEVICE)
                    rw = rm(obs_w, act_w).mean()
                    rl = rm(obs_l, act_l).mean()
                    val_loss += -F.logsigmoid(rw - rl).item()
                    val_acc += float((rw > rl).item())
                    n_val += 1
        rm.train()

        val_acc_pct = val_acc / n_val
        print(f"  RM Epoch {epoch + 1:>3}/{epochs}  "
              f"| train loss: {total_loss / n_batches:.4f}  "
              f"acc: {total_acc / n_batches:.2%}  "
              f"| val loss: {val_loss / n_val:.4f}  "
              f"acc: {val_acc_pct:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc_pct
            os.makedirs(os.path.dirname(RM_SAVE_PATH), exist_ok=True)
            torch.save(rm.state_dict(), RM_SAVE_PATH)

        if val_acc_pct >= RM_MIN_VAL_ACC:
            print(f"  RM hit target val accuracy {val_acc_pct:.2%} — stopping early.")
            break

    print(f"  Best RM val acc: {best_val_acc:.2%} → saved to {RM_SAVE_PATH}\n")

    if best_val_acc < 0.62:
        print(f"  [WARN] RM val accuracy {best_val_acc:.2%} is close to random.")
        print(f"  [WARN] Check label_data.py output — σ may still be too large.")

    rm.load_state_dict(torch.load(RM_SAVE_PATH, map_location=DEVICE))
    rm.eval()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: PPO
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(rewards, values, dones, gamma=PPO_GAMMA, lam=PPO_GAE_LAMBDA):
    T = len(rewards)
    advantages = torch.zeros(T, device=DEVICE)
    last_gae = 0.0
    for t in reversed(range(T)):
        non_terminal = 1.0 - dones[t].float()
        next_val = values[t + 1] if t + 1 < T else 0.0
        delta = rewards[t] + gamma * next_val * non_terminal - values[t]
        last_gae = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def collect_ppo_rollout(env, tasks, policy, value_net, rm, ref_policy,
                        obs_normalizer, reward_normaliser,
                        kl_coef=PPO_KL_COEF, n_steps=PPO_STEPS):
    obs_list, act_list, logp_list = [], [], []
    rew_list, val_list, done_list = [], [], []

    task = tasks[np.random.randint(len(tasks))]
    env.set_task(task)
    obs, _ = env.reset()

    policy.eval()
    value_net.eval()

    with torch.no_grad():
        steps = 0
        ep_steps = 0
        while steps < n_steps:
            # FIX 1: Normalize obs before feeding to policy/value
            obs_norm = obs_normalizer.normalize(obs.astype(np.float32))
            obs_t = torch.FloatTensor(obs_norm).unsqueeze(0).to(DEVICE)
            action, logp, _ = policy.get_action_and_logprob(obs_t)
            val = value_net(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_obs, _, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Shaped reward — also normalize obs for RM
            rm_reward = rm(obs_t, action).item()
            ref_logp = ref_policy.get_logprob(obs_t, action).item()
            kl_penalty = logp.item() - ref_logp
            shaped_rew = rm_reward - kl_coef * kl_penalty

            obs_list.append(obs_norm)
            act_list.append(action_np.astype(np.float32))
            logp_list.append(logp.item())
            rew_list.append(shaped_rew)
            val_list.append(val.item())
            done_list.append(float(done))

            obs = next_obs
            steps += 1
            ep_steps += 1

            if done or ep_steps >= MAX_EPISODE_STEPS:
                task = tasks[np.random.randint(len(tasks))]
                env.set_task(task)
                obs, _ = env.reset()
                ep_steps = 0

    policy.train()
    value_net.train()

    raw_rewards = np.array(rew_list)
    if PPO_REWARD_NORM:
        reward_normaliser.update(raw_rewards)
        normalised_rewards = reward_normaliser.normalise(raw_rewards)
    else:
        normalised_rewards = raw_rewards

    obs_t = torch.FloatTensor(np.array(obs_list)).to(DEVICE)
    act_t = torch.FloatTensor(np.array(act_list)).to(DEVICE)
    logp_t = torch.FloatTensor(logp_list).to(DEVICE)
    rew_t = torch.FloatTensor(normalised_rewards).to(DEVICE)
    val_t = torch.FloatTensor(val_list).to(DEVICE)
    done_t = torch.FloatTensor(done_list).to(DEVICE)

    adv_t, ret_t = compute_gae(rew_t, val_t, done_t)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    return dict(obs=obs_t, act=act_t, logp=logp_t, adv=adv_t, ret=ret_t, val=val_t)


def ppo_update(policy, value_net, policy_optimizer, value_optimizer, rollout,
               clip_eps=PPO_CLIP_EPS, value_coef=PPO_VALUE_COEF,
               entropy_coef=PPO_ENTROPY_COEF, opt_epochs=PPO_OPT_EPOCHS,
               mini_batch=PPO_MINI_BATCH):
    N = rollout["obs"].shape[0]
    old_vals = rollout["val"].detach()
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n": 0}

    for _ in range(opt_epochs):
        idxs = torch.randperm(N)
        for start in range(0, N, mini_batch):
            b = idxs[start: start + mini_batch]
            obs_b = rollout["obs"][b]
            act_b = rollout["act"][b]
            adv_b = rollout["adv"][b]
            ret_b = rollout["ret"][b]
            old_lp = rollout["logp"][b]
            old_v = old_vals[b]

            _, new_lp, entropy = policy.get_action_and_logprob(obs_b, act_b)
            new_val = value_net(obs_b)

            ratio = torch.exp(new_lp - old_lp)
            clip_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()

            if PPO_VALUE_CLIP:
                v_clipped = old_v + torch.clamp(new_val - old_v, -clip_eps, clip_eps)
                v_loss_raw = F.mse_loss(new_val, ret_b)
                v_loss_clip = F.mse_loss(v_clipped, ret_b)
                value_loss = torch.max(v_loss_raw, v_loss_clip)
            else:
                value_loss = F.mse_loss(new_val, ret_b)

            entropy_loss = -entropy.mean()

            policy_total = policy_loss + entropy_coef * entropy_loss
            policy_optimizer.zero_grad()
            policy_total.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), PPO_MAX_GRAD_NORM)
            policy_optimizer.step()

            value_optimizer.zero_grad()
            (value_coef * value_loss).backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), PPO_MAX_GRAD_NORM)
            value_optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += (-entropy_loss).item()
            stats["n"] += 1

    n = stats["n"]
    return {k: v / n for k, v in stats.items() if k != "n"}


def evaluate_policy(env, tasks, policy, obs_normalizer, n_episodes=20):
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
                action, _, _ = policy.get_action_and_logprob(obs_t)
                action_np = np.clip(action.squeeze(0).cpu().numpy(),
                                    env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_ret += reward
                if info.get("success", False):
                    success = True
                if terminated or truncated:
                    break
            returns.append(ep_ret)
            successes.append(float(success))
    policy.train()
    return {"mean_return": float(np.mean(returns)),
            "success_rate": float(np.mean(successes))}


def save_resume_checkpoint(path, policy, value_net, policy_optimizer,
                           value_optimizer, reward_normaliser, iteration, best_success):
    torch.save({
        "iteration": iteration,
        "best_success": best_success,
        "policy": policy.state_dict(),
        "value_net": value_net.state_dict(),
        "policy_optimizer": policy_optimizer.state_dict(),
        "value_optimizer": value_optimizer.state_dict(),
        "rms_mean": reward_normaliser.mean,
        "rms_var": reward_normaliser.var,
        "rms_count": reward_normaliser.count,
    }, path)


def load_resume_checkpoint(path, policy, value_net, policy_optimizer,
                           value_optimizer, reward_normaliser):
    ckpt = torch.load(path, map_location=DEVICE)
    policy.load_state_dict(ckpt["policy"])
    value_net.load_state_dict(ckpt["value_net"])
    policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
    value_optimizer.load_state_dict(ckpt["value_optimizer"])
    reward_normaliser.mean = ckpt["rms_mean"]
    reward_normaliser.var = ckpt["rms_var"]
    reward_normaliser.count = ckpt["rms_count"]
    print(f"[INFO] Resumed from iteration {ckpt['iteration'] + 1}")
    return ckpt["iteration"] + 1, ckpt["best_success"]  # start_iter, best_success


def train_ppo(policy, value_net, rm, ref_policy, obs_normalizer,
              ppo_epochs=PPO_EPOCHS, kl_coef=PPO_KL_COEF, resume=False):
    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks

    policy_optimizer = optim.Adam(policy.parameters(), lr=PPO_LR)
    value_optimizer = optim.Adam(value_net.parameters(), lr=PPO_VALUE_LR)
    reward_normaliser = RunningMeanStd()

    start_iter = 0
    best_success = -1.0

    # ── Resume from checkpoint if requested ──────────────────────────────────
    if resume and os.path.exists(RESUME_CKPT_PATH):
        start_iter, best_success = load_resume_checkpoint(
            RESUME_CKPT_PATH, policy, value_net,
            policy_optimizer, value_optimizer, reward_normaliser)
    elif resume:
        print("[WARN] --resume set but no checkpoint found — starting fresh.")

    print(f"\n{'=' * 60}")
    print(f"  Stage 2: PPO Training  ({ppo_epochs} iterations)")
    print(f"  Starting from iteration: {start_iter}")
    print(f"  Policy lr: {PPO_LR}  |  Value lr: {PPO_VALUE_LR}")
    print(f"  Reward norm: {PPO_REWARD_NORM}  |  Value clip: {PPO_VALUE_CLIP}")
    print(f"{'=' * 60}")

    os.makedirs(os.path.dirname(RLHF_POLICY_SAVE_PATH), exist_ok=True)

    for iteration in range(start_iter, ppo_epochs):
        rollout = collect_ppo_rollout(
            env, tasks, policy, value_net, rm, ref_policy,
            obs_normalizer, reward_normaliser, kl_coef=kl_coef
        )
        stats = ppo_update(policy, value_net, policy_optimizer,
                           value_optimizer, rollout)

        if (iteration + 1) % 10 == 0:
            eval_stats = evaluate_policy(env, tasks, policy, obs_normalizer,
                                         n_episodes=20)
            print(f"  Iter {iteration + 1:>4}/{ppo_epochs}  "
                  f"| π_loss: {stats['policy_loss']:.4f}  "
                  f"| v_loss: {stats['value_loss']:.4f}  "
                  f"| entropy: {stats['entropy']:.4f}  "
                  f"| ret: {eval_stats['mean_return']:.2f}  "
                  f"| success: {eval_stats['success_rate']:.2%}")

            # Save best policy checkpoint
            if eval_stats["mean_return"] > best_success:
                best_success = eval_stats["mean_return"]
                torch.save(policy.state_dict(), RLHF_POLICY_SAVE_PATH)

            # Save full resume checkpoint every 10 iters
            save_resume_checkpoint(
                RESUME_CKPT_PATH, policy, value_net,
                policy_optimizer, value_optimizer,
                reward_normaliser, iteration, best_success)

            # Also copy resume checkpoint to Drive for safety
            drive_resume = "/content/drive/MyDrive/CS234_Project/checkpoints/ppo_resume.pt"
            if os.path.exists("/content/drive"):
                import shutil
                shutil.copy(RESUME_CKPT_PATH, drive_resume)

    print(f"  Best RLHF policy saved → {RLHF_POLICY_SAVE_PATH}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=PREF_DATASET_PATH)
    parser.add_argument("--rm_epochs", type=int, default=RM_EPOCHS)
    parser.add_argument("--ppo_epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--kl_coef", type=float, default=PPO_KL_COEF)
    parser.add_argument("--skip_rm", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Resume PPO from last saved checkpoint")
    args = parser.parse_args()

    reward_model = ContinuousRewardModel().to(DEVICE)
    policy = ContinuousGaussianPolicy().to(DEVICE)
    value_net = ValueNetwork().to(DEVICE)
    obs_normalizer = ObsNormalizer()

    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    if args.skip_rm and os.path.exists(RM_SAVE_PATH):
        print(f"[INFO] Loading existing RM from {RM_SAVE_PATH}")
        reward_model.load_state_dict(torch.load(RM_SAVE_PATH, map_location=DEVICE))
        reward_model.eval()
        obs_normalizer.load(OBS_NORM_PATH)
        print(f"[INFO] Loaded obs normalizer from {OBS_NORM_PATH}")
    else:
        train_reward_model(reward_model, obs_normalizer,
                           data_path=args.data, epochs=args.rm_epochs)

    train_ppo(policy, value_net, reward_model, ref_policy, obs_normalizer,
              ppo_epochs=args.ppo_epochs, kl_coef=args.kl_coef, resume=args.resume)
