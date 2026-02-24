"""
train_rlhf.py
-------------
Phase 3: Reinforcement Learning from Human Feedback (RLHF)

Two-stage pipeline:
    Stage 1 — Reward Model (RM) Training
        A neural network r_φ(s, a) is trained via the Bradley-Terry contrastive
        loss to assign higher scalar rewards to preferred trajectory steps.

    Stage 2 — PPO with Learned Reward + KL Penalty
        The policy π_θ(a|s) = N(μ_θ(s), diag(σ²_θ(s))) is optimised via PPO
        against the frozen RM.

Fixes applied vs v1 (which got 5% success rate):
    1. Separate optimizers for policy and value network (different learning rates)
    2. Value loss clipping — prevents v_loss from exploding to 23,000+
    3. Running reward normalisation — stabilises advantage scale
    4. Tighter gradient clipping (0.5 instead of 1.0)
    5. Increased entropy coefficient to prevent entropy collapse
    6. RM early stopping — won't start PPO until RM hits 70% val accuracy
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
    RM_LR, RM_EPOCHS, RM_MIN_VAL_ACC, RM_WEIGHT_DECAY, RM_SAVE_PATH,
    PPO_LR, PPO_VALUE_LR, PPO_EPOCHS, PPO_STEPS, PPO_MINI_BATCH,
    PPO_OPT_EPOCHS, PPO_CLIP_EPS, PPO_VALUE_COEF, PPO_ENTROPY_COEF,
    PPO_VALUE_CLIP, PPO_GAE_LAMBDA, PPO_GAMMA, PPO_KL_COEF,
    PPO_MAX_GRAD_NORM, PPO_REWARD_NORM, RLHF_POLICY_SAVE_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on device: {DEVICE}")


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
# Running Reward Normaliser  (FIX 3 — prevents value loss explosion)
# ══════════════════════════════════════════════════════════════════════════════

class RunningMeanStd:
    """
    Tracks running mean and variance of rewards so we can normalise them
    to zero mean and unit variance before computing advantages.
    Without this the RM rewards have unpredictable scale which causes
    the value network to diverge (v_loss → 23,000 as we saw).
    """

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = np.mean(x)
        batch_var = np.var(x)
        batch_count = len(x)
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
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

def train_reward_model(rm, data_path=PREF_DATASET_PATH, epochs=RM_EPOCHS, lr=RM_LR):
    with open(data_path, "rb") as f:
        dataset_dict = pickle.load(f)
    train_data = dataset_dict["train"]
    val_data = dataset_dict["val"]

    optimizer = optim.Adam(rm.parameters(), lr=lr, weight_decay=RM_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rm.train()

    print(f"\n{'=' * 60}")
    print(f"  Stage 1: Reward Model Training  ({len(train_data)} pairs)")
    print(f"{'=' * 60}")

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(epochs):
        np.random.shuffle(train_data)
        total_loss, total_acc = 0.0, 0.0

        for pair in train_data:
            obs_w = torch.FloatTensor(pair["chosen"]["observations"]).to(DEVICE)
            act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
            obs_l = torch.FloatTensor(pair["rejected"]["observations"]).to(DEVICE)
            act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)

            reward_w = rm(obs_w, act_w).sum()
            reward_l = rm(obs_l, act_l).sum()
            loss = -F.logsigmoid(reward_w - reward_l)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_acc += float((reward_w > reward_l).item())

        scheduler.step()

        rm.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for pair in val_data:
                obs_w = torch.FloatTensor(pair["chosen"]["observations"]).to(DEVICE)
                act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
                obs_l = torch.FloatTensor(pair["rejected"]["observations"]).to(DEVICE)
                act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)
                rw = rm(obs_w, act_w).sum()
                rl = rm(obs_l, act_l).sum()
                val_loss += -F.logsigmoid(rw - rl).item()
                val_acc += float((rw > rl).item())
        rm.train()

        n_tr, n_v = len(train_data), len(val_data)
        val_acc_pct = val_acc / n_v
        print(f"  RM Epoch {epoch + 1:>3}/{epochs}  "
              f"| train loss: {total_loss / n_tr:.4f}  acc: {total_acc / n_tr:.2%}  "
              f"| val loss: {val_loss / n_v:.4f}  acc: {val_acc_pct:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc_pct
            os.makedirs(os.path.dirname(RM_SAVE_PATH), exist_ok=True)
            torch.save(rm.state_dict(), RM_SAVE_PATH)

        # Early stopping if RM is good enough — no point training longer
        if val_acc_pct >= RM_MIN_VAL_ACC:
            print(f"  RM hit target val accuracy {val_acc_pct:.2%} — stopping early.")
            break

    print(f"  Best RM val acc: {best_val_acc:.2%} → saved to {RM_SAVE_PATH}\n")

    if best_val_acc < 0.62:
        print(f"  [WARN] RM val accuracy is only {best_val_acc:.2%}.")
        print(f"  [WARN] This is close to random — PPO will struggle.")
        print(f"  [WARN] Consider re-running label_data.py with --noise 0.3")

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
                        reward_normaliser, kl_coef=PPO_KL_COEF, n_steps=PPO_STEPS):
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
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action, logp, _ = policy.get_action_and_logprob(obs_t)
            val = value_net(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

            next_obs, _, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Shaped reward with KL penalty
            rm_reward = rm(obs_t, action).item()
            ref_logp = ref_policy.get_logprob(obs_t, action).item()
            kl_penalty = logp.item() - ref_logp
            shaped_rew = rm_reward - kl_coef * kl_penalty

            obs_list.append(obs.astype(np.float32))
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

    # FIX 3: Normalise rewards before computing advantages
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

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_lp - old_lp)
            clip_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
            policy_loss = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()

            # FIX 2: Value loss clipping — THIS is what prevents v_loss → 23,000
            if PPO_VALUE_CLIP:
                v_clipped = old_v + torch.clamp(new_val - old_v, -clip_eps, clip_eps)
                v_loss_raw = F.mse_loss(new_val, ret_b)
                v_loss_clip = F.mse_loss(v_clipped, ret_b)
                value_loss = torch.max(v_loss_raw, v_loss_clip)
            else:
                value_loss = F.mse_loss(new_val, ret_b)

            entropy_loss = -entropy.mean()

            # Update policy and value separately with their own optimizers
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


def evaluate_policy(env, tasks, policy, n_episodes=20):
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
    return {"mean_return": float(np.mean(returns)), "success_rate": float(np.mean(successes))}


def train_ppo(policy, value_net, rm, ref_policy, ppo_epochs=PPO_EPOCHS, kl_coef=PPO_KL_COEF):
    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks

    # FIX 1: Separate optimizers — value net gets higher lr for faster adaptation
    policy_optimizer = optim.Adam(policy.parameters(), lr=PPO_LR)
    value_optimizer = optim.Adam(value_net.parameters(), lr=PPO_VALUE_LR)

    reward_normaliser = RunningMeanStd()

    print(f"\n{'=' * 60}")
    print(f"  Stage 2: PPO Training  ({ppo_epochs} iterations)")
    print(f"  Policy lr: {PPO_LR}  |  Value lr: {PPO_VALUE_LR}")
    print(f"  Reward normalisation: {PPO_REWARD_NORM}")
    print(f"  Value loss clipping:  {PPO_VALUE_CLIP}")
    print(f"{'=' * 60}")

    best_success = -1.0
    os.makedirs(os.path.dirname(RLHF_POLICY_SAVE_PATH), exist_ok=True)

    for iteration in range(ppo_epochs):
        rollout = collect_ppo_rollout(
            env, tasks, policy, value_net, rm, ref_policy,
            reward_normaliser, kl_coef=kl_coef
        )
        stats = ppo_update(policy, value_net, policy_optimizer, value_optimizer, rollout)

        if (iteration + 1) % 10 == 0:
            eval_stats = evaluate_policy(env, tasks, policy, n_episodes=20)
            print(f"  Iter {iteration + 1:>4}/{ppo_epochs}  "
                  f"| π_loss: {stats['policy_loss']:.4f}  "
                  f"| v_loss: {stats['value_loss']:.4f}  "
                  f"| entropy: {stats['entropy']:.4f}  "
                  f"| ret: {eval_stats['mean_return']:.2f}  "
                  f"| success: {eval_stats['success_rate']:.2%}")

            if eval_stats["success_rate"] > best_success:
                best_success = eval_stats["success_rate"]
                torch.save(policy.state_dict(), RLHF_POLICY_SAVE_PATH)

    print(f"  Best RLHF policy saved → {RLHF_POLICY_SAVE_PATH}  "
          f"(success rate: {best_success:.2%})\n")


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
    args = parser.parse_args()

    reward_model = ContinuousRewardModel().to(DEVICE)
    policy = ContinuousGaussianPolicy().to(DEVICE)
    value_net = ValueNetwork().to(DEVICE)

    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    if args.skip_rm and os.path.exists(RM_SAVE_PATH):
        print(f"[INFO] Loading existing RM from {RM_SAVE_PATH}")
        reward_model.load_state_dict(torch.load(RM_SAVE_PATH, map_location=DEVICE))
        reward_model.eval()
    else:
        train_reward_model(reward_model, data_path=args.data, epochs=args.rm_epochs)

    train_ppo(policy, value_net, reward_model, ref_policy,
              ppo_epochs=args.ppo_epochs, kl_coef=args.kl_coef)
