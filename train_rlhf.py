"""
train_rlhf.py
-------------
Phase 3: Reinforcement Learning from Human Feedback (RLHF)

Two-stage pipeline:
    Stage 1 — Reward Model (RM) Training
        A neural network r_φ(s, a) is trained via the Bradley-Terry contrastive
        loss to assign higher scalar rewards to preferred trajectory steps.

        L_RM(φ) = -E_{(τ_w,τ_l)~D}[ log σ( Σ_t r_φ(s_t,a_t)|_τ_w
                                          - Σ_t r_φ(s_t,a_t)|_τ_l ) ]

    Stage 2 — PPO with Learned Reward + KL Penalty
        The policy π_θ(a|s) = N(μ_θ(s), diag(σ²_θ(s))) is optimised via PPO
        against the frozen RM.  A KL penalty against the reference policy
        prevents reward hacking / distribution collapse.

        r_shaped(s,a) = r_φ(s,a) - β_KL · [ log π_θ(a|s) - log π_ref(a|s) ]

        L_PPO(θ) = E_t[ min( ρ_t(θ) Â_t,
                              clip(ρ_t(θ), 1-ε, 1+ε) Â_t ) ]
                 - c_v · L_value - c_e · H[π_θ]

Architecture
------------
    - ContinuousRewardModel   : MLP, (obs+act) → scalar
    - ContinuousGaussianPolicy: MLP, obs → (μ, σ) for Gaussian policy
    - ValueNetwork            : MLP, obs → V(s)  (separate head for stability)

Usage
-----
    python train_rlhf.py               # uses config.py defaults
    python train_rlhf.py --ppo_epochs 300 --kl_coef 0.05
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
    RM_LR, RM_EPOCHS, RM_WEIGHT_DECAY, RM_SAVE_PATH,
    PPO_LR, PPO_EPOCHS, PPO_STEPS, PPO_MINI_BATCH,
    PPO_OPT_EPOCHS, PPO_CLIP_EPS, PPO_VALUE_COEF, PPO_ENTROPY_COEF,
    PPO_GAE_LAMBDA, PPO_GAMMA, PPO_KL_COEF, PPO_MAX_GRAD_NORM,
    RLHF_POLICY_SAVE_PATH,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on device: {DEVICE}")


# ══════════════════════════════════════════════════════════════════════════════
# Neural Network Architectures
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousRewardModel(nn.Module):
    """
    r_φ(s, a) → scalar

    Concatenates state and action, feeds through a residual MLP,
    outputs a single scalar value used as a pseudo-reward signal.
    LayerNorm is critical for training stability across trajectory steps
    with widely varying scales.
    """

    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM,
                 hidden_dim: int = HIDDEN_DIM):
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)  # (T,)


class ContinuousGaussianPolicy(nn.Module):
    """
    π_θ(a|s) = N( μ_θ(s), diag(σ²_θ) )

    State-independent log_std (a learnable parameter) is common in
    continuous-control RL and avoids instabilities from state-dependent variance.
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
        # log_std clamped in [-2, 2] for numerical safety
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        features = self.trunk(obs)
        mu = self.mu_head(features)
        std = torch.exp(torch.clamp(self.log_std, -2.0, 2.0)).expand_as(mu)
        return mu, std

    def get_action_and_logprob(self, obs: torch.Tensor, action: torch.Tensor = None):
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        if action is None:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # sum over action dims
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy

    def get_logprob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mu, std = self.forward(obs)
        dist = Normal(mu, std)
        return dist.log_prob(action).sum(dim=-1)


class ValueNetwork(nn.Module):
    """V(s) — separate from policy for PPO's actor-critic stability."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden_dim: int = HIDDEN_DIM):
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

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1: Reward Model Training (Bradley-Terry)
# ══════════════════════════════════════════════════════════════════════════════

def train_reward_model(
        rm: ContinuousRewardModel,
        data_path: str = PREF_DATASET_PATH,
        epochs: int = RM_EPOCHS,
        lr: float = RM_LR,
) -> None:
    """
    Minimise the Bradley-Terry negative log-likelihood:

        L_RM = -log σ( R̂(τ_w) - R̂(τ_l) )
        where R̂(τ) = Σ_t r_φ(s_t, a_t)
    """
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

    for epoch in range(epochs):
        np.random.shuffle(train_data)
        total_loss, total_acc = 0.0, 0.0

        for pair in train_data:
            obs_w = torch.FloatTensor(pair["chosen"]["observations"]).to(DEVICE)
            act_w = torch.FloatTensor(pair["chosen"]["actions"]).to(DEVICE)
            obs_l = torch.FloatTensor(pair["rejected"]["observations"]).to(DEVICE)
            act_l = torch.FloatTensor(pair["rejected"]["actions"]).to(DEVICE)

            # Cumulative trajectory rewards (scalar each)
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

        # ── Validation ───────────────────────────────────────────────────────
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

        n_train, n_val = len(train_data), len(val_data)
        print(f"  RM Epoch {epoch + 1:>3}/{epochs}  "
              f"| train loss: {total_loss / n_train:.4f}  acc: {total_acc / n_train:.2%}  "
              f"| val loss: {val_loss / n_val:.4f}  acc: {val_acc / n_val:.2%}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(RM_SAVE_PATH), exist_ok=True)
            torch.save(rm.state_dict(), RM_SAVE_PATH)

    print(f"  Best RM saved → {RM_SAVE_PATH}\n")
    rm.load_state_dict(torch.load(RM_SAVE_PATH, map_location=DEVICE))
    rm.eval()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2: PPO with Shaped Reward
# ══════════════════════════════════════════════════════════════════════════════

def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = PPO_GAMMA,
        lam: float = PPO_GAE_LAMBDA,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generalised Advantage Estimation (GAE-λ):
        δ_t   = r_t + γ · V(s_{t+1}) - V(s_t)
        Â_t   = Σ_{k=0}^{T-t} (γλ)^k δ_{t+k}

    Returns advantages and value targets (advantages + values).
    """
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


def collect_ppo_rollout(
        env,
        tasks,
        policy: ContinuousGaussianPolicy,
        value_net: ValueNetwork,
        rm: ContinuousRewardModel,
        ref_policy: ContinuousGaussianPolicy,
        kl_coef: float = PPO_KL_COEF,
        n_steps: int = PPO_STEPS,
) -> dict:
    """
    Roll out n_steps transitions.  Reward is shaped as:
        r_shaped = r_φ(s,a) - β_KL · [log π_θ(a|s) - log π_ref(a|s)]
    """
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

            next_obs, env_reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            # Shaped reward: RM reward − KL penalty
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

    obs_t = torch.FloatTensor(np.array(obs_list)).to(DEVICE)
    act_t = torch.FloatTensor(np.array(act_list)).to(DEVICE)
    logp_t = torch.FloatTensor(logp_list).to(DEVICE)
    rew_t = torch.FloatTensor(rew_list).to(DEVICE)
    val_t = torch.FloatTensor(val_list).to(DEVICE)
    done_t = torch.FloatTensor(done_list).to(DEVICE)

    adv_t, ret_t = compute_gae(rew_t, val_t, done_t)

    # Normalise advantages (reduces variance, standard PPO trick)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    return dict(obs=obs_t, act=act_t, logp=logp_t, adv=adv_t, ret=ret_t)


def ppo_update(
        policy: ContinuousGaussianPolicy,
        value_net: ValueNetwork,
        optimizer: optim.Optimizer,
        rollout: dict,
        clip_eps: float = PPO_CLIP_EPS,
        value_coef: float = PPO_VALUE_COEF,
        entropy_coef: float = PPO_ENTROPY_COEF,
        opt_epochs: int = PPO_OPT_EPOCHS,
        mini_batch: int = PPO_MINI_BATCH,
) -> dict:
    """
    Run multiple epochs of mini-batch gradient updates on the collected rollout.
    Returns diagnostics.
    """
    N = rollout["obs"].shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}

    for _ in range(opt_epochs):
        idxs = torch.randperm(N)
        for start in range(0, N, mini_batch):
            batch_idx = idxs[start: start + mini_batch]
            obs_b = rollout["obs"][batch_idx]
            act_b = rollout["act"][batch_idx]
            adv_b = rollout["adv"][batch_idx]
            ret_b = rollout["ret"][batch_idx]
            old_lp = rollout["logp"][batch_idx]

            _, new_lp, entropy = policy.get_action_and_logprob(obs_b, act_b)
            val = value_net(obs_b)

            # Importance sampling ratio
            ratio = torch.exp(new_lp - old_lp)
            clip_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

            # PPO clipped surrogate objective (negate to minimise)
            policy_loss = -torch.min(ratio * adv_b, clip_ratio * adv_b).mean()
            value_loss = F.mse_loss(val, ret_b)
            entropy_loss = -entropy.mean()

            total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_net.parameters()),
                PPO_MAX_GRAD_NORM,
            )
            optimizer.step()

            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += (-entropy_loss).item()
            stats["n_updates"] += 1

    n = stats["n_updates"]
    return {k: v / n for k, v in stats.items() if k != "n_updates"}


def evaluate_policy(env, tasks, policy: ContinuousGaussianPolicy, n_episodes: int = 20) -> dict:
    """Quick evaluation: task success rate and mean episodic return."""
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
                action_np = action.squeeze(0).cpu().numpy()
                action_np = np.clip(action_np, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action_np)
                ep_ret += reward
                if info.get("success", False):
                    success = True
                if terminated or truncated:
                    break

            returns.append(ep_ret)
            successes.append(float(success))

    policy.train()
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
    }


def train_ppo(
        policy: ContinuousGaussianPolicy,
        value_net: ValueNetwork,
        rm: ContinuousRewardModel,
        ref_policy: ContinuousGaussianPolicy,
        ppo_epochs: int = PPO_EPOCHS,
        kl_coef: float = PPO_KL_COEF,
) -> None:
    """Full PPO training loop against the frozen reward model."""

    ml1 = metaworld.ML1(ENV_NAME)
    env = ml1.train_classes[ENV_NAME]()
    tasks = ml1.train_tasks

    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()), lr=PPO_LR
    )

    print(f"\n{'=' * 60}")
    print(f"  Stage 2: PPO Training  ({ppo_epochs} iterations)")
    print(f"{'=' * 60}")

    best_success = -1.0
    os.makedirs(os.path.dirname(RLHF_POLICY_SAVE_PATH), exist_ok=True)

    for iteration in range(ppo_epochs):
        rollout = collect_ppo_rollout(
            env, tasks, policy, value_net, rm, ref_policy, kl_coef=kl_coef
        )
        stats = ppo_update(policy, value_net, optimizer, rollout)

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
    parser = argparse.ArgumentParser(description="Train RLHF (RM + PPO) pipeline.")
    parser.add_argument("--data", default=PREF_DATASET_PATH)
    parser.add_argument("--rm_epochs", type=int, default=RM_EPOCHS)
    parser.add_argument("--ppo_epochs", type=int, default=PPO_EPOCHS)
    parser.add_argument("--kl_coef", type=float, default=PPO_KL_COEF)
    parser.add_argument("--skip_rm", action="store_true",
                        help="Load existing RM and skip Stage 1")
    args = parser.parse_args()

    # ── Build models ──────────────────────────────────────────────────────────
    reward_model = ContinuousRewardModel().to(DEVICE)
    policy = ContinuousGaussianPolicy().to(DEVICE)
    value_net = ValueNetwork().to(DEVICE)

    # Reference policy: deep-frozen copy of the initial policy (π_base)
    ref_policy = copy.deepcopy(policy)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad = False

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    if args.skip_rm and os.path.exists(RM_SAVE_PATH):
        print(f"[INFO] Loading existing RM from {RM_SAVE_PATH}")
        reward_model.load_state_dict(torch.load(RM_SAVE_PATH, map_location=DEVICE))
        reward_model.eval()
    else:
        train_reward_model(reward_model, data_path=args.data, epochs=args.rm_epochs)

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    train_ppo(policy, value_net, reward_model, ref_policy,
              ppo_epochs=args.ppo_epochs, kl_coef=args.kl_coef)
