"""
config.py
---------
Central configuration for the RLHF vs DPO continuous control experiment.
All scripts import from here so hyperparameters are changed in one place.
"""

# ─── Environment ──────────────────────────────────────────────────────────────
ENV_NAME = "reach-v3"  # MetaWorld task name
OBS_DIM = 39  # MetaWorld state space dimensionality
ACT_DIM = 4  # 3-D Cartesian velocity + gripper
MAX_EPISODE_STEPS = 150  # Maximum steps per episode

# ─── Data Collection ──────────────────────────────────────────────────────────
NUM_TRAJECTORIES = 5000  # Total trajectories to collect
EXPLORATION_NOISE = 0.10  # Std-dev of additive Gaussian noise on scripted actions
RAW_DATA_PATH = "data/raw_trajectories.h5"

# ─── Preference Labeling ──────────────────────────────────────────────────────
NUM_PREF_PAIRS = 10000  # Pairwise preference comparisons to synthesize
# Per friend's doc Section 6.4: σ_human_error ≤ ½ × Mean|ΔR|
# Run label_data with --noise 0 first, check Mean|margin|, then set σ = ½ × that value
# For reach-v3 with discounted returns, Mean|margin| ≈ 5-8, so σ ≤ 2.5
# Start at 0.1 (very safe), increase if RM accuracy is too high (>95%)
HUMAN_ERROR_NOISE = 0.1
PREF_DATASET_PATH = "data/preference_dataset.pkl"

# ─── Shared Network Architecture ──────────────────────────────────────────────
HIDDEN_DIM = 256

# ─── RLHF (Reward Model + PPO) ────────────────────────────────────────────────
RM_LR = 3e-4
# FIX 2: Increased from 15 → 25 epochs to give RM more time to converge
RM_EPOCHS = 25
# FIX 3: Raised accuracy threshold — don't start PPO until RM is good enough
RM_MIN_VAL_ACC = 0.70  # Stop early if RM hits this val accuracy
RM_WEIGHT_DECAY = 1e-4
RM_SAVE_PATH = "checkpoints/reward_model.pt"

# FIX 4: Reduced PPO lr from 3e-4 → 1e-4 (was too aggressive, caused instability)
PPO_LR = 1e-4
PPO_VALUE_LR = 3e-4  # Separate (higher) lr for value network
PPO_EPOCHS = 300  # More iterations since we have Colab Pro A100
PPO_STEPS = 2048  # Environment steps per PPO iteration
PPO_MINI_BATCH = 64
PPO_OPT_EPOCHS = 10  # Gradient updates per PPO iteration
PPO_CLIP_EPS = 0.2
# FIX 5: Reduced value coef from 0.5 → 0.25 to reduce value network's influence
PPO_VALUE_COEF = 0.25
# FIX 6: Increased entropy coef from 0.01 → 0.05 to prevent entropy collapse
PPO_ENTROPY_COEF = 0.05
# FIX 7: Add value function clipping (same eps as policy — prevents v_loss explosion)
PPO_VALUE_CLIP = True
PPO_GAE_LAMBDA = 0.95
PPO_GAMMA = 0.99
PPO_KL_COEF = 0.01
# FIX 8: Tighter gradient clipping from 1.0 → 0.5
PPO_MAX_GRAD_NORM = 0.5
# FIX 9: Normalize shaped rewards with running stats before advantage estimation
PPO_REWARD_NORM = True
RLHF_POLICY_SAVE_PATH = "checkpoints/rlhf_policy.pt"

# ─── DPO ──────────────────────────────────────────────────────────────────────
DPO_LR = 1e-5
DPO_EPOCHS = 20
DPO_BETA = 0.1
DPO_SAVE_PATH = "checkpoints/dpo_policy.pt"

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_EPISODES = 100
EVAL_SEEDS = [0, 1, 2, 3, 4]
RESULTS_PATH = "results/eval_results.json"
