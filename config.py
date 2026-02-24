"""
config.py
---------
Central configuration for the RLHF vs DPO continuous control experiment.
All scripts import from here so hyperparameters are changed in one place.
"""

# ─── Environment ──────────────────────────────────────────────────────────────
ENV_NAME = "reach-v3"  # MetaWorld task name (see metaworld.ML1 tasks)
OBS_DIM = 39  # MetaWorld state space dimensionality
ACT_DIM = 4  # 3-D Cartesian velocity + gripper
MAX_EPISODE_STEPS = 150  # Maximum steps per episode

# ─── Data Collection ──────────────────────────────────────────────────────────
NUM_TRAJECTORIES = 2000  # Total trajectories to collect
EXPLORATION_NOISE = 0.10  # Std-dev of additive Gaussian noise on scripted actions
RAW_DATA_PATH = "data/raw_trajectories.h5"

# ─── Preference Labeling ──────────────────────────────────────────────────────
NUM_PREF_PAIRS = 5000  # Pairwise preference comparisons to synthesize
HUMAN_ERROR_NOISE = 1.5  # Std-dev of simulated human labelling noise (σ_human_error)
PREF_DATASET_PATH = "data/preference_dataset.pkl"

# ─── Shared Network Architecture ──────────────────────────────────────────────
HIDDEN_DIM = 256

# ─── RLHF (Reward Model + PPO) ────────────────────────────────────────────────
RM_LR = 3e-4
RM_EPOCHS = 15
RM_WEIGHT_DECAY = 1e-4
RM_SAVE_PATH = "checkpoints/reward_model.pt"

PPO_LR = 3e-4
PPO_EPOCHS = 200  # Outer iterations (each collects fresh rollouts)
PPO_STEPS = 2048  # Environment steps per PPO iteration
PPO_MINI_BATCH = 64
PPO_OPT_EPOCHS = 10  # Gradient updates per PPO iteration
PPO_CLIP_EPS = 0.2
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_GAE_LAMBDA = 0.95
PPO_GAMMA = 0.99
PPO_KL_COEF = 0.01  # KL penalty against reference (foundation) policy
PPO_MAX_GRAD_NORM = 1.0
RLHF_POLICY_SAVE_PATH = "checkpoints/rlhf_policy.pt"

# ─── DPO ──────────────────────────────────────────────────────────────────────
DPO_LR = 1e-5
DPO_EPOCHS = 20
DPO_BETA = 0.1  # KL-divergence regularisation strength
DPO_SAVE_PATH = "checkpoints/dpo_policy.pt"

# ─── Evaluation ───────────────────────────────────────────────────────────────
EVAL_EPISODES = 100
EVAL_SEEDS = [0, 1, 2, 3, 4]  # Multi-seed evaluation for variance estimates
RESULTS_PATH = "results/eval_results.json"
