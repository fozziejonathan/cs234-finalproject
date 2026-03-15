"""
config.py
---------
Central configuration for the RLHF vs DPO continuous control experiment.
All scripts import from here so hyperparameters are changed in one place.
"""

# ─── Environment ──────────────────────────────────────────────────────────────
ENV_NAME = "peg-insert-side-v3"  # MetaWorld task name
OBS_DIM = 39  # MetaWorld state space dimensionality
ACT_DIM = 4  # 3-D Cartesian velocity + gripper
MAX_EPISODE_STEPS = 150  # Maximum steps per episode

# ─── Data Collection ──────────────────────────────────────────────────────────
RAW_DATA_PATH = "data/raw_trajectories_{env}_{n}.h5"

# ─── Preference Labeling ──────────────────────────────────────────────────────
PREF_PAIRS_RATIO = 10  # preference pairs generated per collected trajectory

# ─── Shared Network Architecture ──────────────────────────────────────────────
HIDDEN_DIM = 256

# ─── RLHF (Reward Model + PPO) ────────────────────────────────────────────────
PPO_GAMMA = 0.99  # used by label_data.py for discounted returns

# ─── DPO ──────────────────────────────────────────────────────────────────────
DPO_LR = 1e-5
DPO_EPOCHS = 20
DPO_BETA = 0.01
