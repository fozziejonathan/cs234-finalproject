# Preference-Based Learning for Continuous Control
CS 234 Final Project — Jonathan Lu

We compare RLHF and DPO for fine-tuning robot manipulation policies in Meta-World, starting from a BC baseline trained on expert demos. We also implement DPOP, a DPO variant that adds a regularization penalty to prevent likelihood collapse.

## Setup

```bash
pip install -r requirements.txt
```

Requires MuJoCo rendering support (`mujoco==3.5.0`, `gymnasium==1.2.3`).

## Pipeline

Run scripts in this order:

```bash
# 1. Collect expert trajectories
python collect_data.py --env peg-insert-side-v3 --n_seeds 50

# 2. Train BC baseline
python train_bc.py --h5 data/raw_trajectories_peg-insert-side-v3_50seeds.h5

# 3. Collect BC policy rollouts (for preference data)
python collect_data.py --env peg-insert-side-v3 --n_seeds 30 \
  --bc_ckpt checkpoints/bc_policy_peg-insert-side-v3.pt --n_rollouts_per_task 3

# 4. Label preference pairs
python label_data.py \
  --raw data/raw_trajectories_peg-insert-side-v3_30seeds_3rpt_bc.h5 --pairs 45000

# 5. Fine-tune (pick one or all)
python train_rlhf.py --data data/preferences_*.pkl --bc checkpoints/bc_policy_*.pt --obs_norm checkpoints/obs_normalizer_*.npz
python train_dpo.py  --data data/preferences_*.pkl --bc checkpoints/bc_policy_*.pt --obs_norm checkpoints/obs_normalizer_*.npz
python train_dpo.py  --data data/preferences_*.pkl --bc checkpoints/bc_policy_*.pt --obs_norm checkpoints/obs_normalizer_*.npz --dpop

# 6. Evaluate
python evaluate.py --ckpt checkpoints/dpo_policy_peg-insert-side-v3.pt \
  --obs_norm checkpoints/obs_normalizer_peg-insert-side-v3.npz --n 50 --n_seeds 24
```

## Methods

**BC** — supervised learning on successful expert trajectories (NLL loss on expert actions).

**RLHF** — trains a Bradley-Terry reward model on preference pairs first. Freeze the RM. Then does Reward-Weighted Regression (RWR) to fine-tune the policy without any RL rollouts. Basically supervised learning with weights.

**DPO** — directly optimizes the policy on preference pairs using the implicit reward formulation. Skips the reward model.

**DPOP** — DPO with an extra penalty that prevents the chosen trajectory's log-prob from falling below the reference policy. Does significantly better than DPO, but on peg-insert-side performs worse than RLHF, suggesting the richness of learned RL is better for robotic manipulation tasks. To be investigated

## Results

Vanilla DPO runs into a failure mode where preference accuracy goes up but task success rate sinks to 0%. The log-prob of chosen trajectories collapses to very negative values (around -700 for peg insert), meaning the policy drifts far from the BC reference in a way that doesn't transfer to task performance. DPOP largely fixes this. BC ends up being a strong baseline (~82% success on `peg-insert-side-v3`).

## Repo structure

```
config.py               shared hyperparameters
collect_data.py         trajectory collection
label_data.py           preference pair generation
train_bc.py             BC training
train_rlhf.py           reward model + RWR
train_dpo.py            DPO / DPOP fine-tuning
evaluate.py             evaluation
plot_training.py        training curves
render_trajectory.py    rollout videos
data/                   trajectories and preference datasets
checkpoints/            saved policies, reward models, normalizers
figures/
```
