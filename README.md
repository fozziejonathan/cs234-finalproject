# cs234-finalproject
This is our CS 234 final project.

## Preference Learning Baselines (Robosuite / Meta-World / LIBERO)

This repo includes RLHF and DPO baselines for preference-based policy learning with benchmark-aware training support.

- `train_rlhf.py`: learns a reward model from trajectory preferences, then optimizes the policy with PPO.
- `train_dpo.py`: trains a reference policy with behavior cloning, then applies direct preference optimization.

Both pipelines currently use **synthetic preference labels** generated from true episodic returns, with optional label noise via `--preference-noise`.

### Install

```bash
pip install -r requirements.txt
```

### Run RLHF (single benchmark)

```bash
python train_rlhf.py --benchmark robosuite --env-name Lift --robot Panda --iterations 20
```

### Run DPO (single benchmark)

```bash
python train_dpo.py --benchmark robosuite --env-name Lift --robot Panda --iterations 20
```

### Save logs + metric images (Robosuite)

```bash
python train_rlhf.py \
  --benchmark robosuite \
  --env-name Lift \
  --robot Panda \
  --iterations 20 \
  --output checkpoints/rlhf_robosuite_lift.pt \
  --log-file logs/rlhf_lift.log \
  --metrics-output logs/rlhf_lift_metrics.csv \
  --plot-output logs/rlhf_lift_metrics.png
```

```bash
python train_dpo.py \
  --benchmark robosuite \
  --env-name Lift \
  --robot Panda \
  --iterations 20 \
  --output checkpoints/dpo_robosuite_lift.pt \
  --log-file logs/dpo_lift.log \
  --metrics-output logs/dpo_lift_metrics.csv \
  --plot-output logs/dpo_lift_metrics.png
```

### Save best-checkpoint snapshots during training

Use this when final policy quality drifts late in training and you want the best iteration for videos / eval.

```bash
python train_rlhf.py \
  --benchmark robosuite \
  --env-name Lift \
  --robot Panda \
  --iterations 80 \
  --output checkpoints/rlhf_lift_final.pt \
  --save-best-checkpoint \
  --best-metric eval_success \
  --best-checkpoint-path checkpoints/rlhf_lift_best.pt \
  --log-file logs/rlhf_lift.log \
  --metrics-output logs/rlhf_lift_metrics.csv \
  --plot-output logs/rlhf_lift_metrics.png
```

```bash
python train_dpo.py \
  --benchmark robosuite \
  --env-name Lift \
  --robot Panda \
  --iterations 80 \
  --output checkpoints/dpo_lift_final.pt \
  --save-best-checkpoint \
  --best-metric eval_success \
  --best-checkpoint-path checkpoints/dpo_lift_best.pt \
  --log-file logs/dpo_lift.log \
  --metrics-output logs/dpo_lift_metrics.csv \
  --plot-output logs/dpo_lift_metrics.png
```

### Record rollout videos from checkpoints

```bash
python record_robosuite_rollout.py \
  --checkpoint checkpoints/rlhf_robosuite_lift.pt \
  --output videos/rlhf_lift.gif \
  --episodes 1 \
  --camera-name agentview \
  --metrics-json videos/rlhf_lift_metrics.json
```

```bash
python record_robosuite_rollout.py \
  --checkpoint checkpoints/dpo_robosuite_lift.pt \
  --output videos/dpo_lift.mp4 \
  --episodes 3 \
  --camera-name agentview \
  --metrics-json videos/dpo_lift_metrics.json
```

### Run benchmark comparisons (Milestone-style)

```bash
# RLHF on Robosuite + Meta-World + LIBERO with one command
python train_rlhf.py \
  --compare-benchmarks robosuite,metaworld,libero \
  --env-name Lift \
  --metaworld-env-name reach-v3 \
  --libero-suite libero_object \
  --libero-task-ids 0,3,4,7 \
  --comparison-seeds 0,1,2 \
  --comparison-output checkpoints/rlhf_comparison.md
```

```bash
# DPO on Robosuite + Meta-World + LIBERO with one command
python train_dpo.py \
  --compare-benchmarks robosuite,metaworld,libero \
  --env-name Lift \
  --metaworld-env-name reach-v3 \
  --libero-suite libero_object \
  --libero-task-ids 0,3,4,7 \
  --comparison-seeds 0,1,2 \
  --comparison-output checkpoints/dpo_comparison.md
```

### Helpful flags

- `--benchmark {robosuite,metaworld,libero}`: choose a single benchmark.
- `--compare-benchmarks robosuite,metaworld,libero`: run cross-benchmark comparisons in one execution.
- `--comparison-seeds 0,1,2`: run multiple seeds for comparison stability.
- `--comparison-output checkpoints/<name>.md`: write markdown comparison table with final metrics.
- `--log-file logs/<name>.log`: save full stdout/stderr training logs.
- `--metrics-output logs/<name>.csv`: save per-iteration metrics table.
- `--plot-output logs/<name>.png`: save per-iteration metric curves as an image.
- `--save-best-checkpoint`: write checkpoint snapshots whenever eval metric improves.
- `--best-checkpoint-path checkpoints/<name>.pt`: path for best snapshot (default: `<output>_best.pt`).
- `--best-metric {eval_success,eval_return}`: choose snapshot criterion.
- `--metaworld-env-name reach-v3`: Meta-World task in comparison mode.
- `--libero-suite libero_object --libero-task-ids 0,3,4,7`: sweep milestone LIBERO task ids in one run.
- `--preference-noise 0.1`: flips 10% of pairwise labels.
- `--sparse-reward`: uses sparse task reward instead of dense shaping.
- `--output checkpoints/<name>.pt`: where to save the final checkpoint.
- `--robosuite-log-level WARNING`: suppresses Robosuite INFO spam (default).
- `--hard-reset`: enable Robosuite hard reset each episode (disabled by default for faster training).
- `--pref-buffer-size 512` (DPO): use a larger rolling preference buffer for more stable pairwise updates.

### Optional dependencies for non-Robosuite benchmarks

Meta-World and LIBERO are optional and not pinned in `requirements.txt`. Install them separately before running those benchmarks:

- Meta-World: `pip install metaworld`
- LIBERO: install LIBERO and its environment dependencies, then pass valid LIBERO env identifiers/suite settings.
