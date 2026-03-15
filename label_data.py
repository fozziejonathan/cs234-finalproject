"""
label_data.py
-------------
Builds preference pairs from raw trajectories.

Labels:
  success vs. failure   → label 1.0
  success vs. success   → label 1.0 (higher return wins)
  failure vs. failure   → label 0.5 (no preference)

Output: pkl with {"train": [...], "val": [...]}
"""

import argparse
import os
import pickle
import numpy as np
import h5py
from config import RAW_DATA_PATH, PREF_PAIRS_RATIO, PPO_GAMMA


def load_trajectories(path):
    trajs = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            g = f[key]
            rewards = g["rewards"][:]
            discounts = PPO_GAMMA ** np.arange(len(rewards))
            trajs.append({
                "observations": g["observations"][:],
                "actions": g["actions"][:],
                "true_return": float(np.dot(discounts, rewards)),
                "ep_return": float(rewards.sum()),
                "success": bool(g.attrs.get("success", False)),
            })

    returns = [t["true_return"] for t in trajs]
    n_success = sum(t["success"] for t in trajs)
    print(f"[INFO] Loaded {len(trajs)} trajectories from {path}")
    print(f"[INFO] Return — mean: {np.mean(returns):.2f}  "
          f"std: {np.std(returns):.2f}  "
          f"range: [{np.min(returns):.2f}, {np.max(returns):.2f}]")
    print(f"[INFO] Dataset success rate: {n_success / len(trajs):.2%} "
          f"({n_success}/{len(trajs)})")
    return trajs


def make_pairs(trajs, n_pairs, train_frac=0.9):
    n = len(trajs)
    pairs = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, size=2, replace=False)
        a, b = trajs[i], trajs[j]

        # Case 1: success vs failure — hard preference for success
        # Case 2: both success — hard preference for higher return
        # Case 3: both failure — soft label, no preference signal
        if a["success"] and not b["success"]:
            chosen, rejected, label = a, b, 1.0
        elif b["success"] and not a["success"]:
            chosen, rejected, label = b, a, 1.0
        elif a["success"] and b["success"]:
            # Both succeeded: prefer higher return (more efficient)
            if a["true_return"] >= b["true_return"]:
                chosen, rejected = a, b
            else:
                chosen, rejected = b, a
            label = 1.0
        else:
            # Both failed: no preference signal
            chosen, rejected = (a, b) if a["true_return"] >= b["true_return"] else (b, a)
            label = 0.5

        pairs.append({
            "chosen": {
                "observations": chosen["observations"],
                "actions": chosen["actions"],
                "true_return": chosen["true_return"],
                "success": chosen["success"],
            },
            "rejected": {
                "observations": rejected["observations"],
                "actions": rejected["actions"],
                "true_return": rejected["true_return"],
                "success": rejected["success"],
            },
            "margin": chosen["true_return"] - rejected["true_return"],
            "label": label,  # 1.0 = hard preference, 0.5 = no preference
        })

    np.random.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    train, val = pairs[:split], pairs[split:]

    margins = [p["margin"] for p in pairs]
    n_hard = sum(1 for p in pairs if p["label"] == 1.0)
    n_soft = sum(1 for p in pairs if p["label"] == 0.5)
    n_s_vs_f = sum(1 for p in pairs if p["chosen"]["success"] and not p["rejected"]["success"])
    n_s_vs_s = sum(1 for p in pairs if p["chosen"]["success"] and p["rejected"]["success"])
    print(f"[INFO] {n_pairs} pairs — train: {len(train)}  val: {len(val)}")
    print(f"[INFO] Case breakdown — "
          f"success>failure: {n_s_vs_f} ({n_s_vs_f / len(pairs):.1%})  "
          f"success>success: {n_s_vs_s} ({n_s_vs_s / len(pairs):.1%})  "
          f"fail~fail (soft): {n_soft} ({n_soft / len(pairs):.1%})")
    print(f"[INFO] Hard labels: {n_hard}  Soft labels: {n_soft}")
    print(f"[INFO] Mean |margin|: {np.mean(np.abs(margins)):.2f}  "
          f"range: [{np.min(margins):.2f}, {np.max(margins):.2f}]")
    n_chosen_success = sum(p["chosen"]["success"] for p in pairs)
    print(f"[INFO] Chosen trajectory success rate: {n_chosen_success / len(pairs):.2%}")
    return train, val


def run(raw=None, out=None, n_pairs=None):
    if raw is None:
        raise ValueError("--raw must be provided (e.g. data/raw_trajectories_reach-v3_20seeds.h5)")

    trajs = load_trajectories(raw)

    if n_pairs is None:
        n_pairs = len(trajs) * PREF_PAIRS_RATIO
        print(f"[INFO] n_pairs not specified — using {PREF_PAIRS_RATIO}x trajectories = {n_pairs}")

    if out is None:
        stem = os.path.splitext(os.path.basename(raw))[0]  # e.g. raw_trajectories_reach-v3_20seeds
        stem = stem.replace("raw_trajectories_", "")        # e.g. reach-v3_20seeds
        out = f"data/preferences_{stem}_{n_pairs}.pkl"

    train, val = make_pairs(trajs, n_pairs)
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"train": train, "val": val}, f)
    print(f"[INFO] Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--pairs", type=int, default=None,
                        help="Number of preference pairs (default: 10x trajectory count)")
    args = parser.parse_args()
    run(args.raw, args.out, args.pairs)
