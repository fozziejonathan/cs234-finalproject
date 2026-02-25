"""
label_data.py
-------------
Phase 2: Preference Dataset

Reads trajectories from collect_data.py and builds preference pairs using
actual environment returns — no synthetic human noise model needed. Cleaner
signal for both RLHF and DPO since the return is the ground truth.

Each pair stores success flag so BC can filter to successful demos only.

Output: data/preference_dataset.pkl
    {
      "train": [ {chosen: {...}, rejected: {...}, margin: float}, ... ],
      "val":   [ ... ]
    }
"""

import argparse
import os
import pickle
import numpy as np
import h5py
from config import RAW_DATA_PATH, NUM_PREF_PAIRS, PREF_DATASET_PATH, PPO_GAMMA


def load_trajectories(path):
    trajs = []
    with h5py.File(path, "r") as f:
        for key in f.keys():
            g = f[key]
            rewards = g["rewards"][:]
            discounts = PPO_GAMMA ** np.arange(len(rewards))
            trajs.append({
                "observations": g["observations"][:],
                "actions":      g["actions"][:],
                "true_return":  float(np.dot(discounts, rewards)),
                "ep_return":    float(rewards.sum()),
                "success":      bool(g.attrs.get("success", False)),
            })

    returns   = [t["true_return"] for t in trajs]
    n_success = sum(t["success"] for t in trajs)
    print(f"[INFO] Loaded {len(trajs)} trajectories from {path}")
    print(f"[INFO] Return — mean: {np.mean(returns):.2f}  "
          f"std: {np.std(returns):.2f}  "
          f"range: [{np.min(returns):.2f}, {np.max(returns):.2f}]")
    print(f"[INFO] Dataset success rate: {n_success/len(trajs):.2%} "
          f"({n_success}/{len(trajs)})")
    return trajs


def make_pairs(trajs, n_pairs, train_frac=0.9):
    n = len(trajs)
    pairs = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n, size=2, replace=False)
        a, b = trajs[i], trajs[j]
        # Pure return comparison — no noise
        if a["true_return"] >= b["true_return"]:
            chosen, rejected = a, b
        else:
            chosen, rejected = b, a

        pairs.append({
            "chosen": {
                "observations": chosen["observations"],
                "actions":      chosen["actions"],
                "true_return":  chosen["true_return"],
                "success":      chosen["success"],
            },
            "rejected": {
                "observations": rejected["observations"],
                "actions":      rejected["actions"],
                "true_return":  rejected["true_return"],
                "success":      rejected["success"],
            },
            "margin": chosen["true_return"] - rejected["true_return"],
        })

    np.random.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    train, val = pairs[:split], pairs[split:]

    margins = [p["margin"] for p in pairs]
    print(f"[INFO] {n_pairs} pairs — train: {len(train)}  val: {len(val)}")
    print(f"[INFO] Mean |margin|: {np.mean(np.abs(margins)):.2f}  "
          f"range: [{np.min(margins):.2f}, {np.max(margins):.2f}]")
    n_chosen_success = sum(p["chosen"]["success"] for p in pairs)
    print(f"[INFO] Chosen trajectory success rate: {n_chosen_success/len(pairs):.2%}")
    return train, val


def run(raw=RAW_DATA_PATH, out=PREF_DATASET_PATH,
        n_pairs=NUM_PREF_PAIRS):
    trajs = load_trajectories(raw)
    train, val = make_pairs(trajs, n_pairs)
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump({"train": train, "val": val}, f)
    print(f"[INFO] Saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",    default=RAW_DATA_PATH)
    parser.add_argument("--out",    default=PREF_DATASET_PATH)
    parser.add_argument("--pairs",  type=int, default=NUM_PREF_PAIRS)
    args = parser.parse_args()
    run(args.raw, args.out, args.pairs)