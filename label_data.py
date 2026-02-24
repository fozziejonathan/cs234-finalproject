"""
label_data.py
-------------
Phase 2: Preference Synthesis

Reads the raw HDF5 trajectories produced by collect_data.py and generates
pairwise preference labels to serve as training data for both RLHF and DPO.

Theory
------
True episodic return of trajectory τ_i:
    R(τ_i) = Σ_t γ^t · r(s_t, a_t)          (γ = discount factor)

Simulated human perception with noise (models bounded rationality):
    R̃(τ_i) = R(τ_i) + ε_i,   ε_i ~ N(0, σ²_human_error)

Preference label for pair (τ_A, τ_B):
    y = A  if  R̃(τ_A) > R̃(τ_B)
    y = B  otherwise

Modulating σ_human_error:
    σ = 0.0  →  perfect oracle (reduces to standard RL)
    σ = 1–2  →  realistic noisy human
    σ >> 1   →  near-random labels (stress-tests robustness)

Output
------
data/preference_dataset.pkl
    List of dicts:
        {
          "chosen":   {"observations": (T,39), "actions": (T,4), "true_return": float},
          "rejected": {"observations": (T,39), "actions": (T,4), "true_return": float},
          "margin":   float   # R(τ_chosen) - R(τ_rejected), useful for analysis
        }

Usage
-----
    python label_data.py
    python label_data.py --noise 0.5 --pairs 8000
"""

import argparse
import os
import pickle
import numpy as np
import h5py
from typing import List, Dict
from config import (
    RAW_DATA_PATH, NUM_PREF_PAIRS, HUMAN_ERROR_NOISE,
    PREF_DATASET_PATH, PPO_GAMMA
)


class SyntheticPreferenceGenerator:
    """
    Loads serialized trajectories and synthesizes Bradley-Terry preference pairs.

    Parameters
    ----------
    raw_data_path   : Path to the HDF5 file from collect_data.py
    noise_std       : σ_human_error — noise on perceived trajectory return
    gamma           : Discount factor for computing episodic return
    """

    def __init__(
            self,
            raw_data_path: str = RAW_DATA_PATH,
            noise_std: float = HUMAN_ERROR_NOISE,
            gamma: float = PPO_GAMMA,
    ):
        self.raw_data_path = raw_data_path
        self.noise_std = noise_std
        self.gamma = gamma

    def _discounted_return(self, rewards: np.ndarray) -> float:
        """Compute discounted cumulative return: Σ_t γ^t r_t."""
        discounts = self.gamma ** np.arange(len(rewards))
        return float(np.dot(discounts, rewards))

    def _extract_trajectories(self) -> List[Dict]:
        """Load all trajectories from HDF5 and compute their true returns."""
        trajectories = []
        with h5py.File(self.raw_data_path, "r") as f:
            for key in f.keys():
                grp = f[key]
                rewards = grp["rewards"][:]
                traj = {
                    "observations": grp["observations"][:],  # (T, 39)
                    "actions": grp["actions"][:],  # (T,  4)
                    "true_return": self._discounted_return(rewards),
                    "undiscounted_return": float(rewards.sum()),
                    "length": len(rewards),
                }
                trajectories.append(traj)

        print(f"[INFO] Loaded {len(trajectories)} trajectories from {self.raw_data_path}")
        returns = [t["true_return"] for t in trajectories]
        print(f"[INFO] Return stats — mean: {np.mean(returns):.2f}  "
              f"std: {np.std(returns):.2f}  "
              f"range: [{min(returns):.2f}, {max(returns):.2f}]")
        return trajectories

    def synthesize_datasets(
            self,
            num_pairs: int = NUM_PREF_PAIRS,
            save_path: str = PREF_DATASET_PATH,
            train_frac: float = 0.9,
    ) -> None:
        """
        Sample trajectory pairs, inject perception noise, assign labels, and
        split into train / validation sets.

        Parameters
        ----------
        num_pairs   : Total preference pairs to generate
        save_path   : Output .pkl path
        train_frac  : Fraction of pairs kept for training (rest = validation)
        """
        trajectories = self._extract_trajectories()
        n = len(trajectories)
        assert n >= 2, "Need at least 2 trajectories to form pairs."

        preference_pairs = []
        flip_count = 0  # # of times noise flipped the "true" preference

        for _ in range(num_pairs):
            idx_a, idx_b = np.random.choice(n, size=2, replace=False)
            traj_a = trajectories[idx_a]
            traj_b = trajectories[idx_b]

            # Inject simulated human perception noise
            perceived_a = traj_a["true_return"] + np.random.normal(0.0, self.noise_std)
            perceived_b = traj_b["true_return"] + np.random.normal(0.0, self.noise_std)

            # Ground-truth preference (for diagnostics)
            true_winner = "A" if traj_a["true_return"] > traj_b["true_return"] else "B"

            # Noisy preference label
            if perceived_a > perceived_b:
                chosen, rejected = traj_a, traj_b
                noisy_winner = "A"
            else:
                chosen, rejected = traj_b, traj_a
                noisy_winner = "B"

            if true_winner != noisy_winner:
                flip_count += 1

            preference_pairs.append({
                "chosen": {k: chosen[k] for k in ("observations", "actions", "true_return")},
                "rejected": {k: rejected[k] for k in ("observations", "actions", "true_return")},
                "margin": chosen["true_return"] - rejected["true_return"],
            })

        # Shuffle and split
        np.random.shuffle(preference_pairs)
        split_idx = int(len(preference_pairs) * train_frac)
        train_data = preference_pairs[:split_idx]
        val_data = preference_pairs[split_idx:]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump({"train": train_data, "val": val_data}, f)

        label_noise_rate = flip_count / num_pairs * 100
        margins = [p["margin"] for p in preference_pairs]
        print(f"[INFO] Synthesized {num_pairs} pairs  "
              f"(train: {len(train_data)}, val: {len(val_data)})")
        print(f"[INFO] Label noise rate: {label_noise_rate:.1f}%  "
              f"(σ_human_error = {self.noise_std})")
        print(f"[INFO] Mean |margin|: {np.mean(np.abs(margins)):.2f}  "
              f"— larger = easier preference task")
        print(f"[INFO] Saved → {save_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthesize preference labels.")
    parser.add_argument("--raw", default=RAW_DATA_PATH, help="Input HDF5 path")
    parser.add_argument("--out", default=PREF_DATASET_PATH, help="Output .pkl path")
    parser.add_argument("--pairs", type=int, default=NUM_PREF_PAIRS, help="# preference pairs")
    parser.add_argument("--noise", type=float, default=HUMAN_ERROR_NOISE,
                        help="Human error noise σ")
    args = parser.parse_args()

    gen = SyntheticPreferenceGenerator(
        raw_data_path=args.raw,
        noise_std=args.noise,
    )
    gen.synthesize_datasets(num_pairs=args.pairs, save_path=args.out)
