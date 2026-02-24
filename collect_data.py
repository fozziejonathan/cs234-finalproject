"""
collect_data.py
---------------
Phase 1: Trajectory Generation

Rolls out episodes in a MetaWorld environment using the task's built-in
SCRIPTED EXPERT POLICY as the baseline (π_base).  This is the recommended
baseline for MuJoCo/MetaWorld experiments — it is near-optimal, ships with
the metaworld library, and requires zero extra infrastructure (unlike GR00T
or Pi0.5 which need GPU servers / camera observations).

Additive Gaussian exploration noise is injected on top of the scripted actions
to produce behavioural diversity needed for preference ranking.  Without noise,
all trajectories would look similar and preference labels would be uninformative.

Output
------
data/raw_trajectories.h5
    trajectory_0/
        observations  (T, 39)  float32
        actions       (T,  4)  float32
        rewards       (T,)     float32
        dones         (T,)     bool
    trajectory_1/ ...

Usage
-----
    python collect_data.py
    python collect_data.py --env push-v3 --n 3000 --noise 0.15
"""

import argparse
import os
import numpy as np
import h5py  # pip install h5py
import metaworld
import metaworld.policies as mw_policies
from config import (
    ENV_NAME, NUM_TRAJECTORIES, MAX_EPISODE_STEPS,
    EXPLORATION_NOISE, RAW_DATA_PATH
)

# ─── Scripted Policy Lookup ────────────────────────────────────────────────────
# MetaWorld's policy class names vary slightly across versions.
# We use a name-fragment map + dynamic getattr lookup so this is version-robust.
# Format: task-name → substring that uniquely identifies the policy class name.
_POLICY_NAME_FRAGMENTS = {
    "reach-v3": "Reach",
    "push-v3": "Push",
    "pick-place-v3": "PickPlace",
    "door-open-v3": "Door",
    "drawer-open-v3": "DrawerOpen",
    "drawer-close-v3": "DrawerClose",
    "button-press-topdown-v3": "ButtonPressTopdown",
    "peg-insert-side-v3": "PegInsertionSide",
    "window-open-v3": "WindowOpen",
    "window-close-v3": "WindowClose",
}


def _build_policy_map():
    """
    Discover available Sawyer scripted policies dynamically from metaworld.policies.
    Matches each task to the first class whose name contains the expected fragment.
    This works across metaworld 2.x and 3.x regardless of exact class naming.
    """
    all_policy_names = [name for name in dir(mw_policies) if "Policy" in name]
    policy_map = {}
    for task, fragment in _POLICY_NAME_FRAGMENTS.items():
        matches = [n for n in all_policy_names if fragment in n]
        if matches:
            policy_map[task] = getattr(mw_policies, matches[0])
        else:
            print(f"[WARN] No scripted policy found for '{task}' "
                  f"(searched for fragment '{fragment}' in {all_policy_names[:5]}...)")
    return policy_map


SCRIPTED_POLICY_MAP = _build_policy_map()


def make_env(env_name: str):
    """
    Instantiate a single Meta-World ML1 environment and randomly sample a task.
    Returns (env, task_list) so callers can re-sample tasks for diversity.
    """
    ml1 = metaworld.ML1(env_name)
    env = ml1.train_classes[env_name]()
    tasks = ml1.train_tasks
    return env, tasks


def get_scripted_policy(env_name: str):
    """
    Return an instance of the MetaWorld scripted policy for this task.
    Falls back to a random policy if the task is not in our map.
    """
    if env_name in SCRIPTED_POLICY_MAP:
        policy_cls = SCRIPTED_POLICY_MAP[env_name]
        print(f"[INFO] Using scripted expert policy: {policy_cls.__name__}")
        return policy_cls()
    else:
        print(f"[WARN] No scripted policy found for '{env_name}'. "
              f"Falling back to random actions. Add it to SCRIPTED_POLICY_MAP.")
        return None


class TrajectoryCollector:
    """
    Collects trajectories by running π_base + exploration noise in MetaWorld.

    Parameters
    ----------
    env_name          : MetaWorld task string, e.g. "reach-v3"
    num_trajectories  : How many episodes to record
    max_steps         : Episode time-limit
    exploration_noise : Std-dev of Gaussian noise added to scripted actions
    """

    def __init__(
            self,
            env_name: str = ENV_NAME,
            num_trajectories: int = NUM_TRAJECTORIES,
            max_steps: int = MAX_EPISODE_STEPS,
            exploration_noise: float = EXPLORATION_NOISE,
    ):
        self.env_name = env_name
        self.num_trajectories = num_trajectories
        self.max_steps = max_steps
        self.exploration_noise = exploration_noise

        self.env, self.tasks = make_env(env_name)
        self.scripted_policy = get_scripted_policy(env_name)

    def _get_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute π_base(obs) and add exploration noise.

        The scripted policy returns a 4-D action in [-1, 1].
        We add N(0, σ²_explore) and clip back to action bounds.
        """
        if self.scripted_policy is not None:
            base_action = self.scripted_policy.get_action(obs)
        else:
            base_action = self.env.action_space.sample()

        noise = np.random.normal(0.0, self.exploration_noise, size=base_action.shape)
        noisy_action = np.clip(
            base_action + noise,
            self.env.action_space.low,
            self.env.action_space.high,
        )
        return noisy_action.astype(np.float32)

    def collect_and_serialize(self, save_path: str = RAW_DATA_PATH) -> None:
        """
        Run episodes and write all transitions to an HDF5 file.

        Each trajectory is stored in its own HDF5 group so large datasets
        can be streamed without loading everything into RAM.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[INFO] Collecting {self.num_trajectories} trajectories → {save_path}")

        episode_returns = []

        with h5py.File(save_path, "w") as f:
            for i in range(self.num_trajectories):
                # Rotate through available tasks for goal diversity
                task = self.tasks[i % len(self.tasks)]
                self.env.set_task(task)

                obs, _info = self.env.reset()
                traj_obs, traj_acts, traj_rewards, traj_dones = [], [], [], []

                for _step in range(self.max_steps):
                    action = self._get_action(obs)
                    next_obs, reward, terminated, truncated, _info = self.env.step(action)
                    done = terminated or truncated

                    traj_obs.append(obs.astype(np.float32))
                    traj_acts.append(action.astype(np.float32))
                    traj_rewards.append(float(reward))
                    traj_dones.append(bool(done))

                    obs = next_obs
                    if done:
                        break

                # Serialize to HDF5
                grp = f.create_group(f"trajectory_{i}")
                grp.create_dataset("observations", data=np.array(traj_obs, dtype=np.float32))
                grp.create_dataset("actions", data=np.array(traj_acts, dtype=np.float32))
                grp.create_dataset("rewards", data=np.array(traj_rewards, dtype=np.float32))
                grp.create_dataset("dones", data=np.array(traj_dones, dtype=bool))

                ep_return = sum(traj_rewards)
                episode_returns.append(ep_return)

                if (i + 1) % 200 == 0:
                    mean_ret = np.mean(episode_returns[-200:])
                    print(f"  [{i + 1:>5}/{self.num_trajectories}]  "
                          f"last-200 mean return: {mean_ret:.2f}")

        print(f"[INFO] Done. Mean return across all episodes: "
              f"{np.mean(episode_returns):.2f}  |  "
              f"Std: {np.std(episode_returns):.2f}")
        print(f"[INFO] Return range: [{min(episode_returns):.2f}, {max(episode_returns):.2f}]")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect MetaWorld trajectories.")
    parser.add_argument("--env", default=ENV_NAME, help="MetaWorld task name")
    parser.add_argument("--n", type=int, default=NUM_TRAJECTORIES, help="# trajectories")
    parser.add_argument("--noise", type=float, default=EXPLORATION_NOISE, help="Exploration noise σ")
    parser.add_argument("--out", default=RAW_DATA_PATH, help="Output HDF5 path")
    args = parser.parse_args()

    collector = TrajectoryCollector(
        env_name=args.env,
        num_trajectories=args.n,
        exploration_noise=args.noise,
    )
    collector.collect_and_serialize(save_path=args.out)
