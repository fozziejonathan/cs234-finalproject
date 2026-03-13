"""
render_comparison.py
--------------------
Roll out multiple policy checkpoints in MetaWorld and produce:
  1. A side-by-side keyframe figure (like the paper figure)
  2. Per-policy videos: {name}_success.mp4, {name}_failure.mp4
  3. A single side-by-side comparison video of all policies

Usage
-----
    python render_comparison.py \
        --ckpts bc=checkpoints/bc_policy_bin-picking-v3.pt \
                dpo=checkpoints/dpo_policy_bin-picking-v3.pt \
                dpop=checkpoints/dpop_policy_bin-picking-v3.pt \
        --obs_norm checkpoints/obs_normalizer_bin-picking-v3.npz \
        --n_frames 5 \
        --n_eval 20 \
        --save_videos \
        --out figures/policy_comparison.png

Requirements
------------
    pip install imageio imageio-ffmpeg opencv-python matplotlib numpy torch metaworld
"""

import argparse
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import metaworld

from config import ENV_NAME, MAX_EPISODE_STEPS
from train_bc import GaussianPolicy, ObsNormalizer, DEVICE


# =============================================================================
# Policy loading
# =============================================================================

def load_policy(ckpt_path, obs_norm_path):
    obs_norm = ObsNormalizer()
    obs_norm.load(obs_norm_path)
    policy = GaussianPolicy().to(DEVICE)
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()
    return policy, obs_norm


# =============================================================================
# Environment rollout
# =============================================================================

def rollout_single(policy, obs_norm, env, task, deterministic=True):
    """Run one episode. Returns (frames, success, ep_return)."""
    env.set_task(task)
    obs, _ = env.reset()

    frames = []
    ep_return = 0.0
    success = False
    consec = 0

    with torch.no_grad():
        for _ in range(MAX_EPISODE_STEPS):
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())

            obs_n = obs_norm.normalize(obs.astype(np.float32))
            obs_t = torch.FloatTensor(obs_n).unsqueeze(0).to(DEVICE)
            action = policy.get_action(obs_t, deterministic=deterministic)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward

            if info.get("success", False):
                consec += 1
            else:
                consec = 0
            if not success and consec >= 5:
                success = True

            if terminated or truncated:
                break

    return frames, success, ep_return


def eval_policy(policy, obs_norm, env_name, n_eval, seed=0):
    """
    Evaluate over n_eval episodes.
    Returns (success_rate, list of (frames, success)).
    """
    mt1 = metaworld.MT1(env_name, seed=seed)
    env = mt1.train_classes[env_name](render_mode="rgb_array")
    tasks = mt1.train_tasks

    successes = []
    all_episodes = []

    for i in range(n_eval):
        task = tasks[i % len(tasks)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames, success, ep_return = rollout_single(
                policy, obs_norm, env, task, deterministic=True
            )
        successes.append(float(success))
        all_episodes.append((frames, success))
        print(f"  episode {i+1:>3}/{n_eval}  success={success}  return={ep_return:.1f}")

    env.close()
    return float(np.mean(successes)), all_episodes


def find_shared_task(loaded, env_name, seed=0, max_tries=50, skip_names=None):
    """
    Search the task list for the first task where all non-skipped policies
    succeed. Falls back to the task where the most non-skipped policies succeed.

    loaded:      list of (name, policy, obs_norm)
    skip_names:  set of lowercase policy names excluded from the success
                 criterion (e.g. {"dpo"} when DPO is known to fail).
                 All policies are still rolled out on the chosen task.
    Returns the chosen task object.
    """
    if skip_names is None:
        skip_names = set()

    mt1 = metaworld.MT1(env_name, seed=seed)
    tasks = mt1.train_tasks
    envs = [mt1.train_classes[env_name](render_mode="rgb_array") for _ in loaded]
    scored = [(name, policy, obs_norm)
              for name, policy, obs_norm in loaded
              if name.lower() not in skip_names]

    best_idx, best_count, chosen_idx = 0, 0, None
    print(f"\n[INFO] Searching for shared task "
          f"(scoring {len(scored)}/{len(loaded)} policies, up to {max_tries} candidates)...")

    for task_idx in range(min(max_tries, len(tasks))):
        task = tasks[task_idx]
        count = 0
        for (name, policy, obs_norm), env in zip(loaded, envs):
            if name.lower() in skip_names:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, success, _ = rollout_single(policy, obs_norm, env, task)
            if success:
                count += 1
        print(f"  task {task_idx:>3}: {count}/{len(scored)} succeeded")
        if count > best_count:
            best_count = count
            best_idx = task_idx
        if count == len(scored):
            chosen_idx = task_idx
            break

    for env in envs:
        env.close()

    idx = chosen_idx if chosen_idx is not None else best_idx
    print(f"  -> Using task {idx} ({best_count}/{len(scored)} scored policies succeed)\n")
    return tasks[idx]


def pick_keyframes(frames, n_frames):
    """Pick n_frames evenly spaced frames."""
    if len(frames) == 0:
        return [np.zeros((480, 480, 3), dtype=np.uint8)] * n_frames
    indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
    return [frames[i] for i in indices]


# =============================================================================
# Video helpers
# =============================================================================

def _save_video(frames, path, fps=30):
    """Write a list of (H, W, 3) uint8 frames to an mp4."""
    import imageio
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"  [VIDEO] {len(frames)} frames ({len(frames)/fps:.1f}s) -> {path}")


def save_policy_videos(methods, video_dir, fps=30):
    """
    Save per-policy success and failure clips.
    Outputs:
        {video_dir}/{name}_success.mp4
        {video_dir}/{name}_failure.mp4
    """
    os.makedirs(video_dir, exist_ok=True)
    for method in methods:
        name = method["name"].lower()
        episodes = method["episodes"]

        success_ep = next((ep for ep in episodes if ep[1]), None)
        failure_ep = next((ep for ep in episodes if not ep[1]), None)

        if success_ep:
            _save_video(success_ep[0],
                        os.path.join(video_dir, f"{name}_success.mp4"), fps)
        else:
            print(f"  [VIDEO] {name}: no successful episode to save")

        if failure_ep:
            _save_video(failure_ep[0],
                        os.path.join(video_dir, f"{name}_failure.mp4"), fps)
        else:
            print(f"  [VIDEO] {name}: no failure episode to save")


def save_sidebyside_video(methods, path, fps=30, gap=4):
    """
    Stitch one episode per method horizontally into one comparison video.
    Uses same episode as the figure (first success, else first).
    Shorter episodes are padded with their last frame.
    A colored label bar is burned into the top of each panel.
    """
    import imageio
    import cv2

    chosen = []
    for method in methods:
        ep = method["figure_episode"]
        chosen.append((method["name"], ep[0], ep[1]))

    max_len = max(len(c[1]) for c in chosen)
    target_h = 256
    gap_col = np.zeros((target_h, gap, 3), dtype=np.uint8)

    # Pre-resize and label all panels
    all_panels = []
    for label, frames, success in chosen:
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            new_w = int(w * target_h / h)
            panel = cv2.resize(f, (new_w, target_h), interpolation=cv2.INTER_AREA)
            bar_color = (46, 204, 113) if success else (231, 76, 60)
            cv2.rectangle(panel, (0, 0), (new_w, 28), bar_color, -1)
            cv2.putText(
                panel, f"{label} {'OK' if success else 'FAIL'}",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            resized.append(panel)
        # Pad to max episode length
        while len(resized) < max_len:
            resized.append(resized[-1].copy())
        all_panels.append(resized)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264",
                            quality=8, macro_block_size=1) as writer:
        for t in range(max_len):
            strips = []
            for i, panels in enumerate(all_panels):
                strips.append(panels[t])
                if i < len(all_panels) - 1:
                    strips.append(gap_col)
            row = np.concatenate(strips, axis=1)
            writer.append_data(row)

    print(f"  [VIDEO] Side-by-side -> {path}")


# =============================================================================
# Figure builder
# =============================================================================

def build_figure(methods, n_frames, out_path):
    """
    Grid figure: rows = methods, columns = keyframes.
    Label cell on the left of each row shows method name and success rate.
    """
    n_rows = len(methods)
    n_cols = n_frames
    fig_w = n_cols * 3.2 + 1.2   # +1.2 for label column
    fig_h = n_rows * 3.2

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=120)
    fig.patch.set_facecolor("#1a1a1a")

    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[0.35] + [1.0] * n_cols,
        hspace=0.04, wspace=0.04,
        left=0.01, right=0.99, top=0.99, bottom=0.01
    )

    for row_idx, method in enumerate(methods):
        sr = method["success_rate"]
        name = method["name"]

        # Label cell on the left
        ax_hdr = fig.add_subplot(gs[row_idx, 0])
        ax_hdr.set_xlim(0, 1)
        ax_hdr.set_ylim(0, 1)
        ax_hdr.axis("off")
        bg_color = "#2ecc71" if sr >= 0.5 else "#e74c3c"
        tick = "\u2713" if sr >= 0.5 else "\u2717"
        rect = mpatches.FancyBboxPatch(
            (0.05, 0.01), 0.90, 0.98,
            boxstyle="round,pad=0.02",
            linewidth=0, facecolor=bg_color
        )
        ax_hdr.add_patch(rect)
        ax_hdr.text(
            0.5, 0.6, name,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="white",
            transform=ax_hdr.transAxes
        )
        ax_hdr.text(
            0.5, 0.35, f"{sr:.0%} {tick}",
            ha="center", va="center",
            fontsize=11, color="white",
            transform=ax_hdr.transAxes
        )

        # Use the shared-task episode for consistent object placement across columns
        keyframes = pick_keyframes(method["figure_episode"][0], n_frames)

        # Frame cells
        for col_idx, frame in enumerate(keyframes):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            ax.imshow(frame)
            ax.axis("off")
            ax.text(0.03, 0.97, f"t{col_idx + 1}",
                    transform=ax.transAxes,
                    fontsize=10, fontweight="bold",
                    color="#f1c40f", va="top", ha="left")

    os.makedirs(
        os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
        exist_ok=True
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n[INFO] Saved figure -> {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def parse_ckpt_args(raw_list):
    result = []
    for item in raw_list:
        if "=" not in item:
            raise ValueError(f"Expected 'name=path', got: {item!r}")
        name, path = item.split("=", 1)
        result.append((name.strip(), path.strip()))
    return result


def parse_success_rates(raw_list):
    """Parse 'name=rate' pairs into a dict of {name_lower: float}."""
    result = {}
    for item in raw_list:
        if "=" not in item:
            raise ValueError(f"Expected 'name=rate', got: {item!r}")
        n, r = item.split("=", 1)
        result[n.strip().lower()] = float(r.strip())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Roll out and compare multiple policies in MetaWorld."
    )
    parser.add_argument("--ckpts", nargs="+", required=True, metavar="NAME=PATH",
                        help="Policy checkpoints as 'label=path' pairs.")
    parser.add_argument("--obs_norm", required=True,
                        help="Shared obs normalizer .npz")
    parser.add_argument("--env", default=ENV_NAME)
    parser.add_argument("--n_frames", type=int, default=5,
                        help="Keyframe rows in the figure (default: 5)")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Episodes per policy for success rate (default: 20)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=f"figures/policy_comparison_{ENV_NAME}.png")
    parser.add_argument("--save_videos", action="store_true",
                        help="Also save per-policy and side-by-side mp4 videos")
    parser.add_argument("--video_dir", default=f"videos/{ENV_NAME}",
                        help="Output directory for videos (default: videos/)")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--success_rates", nargs="*", default=[], metavar="NAME=RATE",
                        help="Precomputed success rates as 'name=rate' pairs "
                             "(e.g. bc=0.84 dpo=0.00). Overrides eval when provided.")
    parser.add_argument("--skip_task_search", nargs="*", default=["dpo"], metavar="NAME",
                        help="Policy names excluded from the shared-task success "
                             "criterion (default: dpo). All policies are still rolled "
                             "out on the chosen task.")
    args = parser.parse_args()

    ckpt_pairs = parse_ckpt_args(args.ckpts)
    precomp_sr = parse_success_rates(args.success_rates)
    skip_names = {n.lower() for n in args.skip_task_search}

    # ── Load all policies ─────────────────────────────────────────────────────
    loaded = []
    for name, ckpt_path in ckpt_pairs:
        print(f"[INFO] Loading {name}  ({ckpt_path})")
        policy, obs_norm = load_policy(ckpt_path, args.obs_norm)
        loaded.append((name, policy, obs_norm))

    # ── Find shared task (exclude known-failing policies from criterion) ───────
    shared_task = find_shared_task(
        loaded, args.env, seed=args.seed, skip_names=skip_names
    )

    # ── Roll out each policy on the shared task (for the figure) ──────────────
    mt1_fig = metaworld.MT1(args.env, seed=args.seed)
    methods = []
    for name, policy, obs_norm in loaded:
        print(f"\n{'='*55}")
        print(f"  {name}  ({[p for n, p in ckpt_pairs if n == name][0]})")
        print(f"{'='*55}")

        fig_env = mt1_fig.train_classes[args.env](render_mode="rgb_array")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig_frames, fig_success, fig_ret = rollout_single(
                policy, obs_norm, fig_env, shared_task
            )
        fig_env.close()
        figure_episode = (fig_frames, fig_success)
        print(f"  Shared task: success={fig_success}  return={fig_ret:.1f}")

        # Success rate: precomputed or evaluated
        if name.lower() in precomp_sr:
            success_rate = precomp_sr[name.lower()]
            episodes = [figure_episode]
            print(f"  Using precomputed success rate: {success_rate:.2%}")
        else:
            success_rate, episodes = eval_policy(
                policy, obs_norm, env_name=args.env,
                n_eval=args.n_eval, seed=args.seed,
            )
            print(f"  -> {name} success rate: {success_rate:.2%}")

        methods.append({
            "name": name.upper(),
            "success_rate": success_rate,
            "episodes": episodes,
            "figure_episode": figure_episode,
        })

    print(f"\n[INFO] Building figure ({args.n_frames} keyframes x {len(methods)} methods)...")
    build_figure(methods, n_frames=args.n_frames, out_path=args.out)

    if args.save_videos:
        print(f"\n[INFO] Saving per-policy videos -> {args.video_dir}/")
        save_policy_videos(methods, video_dir=args.video_dir, fps=args.fps)

        sbs_path = os.path.join(args.video_dir, f"comparison_sidebyside_{ENV_NAME}.mp4")
        print(f"\n[INFO] Saving side-by-side comparison video...")
        save_sidebyside_video(methods, path=sbs_path, fps=args.fps)