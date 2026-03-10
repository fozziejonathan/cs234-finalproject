from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.4f}"


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _summarize_training(csv_path: Path) -> dict[str, Any]:
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    method = rows[0].get("method", csv_path.stem).upper()
    benchmark = rows[0].get("benchmark", "")
    env_name = rows[0].get("env_name", "")
    seed = rows[0].get("seed", "")
    iterations = len(rows)

    def best_of(metric: str) -> tuple[float, float]:
        best_value = -math.inf
        best_iter = math.nan
        for row in rows:
            val = _to_float(row.get(metric))
            if not math.isfinite(val):
                continue
            if val > best_value:
                best_value = val
                best_iter = _to_float(row.get("iteration"))
        if best_value == -math.inf:
            return math.nan, math.nan
        return best_value, best_iter

    final = rows[-1]
    final_eval_return = _to_float(final.get("eval_return"))
    final_eval_success = _to_float(final.get("eval_success"))
    final_train_return = _to_float(final.get("train_return"))
    final_train_success = _to_float(final.get("train_success"))
    best_eval_return, best_eval_return_iter = best_of("eval_return")
    best_eval_success, best_eval_success_iter = best_of("eval_success")

    return {
        "method": method,
        "benchmark": benchmark,
        "env_name": env_name,
        "seed": seed,
        "iterations": iterations,
        "final_eval_return": final_eval_return,
        "best_eval_return": best_eval_return,
        "best_eval_return_iter": best_eval_return_iter,
        "final_eval_success": final_eval_success,
        "best_eval_success": best_eval_success,
        "best_eval_success_iter": best_eval_success_iter,
        "final_train_return": final_train_return,
        "final_train_success": final_train_success,
        "csv_path": str(csv_path),
    }


def _load_rollout_summary(json_path: Path, method_hint: str) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list):
        episodes = []

    returns: list[float] = []
    successes: list[float] = []
    steps: list[float] = []
    video_paths: list[str] = []
    for ep in episodes:
        if not isinstance(ep, dict):
            continue
        returns.append(_to_float(ep.get("total_reward")))
        successes.append(_to_float(ep.get("success")))
        steps.append(_to_float(ep.get("num_steps")))
        if ep.get("video_path"):
            video_paths.append(str(ep["video_path"]))

    def safe_mean(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        if not finite:
            return math.nan
        return float(sum(finite) / len(finite))

    return {
        "method": method_hint.upper(),
        "checkpoint": str(payload.get("checkpoint", "")),
        "num_episodes": len(episodes),
        "mean_rollout_return": safe_mean(returns),
        "mean_rollout_success": safe_mean(successes),
        "mean_steps": safe_mean(steps),
        "video_path": video_paths[0] if video_paths else "",
        "json_path": str(json_path),
    }


def _build_markdown(
    training: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    output_path: Path,
) -> str:
    lines: list[str] = []
    lines.append("# RLHF vs DPO Comparison")
    lines.append("")
    lines.append("## Training Metrics")
    lines.append("")
    lines.append(
        "| Method | Benchmark | Env | Seed | Iterations | Final Eval Return | Best Eval Return (iter) | Final Eval Success | Best Eval Success (iter) | Final Train Return | Final Train Success |"
    )
    lines.append("|---|---|---|---:|---:|---:|---|---:|---|---:|---:|")
    for item in training:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["method"]),
                    str(item["benchmark"]),
                    str(item["env_name"]),
                    str(item["seed"]),
                    str(item["iterations"]),
                    _fmt(item["final_eval_return"]),
                    f"{_fmt(item['best_eval_return'])} (iter {int(item['best_eval_return_iter']) if math.isfinite(item['best_eval_return_iter']) else 'nan'})",
                    _fmt(item["final_eval_success"]),
                    f"{_fmt(item['best_eval_success'])} (iter {int(item['best_eval_success_iter']) if math.isfinite(item['best_eval_success_iter']) else 'nan'})",
                    _fmt(item["final_train_return"]),
                    _fmt(item["final_train_success"]),
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Recorded Rollout Metrics")
    lines.append("")
    lines.append("| Method | Episodes | Mean Return | Mean Success | Mean Steps | Checkpoint | Video |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    if rollouts:
        for item in rollouts:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item["method"]),
                        str(item["num_episodes"]),
                        _fmt(item["mean_rollout_return"]),
                        _fmt(item["mean_rollout_success"]),
                        _fmt(item["mean_steps"]),
                        str(item["checkpoint"]),
                        str(item["video_path"]),
                    ]
                )
                + " |"
            )
    else:
        lines.append("| n/a | 0 | nan | nan | nan |  |  |")

    lines.append("")
    lines.append("## Artifact Paths")
    lines.append("")
    for item in training:
        lines.append(f"- {item['method']}: `{item['csv_path']}`")
    for item in rollouts:
        lines.append(f"- {item['method']} rollout: `{item['json_path']}`")
    lines.append(f"- report: `{output_path}`")
    lines.append("")
    return "\n".join(lines)


def _write_matplotlib_table_image(
    training: list[dict[str, Any]],
    rollouts: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Matplotlib is not installed in this environment. "
            "Install it with `./venv/bin/pip install matplotlib` and rerun."
        ) from exc

    def short_path(path_value: str) -> str:
        if not path_value:
            return ""
        return Path(path_value).name

    train_headers = [
        "Method",
        "Benchmark",
        "Env",
        "Seed",
        "Iters",
        "Final Eval Return",
        "Best Eval Return",
        "Final Eval Success",
        "Best Eval Success",
    ]
    train_rows: list[list[str]] = []
    for item in training:
        best_return_iter = (
            int(item["best_eval_return_iter"]) if math.isfinite(item["best_eval_return_iter"]) else "nan"
        )
        best_success_iter = (
            int(item["best_eval_success_iter"]) if math.isfinite(item["best_eval_success_iter"]) else "nan"
        )
        train_rows.append(
            [
                str(item["method"]),
                str(item["benchmark"]),
                str(item["env_name"]),
                str(item["seed"]),
                str(item["iterations"]),
                _fmt(item["final_eval_return"]),
                f"{_fmt(item['best_eval_return'])} (i{best_return_iter})",
                _fmt(item["final_eval_success"]),
                f"{_fmt(item['best_eval_success'])} (i{best_success_iter})",
            ]
        )

    rollout_headers = ["Method", "Episodes", "Mean Return", "Mean Success", "Mean Steps", "Checkpoint", "Video"]
    rollout_rows: list[list[str]] = []
    for item in rollouts:
        rollout_rows.append(
            [
                str(item["method"]),
                str(item["num_episodes"]),
                _fmt(item["mean_rollout_return"]),
                _fmt(item["mean_rollout_success"]),
                _fmt(item["mean_steps"]),
                short_path(str(item["checkpoint"])),
                short_path(str(item["video_path"])),
            ]
        )
    if not rollout_rows:
        rollout_rows.append(["n/a", "0", "nan", "nan", "nan", "", ""])

    fig, axes = plt.subplots(2, 1, figsize=(18, 8), gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle("RLHF vs DPO Comparison Tables", fontsize=16, y=0.98)

    for ax in axes:
        ax.axis("off")

    train_table = axes[0].table(cellText=train_rows, colLabels=train_headers, cellLoc="center", loc="center")
    train_table.auto_set_font_size(False)
    train_table.set_fontsize(10)
    train_table.scale(1.0, 1.6)
    axes[0].set_title("Training Metrics", fontsize=12, pad=10)

    rollout_table = axes[1].table(
        cellText=rollout_rows,
        colLabels=rollout_headers,
        cellLoc="center",
        loc="center",
    )
    rollout_table.auto_set_font_size(False)
    rollout_table.set_fontsize(10)
    rollout_table.scale(1.0, 1.6)
    axes[1].set_title("Recorded Rollout Metrics", fontsize=12, pad=10)

    for table in (train_table, rollout_table):
        for (row, _col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#e6eef8")
                cell.set_text_props(weight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build RLHF vs DPO markdown comparison tables from metrics files.")
    parser.add_argument("--rlhf-csv", type=str, required=True, help="Path to RLHF metrics CSV.")
    parser.add_argument("--dpo-csv", type=str, required=True, help="Path to DPO metrics CSV.")
    parser.add_argument("--rlhf-rollout-json", type=str, default="", help="Optional RLHF rollout metrics JSON.")
    parser.add_argument("--dpo-rollout-json", type=str, default="", help="Optional DPO rollout metrics JSON.")
    parser.add_argument(
        "--output",
        type=str,
        default="reports/rlhf_dpo_comparison.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--table-image",
        type=str,
        default="",
        help="Optional matplotlib PNG output path for table visualization.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    training = [
        _summarize_training(Path(args.rlhf_csv)),
        _summarize_training(Path(args.dpo_csv)),
    ]

    rollouts: list[dict[str, Any]] = []
    if args.rlhf_rollout_json:
        rollouts.append(_load_rollout_summary(Path(args.rlhf_rollout_json), method_hint="rlhf"))
    if args.dpo_rollout_json:
        rollouts.append(_load_rollout_summary(Path(args.dpo_rollout_json), method_hint="dpo"))

    markdown = _build_markdown(training=training, rollouts=rollouts, output_path=output_path)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote comparison report to {output_path}")
    if args.table_image:
        table_image_path = Path(args.table_image)
        _write_matplotlib_table_image(training=training, rollouts=rollouts, output_path=table_image_path)
        print(f"Wrote matplotlib table image to {table_image_path}")


if __name__ == "__main__":
    main()
