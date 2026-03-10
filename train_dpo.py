from __future__ import annotations

import argparse
from pathlib import Path

from training_artifacts import tee_stdout_stderr, write_run_metrics_csv, write_run_plot


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not sizes:
        raise argparse.ArgumentTypeError("Hidden sizes must contain at least one integer.")
    return sizes


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_csv_ints(value: str) -> list[int]:
    items = parse_csv_strings(value)
    return [int(item) for item in items]


def sanitize_token(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum():
            safe.append(char)
        else:
            safe.append("_")
    token = "".join(safe).strip("_")
    return token or "task"


def build_output_path(base_output: str, benchmark: str, env_name: str, seed: int, multi_run: bool) -> str:
    if not base_output:
        return ""
    if not multi_run:
        return base_output
    base = Path(base_output)
    suffix = f"{sanitize_token(benchmark)}_{sanitize_token(env_name)}_seed{seed}"
    return str(base.with_name(f"{base.stem}_{suffix}{base.suffix}"))


def format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def build_runs(args: argparse.Namespace) -> tuple[list[argparse.Namespace], bool]:
    benchmarks = parse_csv_strings(args.compare_benchmarks) if args.compare_benchmarks else [args.benchmark]
    benchmarks = [benchmark.lower() for benchmark in benchmarks]
    seeds = parse_csv_ints(args.comparison_seeds) if args.comparison_seeds else [args.seed]
    libero_task_ids = parse_csv_ints(args.libero_task_ids) if args.libero_task_ids else [args.libero_task_id]

    runs: list[argparse.Namespace] = []
    total_run_count = 0
    for benchmark in benchmarks:
        total_run_count += (len(libero_task_ids) if benchmark == "libero" else 1) * len(seeds)
    multi_run = total_run_count > 1

    for benchmark in benchmarks:
        if benchmark not in {"robosuite", "metaworld", "libero"}:
            raise SystemExit(
                f"Unsupported benchmark `{benchmark}` in --compare-benchmarks. "
                "Use only: robosuite, metaworld, libero."
            )

        benchmark_task_ids = libero_task_ids if benchmark == "libero" else [None]
        for task_id in benchmark_task_ids:
            for seed in seeds:
                run_args = argparse.Namespace(**vars(args))
                run_args.benchmark = benchmark
                run_args.seed = seed
                if benchmark == "robosuite":
                    run_args.env_name = args.env_name
                elif benchmark == "metaworld":
                    run_args.env_name = args.metaworld_env_name if args.metaworld_env_name else args.env_name
                else:
                    run_args.libero_task_id = int(task_id if task_id is not None else args.libero_task_id)
                    run_args.env_name = args.libero_env_name if args.libero_env_name else args.env_name
                output_env_name = run_args.env_name
                if benchmark == "libero":
                    base_name = output_env_name if output_env_name else args.libero_suite
                    output_env_name = f"{base_name}_task_{run_args.libero_task_id}"
                run_args.output = build_output_path(
                    base_output=args.output,
                    benchmark=benchmark,
                    env_name=output_env_name,
                    seed=seed,
                    multi_run=multi_run,
                )
                run_args.log_file = build_output_path(
                    base_output=args.log_file,
                    benchmark=benchmark,
                    env_name=output_env_name,
                    seed=seed,
                    multi_run=multi_run,
                )
                run_args.metrics_output = build_output_path(
                    base_output=args.metrics_output,
                    benchmark=benchmark,
                    env_name=output_env_name,
                    seed=seed,
                    multi_run=multi_run,
                )
                run_args.plot_output = build_output_path(
                    base_output=args.plot_output,
                    benchmark=benchmark,
                    env_name=output_env_name,
                    seed=seed,
                    multi_run=multi_run,
                )
                run_args.best_checkpoint_path = build_output_path(
                    base_output=args.best_checkpoint_path,
                    benchmark=benchmark,
                    env_name=output_env_name,
                    seed=seed,
                    multi_run=multi_run,
                )
                runs.append(run_args)
    return runs, multi_run


def maybe_write_comparison_report(output_path: str, results: list[dict[str, object]], args: argparse.Namespace) -> None:
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DPO Benchmark Comparison",
        "",
        f"- Benchmarks: `{args.compare_benchmarks or args.benchmark}`",
        f"- Seeds: `{args.comparison_seeds or args.seed}`",
        "",
        "| method | benchmark | env | seed | final_eval_return | final_eval_success | best_eval_success | best_checkpoint | checkpoint |",
        "|---|---|---|---:|---:|---:|---:|---|---|",
    ]
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(result.get("method", "")),
                    str(result.get("benchmark", "")),
                    str(result.get("env_name", "")),
                    str(result.get("seed", "")),
                    format_metric(result.get("final_eval_return", "")),
                    format_metric(result.get("final_eval_success", "")),
                    format_metric(result.get("best_eval_success", "")),
                    str(result.get("best_checkpoint", "")),
                    str(result.get("checkpoint", "")),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote DPO comparison report to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Robosuite policy with direct preference optimization (DPO)."
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="robosuite",
        choices=("robosuite", "metaworld", "libero"),
        help="Single benchmark to train on.",
    )
    parser.add_argument("--env-name", type=str, default="Lift")
    parser.add_argument(
        "--compare-benchmarks",
        type=str,
        default="",
        help="Comma-separated benchmark sweep, e.g. robosuite,metaworld,libero.",
    )
    parser.add_argument(
        "--metaworld-env-name",
        type=str,
        default="",
        help="Task name used when benchmark=metaworld in comparison mode.",
    )
    parser.add_argument(
        "--libero-env-name",
        type=str,
        default="",
        help="Gym env id for LIBERO (if registered). Leave empty to resolve from suite/task metadata.",
    )
    parser.add_argument("--libero-suite", type=str, default="libero_object")
    parser.add_argument("--libero-task-id", type=int, default=0)
    parser.add_argument(
        "--libero-task-ids",
        type=str,
        default="",
        help="Comma-separated LIBERO task ids for comparison sweeps (e.g., 0,3,4,7).",
    )

    parser.add_argument("--robot", type=str, default="Panda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--comparison-seeds",
        type=str,
        default="",
        help="Comma-separated seeds for comparison runs.",
    )
    parser.add_argument(
        "--comparison-output",
        type=str,
        default="",
        help="Optional markdown file path for comparison summary table.",
    )
    parser.add_argument(
        "--fail-on-missing-benchmark",
        action="store_true",
        help="Fail instead of skipping when a benchmark dependency/environment is missing.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Optional plain-text log file path for full stdout/stderr capture.",
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="",
        help="Optional per-run metrics CSV output path.",
    )
    parser.add_argument(
        "--plot-output",
        type=str,
        default="",
        help="Optional per-run PNG plot output path.",
    )
    parser.add_argument(
        "--save-best-checkpoint",
        action="store_true",
        help="Save checkpoint snapshots whenever eval metric improves.",
    )
    parser.add_argument(
        "--best-checkpoint-path",
        type=str,
        default="",
        help="Output path for best checkpoint. Defaults to <output>_best.pt when enabled.",
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="eval_success",
        choices=("eval_success", "eval_return"),
        help="Evaluation metric used to decide best checkpoint snapshots.",
    )
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--hard-reset", action="store_true", help="Enable hard environment reset each episode.")
    parser.add_argument(
        "--robosuite-log-level",
        type=str,
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )

    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--episodes-per-iter", type=int, default=16)
    parser.add_argument("--pairs-per-iter", type=int, default=64)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--preference-noise", type=float, default=0.0)
    parser.add_argument("--pref-buffer-size", type=int, default=256)

    parser.add_argument("--policy-hidden-sizes", type=parse_hidden_sizes, default=(256, 256))
    parser.add_argument("--init-log-std", type=float, default=-0.5)
    parser.add_argument("--policy-lr", type=float, default=3e-4)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--dpo-epochs", type=int, default=6)
    parser.add_argument("--dpo-batch-size", type=int, default=16)
    parser.add_argument("--bc-regularizer", type=float, default=0.02)

    parser.add_argument("--bc-pool-episodes", type=int, default=40)
    parser.add_argument("--bc-top-fraction", type=float, default=0.25)
    parser.add_argument("--bc-epochs", type=int, default=10)
    parser.add_argument("--bc-batch-size", type=int, default=512)
    parser.add_argument("--bc-lr", type=float, default=1e-3)

    parser.add_argument("--sparse-reward", action="store_true", help="Use sparse task reward instead of shaped reward.")
    parser.add_argument("--output", type=str, default="checkpoints/dpo_policy.pt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.reward_shaping = not args.sparse_reward

    try:
        from robosuite_pref_learning import run_dpo
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit(
                "Missing dependency `torch`. Install it with `pip install torch` and rerun."
            ) from exc
        raise

    runs, multi_run = build_runs(args)
    results: list[dict[str, object]] = []
    for run_args in runs:
        with tee_stdout_stderr(run_args.log_file):
            try:
                summary = run_dpo(run_args)
            except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
                if multi_run and not args.fail_on_missing_benchmark:
                    print(
                        f"Skipping benchmark={run_args.benchmark} env={run_args.env_name} "
                        f"task_id={getattr(run_args, 'libero_task_id', 'N/A')} seed={run_args.seed}: {exc}"
                    )
                    continue
                raise

            if run_args.metrics_output:
                if write_run_metrics_csv(summary, run_args.metrics_output):
                    print(f"Wrote DPO metrics CSV to {run_args.metrics_output}")
                else:
                    print(f"Skipped DPO metrics CSV for {run_args.benchmark}/{run_args.env_name}: no history found.")

            if run_args.plot_output:
                if write_run_plot(summary, run_args.plot_output):
                    print(f"Wrote DPO metrics plot to {run_args.plot_output}")
                else:
                    print(f"Skipped DPO plot for {run_args.benchmark}/{run_args.env_name}: no history found.")
        results.append(summary)

    if not results:
        raise SystemExit("No DPO runs completed successfully.")

    if multi_run:
        print("DPO comparison summary:")
        for result in results:
            print(
                f"- benchmark={result['benchmark']} env={result['env_name']} seed={result['seed']} "
                f"final_eval_success={format_metric(result['final_eval_success'])} "
                f"best_eval_success={format_metric(result['best_eval_success'])}"
            )

    maybe_write_comparison_report(args.comparison_output, results, args)


if __name__ == "__main__":
    main()
