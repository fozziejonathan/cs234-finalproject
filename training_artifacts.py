from __future__ import annotations

import csv
import math
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Iterator


class _Tee:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            write_fn = getattr(stream, "write", None)
            if callable(write_fn):
                write_fn(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            flush_fn = getattr(stream, "flush", None)
            if callable(flush_fn):
                flush_fn()


@contextmanager
def tee_stdout_stderr(log_path: str) -> Iterator[None]:
    if not log_path:
        yield
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as log_file:
        stdout_tee = _Tee(sys.stdout, log_file)
        stderr_tee = _Tee(sys.stderr, log_file)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _is_finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def write_run_metrics_csv(summary: dict[str, Any], output_path: str) -> bool:
    if not output_path:
        return False

    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return False

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metric_keys: list[str] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        for key in row:
            if key == "iteration":
                continue
            if key not in metric_keys:
                metric_keys.append(key)

    header = ["method", "benchmark", "env_name", "seed", "iteration"] + metric_keys
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        for row in history:
            if not isinstance(row, dict):
                continue
            output_row: dict[str, Any] = {
                "method": summary.get("method", ""),
                "benchmark": summary.get("benchmark", ""),
                "env_name": summary.get("env_name", ""),
                "seed": summary.get("seed", ""),
                "iteration": int(_to_float(row.get("iteration"))),
            }
            for key in metric_keys:
                value = _to_float(row.get(key))
                output_row[key] = "" if not _is_finite(value) else value
            writer.writerow(output_row)
    return True


def _draw_panel(
    draw: Any,
    panel_box: tuple[int, int, int, int],
    history: list[dict[str, Any]],
    title: str,
    series_defs: list[tuple[str, str, tuple[int, int, int]]],
    font: Any,
) -> None:
    left, top, right, bottom = panel_box
    draw.rectangle(panel_box, outline=(200, 200, 200), width=2)
    draw.text((left + 8, top + 6), title, fill=(10, 10, 10), font=font)

    chart_left = left + 55
    chart_top = top + 28
    chart_right = right - 14
    chart_bottom = bottom - 24
    if chart_right <= chart_left or chart_bottom <= chart_top:
        return

    draw.rectangle((chart_left, chart_top, chart_right, chart_bottom), outline=(220, 220, 220), width=1)

    xs: list[float] = []
    all_ys: list[float] = []
    series_values: list[tuple[str, tuple[int, int, int], list[tuple[float, float]]]] = []
    for key, label, color in series_defs:
        points: list[tuple[float, float]] = []
        for row in history:
            x = _to_float(row.get("iteration"))
            y = _to_float(row.get(key))
            if _is_finite(x) and _is_finite(y):
                points.append((x, y))
                xs.append(x)
                all_ys.append(y)
        series_values.append((label, color, points))

    if not xs or not all_ys:
        draw.text((chart_left + 8, chart_top + 8), "No finite data", fill=(120, 120, 120), font=font)
        return

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(all_ys), max(all_ys)
    if min_x == max_x:
        min_x -= 1.0
        max_x += 1.0
    if min_y == max_y:
        margin = 1.0 if min_y == 0 else abs(min_y) * 0.1
        min_y -= margin
        max_y += margin

    def to_px(x_val: float, y_val: float) -> tuple[int, int]:
        x_ratio = (x_val - min_x) / (max_x - min_x)
        y_ratio = (y_val - min_y) / (max_y - min_y)
        px = int(chart_left + x_ratio * (chart_right - chart_left))
        py = int(chart_bottom - y_ratio * (chart_bottom - chart_top))
        return px, py

    for grid_i in range(0, 5):
        y_ratio = grid_i / 4.0
        y_px = int(chart_bottom - y_ratio * (chart_bottom - chart_top))
        draw.line((chart_left, y_px, chart_right, y_px), fill=(238, 238, 238), width=1)

    for _, color, points in series_values:
        if len(points) < 2:
            continue
        pixel_points = [to_px(x_val, y_val) for x_val, y_val in points]
        draw.line(pixel_points, fill=color, width=3)

    legend_x = chart_left + 8
    legend_y = chart_top + 8
    for label, color, points in series_values:
        draw.rectangle((legend_x, legend_y + 3, legend_x + 10, legend_y + 13), fill=color, outline=color)
        suffix = "" if points else " (no data)"
        draw.text((legend_x + 16, legend_y), f"{label}{suffix}", fill=(40, 40, 40), font=font)
        legend_y += 16

    draw.text((chart_left - 40, chart_top - 2), f"{max_y:.2f}", fill=(90, 90, 90), font=font)
    draw.text((chart_left - 40, chart_bottom - 10), f"{min_y:.2f}", fill=(90, 90, 90), font=font)
    draw.text((chart_left, chart_bottom + 6), f"iter {int(min_x)}", fill=(90, 90, 90), font=font)
    draw.text((chart_right - 56, chart_bottom + 6), f"iter {int(max_x)}", fill=(90, 90, 90), font=font)


def write_run_plot(summary: dict[str, Any], output_path: str) -> bool:
    if not output_path:
        return False

    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return False

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ModuleNotFoundError:
        return False

    width, height = 1300, 900
    image = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    method = str(summary.get("method", "")).upper()
    benchmark = str(summary.get("benchmark", ""))
    env_name = str(summary.get("env_name", ""))
    seed = summary.get("seed", "")
    title = f"{method} metrics | benchmark={benchmark} env={env_name} seed={seed}"
    draw.text((24, 16), title, fill=(0, 0, 0), font=font)

    panels = [
        (
            "Returns",
            [
                ("train_return", "train_return", (40, 120, 220)),
                ("eval_return", "eval_return", (220, 120, 40)),
            ],
        ),
        (
            "Success Rate",
            [
                ("train_success", "train_success", (40, 170, 90)),
                ("eval_success", "eval_success", (210, 70, 100)),
            ],
        ),
    ]

    if any(("reward_loss" in row) or ("dpo_loss" in row) for row in history if isinstance(row, dict)):
        loss_series: list[tuple[str, str, tuple[int, int, int]]] = []
        if any("reward_loss" in row for row in history if isinstance(row, dict)):
            loss_series.append(("reward_loss", "reward_loss", (120, 80, 220)))
        if any("policy_loss" in row for row in history if isinstance(row, dict)):
            loss_series.append(("policy_loss", "policy_loss", (255, 120, 120)))
        if any("dpo_loss" in row for row in history if isinstance(row, dict)):
            loss_series.append(("dpo_loss", "dpo_loss", (100, 100, 100)))
        if any("bc_loss" in row for row in history if isinstance(row, dict)):
            loss_series.append(("bc_loss", "bc_loss", (30, 150, 150)))
        if loss_series:
            panels.append(("Losses", loss_series))

    top = 48
    panel_height = (height - top - 18) // max(1, len(panels))
    for idx, (panel_title, series_defs) in enumerate(panels):
        panel_top = top + idx * panel_height + 8
        panel_bottom = panel_top + panel_height - 14
        _draw_panel(
            draw=draw,
            panel_box=(20, panel_top, width - 20, panel_bottom),
            history=history,
            title=panel_title,
            series_defs=series_defs,
            font=font,
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    return True
