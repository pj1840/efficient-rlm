from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def load_records(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("records", [])


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _stdev(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def mode_metric_stats(records: list[dict], metric: str) -> tuple[list[str], list[float], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        mode = record.get("mode")
        value = record.get("metrics", {}).get(metric)
        if mode and isinstance(value, (int, float)):
            grouped[str(mode)].append(float(value))
    labels = sorted(grouped)
    return labels, [_mean(grouped[label]) for label in labels], [_stdev(grouped[label]) for label in labels]


def mode_score_stats(records: list[dict]) -> tuple[list[str], list[float], list[float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        mode = record.get("mode")
        value = record.get("evaluation", {}).get("score")
        if mode and isinstance(value, (int, float)):
            grouped[str(mode)].append(float(value))
    labels = sorted(grouped)
    return labels, [_mean(grouped[label]) for label in labels], [_stdev(grouped[label]) for label in labels]


def svg_text(x: float, y: float, text: str, *, anchor: str = "middle", size: int = 12) -> str:
    escaped = html.escape(text)
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}">{escaped}</text>'


def save_bar_with_error(
    labels: list[str],
    means: list[float],
    errors: list[float],
    title: str,
    ylabel: str,
    output: Path,
    *,
    color: str,
    max_y: float | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 840
    height = 460
    margin = 72
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    y_max = max_y or max((mean_value + err for mean_value, err in zip(means, errors)), default=1.0)
    y_max = y_max if y_max > 0 else 1.0
    step = plot_width / max(1, len(labels))
    bar_width = step * 0.62
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="460" viewBox="0 0 840 460">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 32, title, size=18),
        svg_text(25, height / 2, ylabel, size=13),
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
    ]
    for index, (label, mean_value, err) in enumerate(zip(labels, means, errors)):
        x = margin + index * step + (step - bar_width) / 2
        bar_height = (mean_value / y_max) * plot_height
        y = height - margin - bar_height
        center = x + bar_width / 2
        err_height = (err / y_max) * plot_height
        err_top = max(margin, y - err_height)
        err_bottom = min(height - margin, y + err_height)
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height:.1f}" fill="{color}"/>'
        )
        if err > 0:
            parts.append(
                f'<line x1="{center:.1f}" y1="{err_top:.1f}" '
                f'x2="{center:.1f}" y2="{err_bottom:.1f}" stroke="#222"/>'
            )
            parts.append(
                f'<line x1="{center - 5:.1f}" y1="{err_top:.1f}" '
                f'x2="{center + 5:.1f}" y2="{err_top:.1f}" stroke="#222"/>'
            )
            parts.append(
                f'<line x1="{center - 5:.1f}" y1="{err_bottom:.1f}" '
                f'x2="{center + 5:.1f}" y2="{err_bottom:.1f}" stroke="#222"/>'
            )
        parts.append(svg_text(center, height - 34, label, size=11))
        parts.append(svg_text(center, y - 8, f"{mean_value:.3g}", size=11))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def save_quality_latency(records: list[dict], output: Path) -> None:
    latency_labels, latencies, _ = mode_metric_stats(records, "wall_time_seconds")
    score_labels, scores, _ = mode_score_stats(records)
    score_by_label = dict(zip(score_labels, scores))
    points = [(label, latency, score_by_label[label]) for label, latency in zip(latency_labels, latencies)]
    output.parent.mkdir(parents=True, exist_ok=True)
    width = 760
    height = 500
    margin = 76
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    max_latency = max((latency for _, latency, _ in points), default=1.0)
    max_latency = max_latency if max_latency > 0 else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 32, "Quality vs Latency", size=18),
        svg_text(width / 2, height - 24, "mean wall time (seconds)", size=13),
        svg_text(28, height / 2, "mean fact coverage", size=13),
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
    ]
    for label, latency, score in points:
        x = margin + (latency / max_latency) * plot_width
        y = height - margin - score * plot_height
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#E15759"/>')
        parts.append(svg_text(x + 8, y - 8, label, anchor="start", size=11))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def save_per_task_fact_coverage(records: list[dict], output: Path) -> None:
    task_ids = sorted({str(record["task_id"]) for record in records})
    modes = sorted({str(record["mode"]) for record in records})
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        score = record.get("evaluation", {}).get("score")
        if isinstance(score, (int, float)):
            grouped[(str(record["task_id"]), str(record["mode"]))].append(float(score))

    output.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 540
    margin = 86
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    group_width = plot_width / max(1, len(task_ids))
    bar_width = group_width / max(1, len(modes)) * 0.72
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1", "#76B7B2"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 32, "Per-Task Fact Coverage", size=18),
        svg_text(28, height / 2, "fact coverage", size=13),
        f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#333"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#333"/>',
    ]
    for task_index, task_id in enumerate(task_ids):
        base_x = margin + task_index * group_width
        for mode_index, mode in enumerate(modes):
            score = _mean(grouped.get((task_id, mode), []))
            x = base_x + mode_index * (group_width / len(modes)) + bar_width * 0.2
            y = height - margin - score * plot_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{score * plot_height:.1f}" fill="{colors[mode_index % len(colors)]}"/>'
            )
        parts.append(svg_text(base_x + group_width / 2, height - 44, task_id.replace("_", " "), size=10))
    for mode_index, mode in enumerate(modes):
        x = margin + mode_index * 125
        y = height - 18
        parts.append(
            f'<rect x="{x:.1f}" y="{y - 10:.1f}" width="10" height="10" '
            f'fill="{colors[mode_index % len(colors)]}"/>'
        )
        parts.append(svg_text(x + 16, y, mode, anchor="start", size=11))
    parts.append("</svg>")
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SVG charts from efficient_rlm benchmark JSON.")
    parser.add_argument("benchmark_json")
    parser.add_argument("--out-dir", default="results/figures")
    args = parser.parse_args()

    records = load_records(Path(args.benchmark_json))
    out_dir = Path(args.out_dir)
    labels, latencies, latency_errors = mode_metric_stats(records, "wall_time_seconds")
    save_bar_with_error(
        labels,
        latencies,
        latency_errors,
        "Latency by Mode",
        "seconds",
        out_dir / "latency_by_mode.svg",
        color="#4C78A8",
    )
    labels, tokens, token_errors = mode_metric_stats(records, "total_tokens")
    save_bar_with_error(
        labels,
        tokens,
        token_errors,
        "Tokens by Mode",
        "tokens",
        out_dir / "tokens_by_mode.svg",
        color="#F28E2B",
    )
    labels, scores, score_errors = mode_score_stats(records)
    save_bar_with_error(
        labels,
        scores,
        score_errors,
        "Fact Coverage by Mode",
        "score",
        out_dir / "fact_coverage_by_mode.svg",
        color="#59A14F",
        max_y=1.0,
    )
    save_quality_latency(records, out_dir / "quality_vs_latency.svg")
    save_per_task_fact_coverage(records, out_dir / "per_task_fact_coverage.svg")
    print(f"Saved charts to {out_dir}")


if __name__ == "__main__":
    main()
