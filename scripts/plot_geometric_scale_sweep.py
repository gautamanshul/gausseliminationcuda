#!/usr/bin/env python3
"""Create dependency-free SVG figures from a paired-sweep summary CSV."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


FAMILIES = {
    "rank1": ("Rank-1 progression", {"V1", "V2", "V3", "V3f"}),
    "tiling": ("Tiling and loop-unrolling progression",
               {"V3f", "V5a", "V5af", "V5bf"}),
    "blocked": ("LU, library, and blocked families",
                {"V4", "VLU", "V5c", "V6a", "V6b", "V6c", "V6d", "V6e"}),
}
COLORS = [
    "#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c",
    "#0891b2", "#be123c", "#4d7c0f", "#4338ca", "#a16207",
    "#0f766e", "#9333ea", "#475569", "#c2410c", "#1d4ed8",
]


def base_variant(label: str) -> str:
    if label.startswith("V6d"):
        return "V6d"
    return label.split("_", 1)[0]


def display_variant(label: str) -> str:
    if base_variant(label) == "V6d":
        return "V6d (adaptive)"
    return label


def read_summary(path: Path):
    series = defaultdict(list)
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            series[display_variant(row["variant"])].append({
                "n": int(row["n"]),
                "median": float(row["median_gpu_ms"]),
                "q1": float(row["q1_gpu_ms"]),
                "q3": float(row["q3_gpu_ms"]),
                "energy": float(row.get("median_energy_j", "nan")),
                "energy_efficiency": float(
                    row.get("median_joules_per_effective_gflop", "nan")),
            })
    for rows in series.values():
        rows.sort(key=lambda item: item["n"])
    if not series:
        raise ValueError(f"No rows found in {path}")
    return dict(series)


def throughput(row):
    return ((2.0 / 3.0) * row["n"] ** 3) / (row["median"] * 1.0e6)


def render_chart(series, destination: Path, title: str, ylabel: str,
                 value_fn, show_iqr: bool = False):
    width, height = 1280, 760
    left, right, top, bottom = 105, 350, 78, 88
    plot_w = width - left - right
    plot_h = height - top - bottom
    labels = sorted(series)
    points = [(row["n"], value_fn(row)) for rows in series.values() for row in rows]
    x_values = sorted({n for n, _ in points})
    y_values = [value for _, value in points if value > 0 and math.isfinite(value)]
    if not y_values:
        raise ValueError("No positive finite values available for logarithmic plot")
    x_min, x_max = min(x_values), max(x_values)
    y_log_min = math.floor(math.log10(min(y_values)))
    y_log_max = math.ceil(math.log10(max(y_values)))
    if y_log_min == y_log_max:
        y_log_max += 1

    def x_pos(value):
        if x_min == x_max:
            return left + plot_w / 2
        return left + (math.log2(value) - math.log2(x_min)) / (
            math.log2(x_max) - math.log2(x_min)) * plot_w

    def y_pos(value):
        return top + (y_log_max - math.log10(value)) / (
            y_log_max - y_log_min) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{html.escape(title)}</text>',
    ]
    for exponent in range(y_log_min, y_log_max + 1):
        value = 10.0 ** exponent
        y = y_pos(value)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#dbe3ee"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial" font-size="13">10^{exponent}</text>')
    for value in x_values:
        x = x_pos(value)
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#edf1f7"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 27}" text-anchor="middle" font-family="Arial" font-size="14">{value}</text>')
    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#1f2937" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#1f2937" stroke-width="2"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="16">Matrix dimension n (log2 scale)</text>',
        f'<text x="28" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 28 {top + plot_h / 2})" font-family="Arial" font-size="16">{html.escape(ylabel)}</text>',
    ])
    for index, label in enumerate(labels):
        color = COLORS[index % len(COLORS)]
        rows = series[label]
        coordinates = [(x_pos(row["n"]), y_pos(value_fn(row))) for row in rows]
        svg.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(
            color, " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)))
        for row, (x, y) in zip(rows, coordinates):
            if show_iqr and row["q1"] > 0 and row["q3"] > 0:
                y1, y2 = y_pos(row["q1"]), y_pos(row["q3"])
                svg.extend([
                    f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="{color}"/>',
                    f'<line x1="{x - 5:.2f}" y1="{y1:.2f}" x2="{x + 5:.2f}" y2="{y1:.2f}" stroke="{color}"/>',
                    f'<line x1="{x - 5:.2f}" y1="{y2:.2f}" x2="{x + 5:.2f}" y2="{y2:.2f}" stroke="{color}"/>',
                ])
            svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
        legend_x = left + plot_w + 28 + (index // 12) * 150
        legend_y = top + 12 + (index % 12) * 30
        svg.extend([
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<circle cx="{legend_x + 12.5}" cy="{legend_y}" r="3.5" fill="{color}"/>',
            f'<text x="{legend_x + 34}" y="{legend_y + 5}" font-family="Arial" font-size="14">{html.escape(label)}</text>',
        ])
    svg.append('</svg>')
    destination.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.summary_csv.with_name(
        args.summary_csv.stem.removesuffix("_summary"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    series = read_summary(args.summary_csv)
    outputs = []
    timing = Path(f"{prefix}_timing_loglog.svg")
    render_chart(series, timing, "GPU solver scaling on the geometric n ladder",
                 "Median GPU elapsed time (ms, log scale)",
                 lambda row: row["median"], show_iqr=True)
    outputs.append(timing)
    rate = Path(f"{prefix}_throughput_loglog.svg")
    render_chart(series, rate, "Effective throughput as problem size grows",
                 "Effective LU-equivalent throughput (GFLOP/s, log scale)", throughput)
    outputs.append(rate)
    if all(math.isfinite(row["energy"]) and row["energy"] > 0
           for rows in series.values() for row in rows):
        energy = Path(f"{prefix}_energy_loglog.svg")
        render_chart(series, energy, "Process-envelope energy to solution",
                     "Median sampled energy (J, log scale)",
                     lambda row: row["energy"])
        outputs.append(energy)
        efficiency = Path(f"{prefix}_energy_efficiency_loglog.svg")
        render_chart(series, efficiency,
                     "Energy per effective LU-equivalent work",
                     "Median J per effective GFLOP (log scale)",
                     lambda row: row["energy_efficiency"])
        outputs.append(efficiency)
    for slug, (title, members) in FAMILIES.items():
        selected = {label: rows for label, rows in series.items()
                    if base_variant(label) in members}
        if not selected:
            continue
        output = Path(f"{prefix}_family_{slug}.svg")
        render_chart(selected, output, title,
                     "Median GPU elapsed time (ms, log scale)",
                     lambda row: row["median"], show_iqr=True)
        outputs.append(output)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
