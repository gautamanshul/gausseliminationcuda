#!/usr/bin/env python3
"""Create dependency-free log-scale SVGs for the M7 conditioning sweep."""

from __future__ import annotations

import argparse
import csv
import html
import math
from collections import defaultdict
from pathlib import Path


COLORS = [
    "#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c",
    "#0891b2", "#be123c", "#4d7c0f", "#4338ca", "#a16207",
    "#0f766e", "#9333ea", "#475569", "#c2410c", "#1d4ed8",
]


def read_summary(path: Path):
    by_n = defaultdict(lambda: defaultdict(list))
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            by_n[int(row["n"])][row["variant"]].append({
                "kappa": float(row["kappa"]),
                "time": float(row["median_gpu_ms"]),
                "q1": float(row["q1_gpu_ms"]),
                "q3": float(row["q3_gpu_ms"]),
                "solution_error": float(row["max_solution_error_norm2"]),
                "residual": float(row["max_residual_norm2"]),
            })
    for variants in by_n.values():
        for rows in variants.values():
            rows.sort(key=lambda item: item["kappa"])
    return by_n


def render(series, destination: Path, title: str, ylabel: str, field: str,
           show_iqr: bool = False):
    width, height = 1280, 760
    left, right, top, bottom = 110, 350, 78, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    labels = sorted(series)
    x_values = sorted({row["kappa"] for rows in series.values() for row in rows})
    y_values = [row[field] for rows in series.values() for row in rows]
    x_lo, x_hi = math.log10(min(x_values)), math.log10(max(x_values))
    y_lo = math.floor(math.log10(min(y_values)))
    y_hi = math.ceil(math.log10(max(y_values)))
    if y_lo == y_hi:
        y_hi += 1

    def xp(value):
        return left + (math.log10(value) - x_lo) / (x_hi - x_lo) * plot_w

    def yp(value):
        return top + (y_hi - math.log10(value)) / (y_hi - y_lo) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700">{html.escape(title)}</text>',
    ]
    for exponent in range(y_lo, y_hi + 1):
        value = 10.0 ** exponent
        y = yp(value)
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#dbe3ee"/>')
        svg.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="Arial" font-size="13">10^{exponent}</text>')
    for value in x_values:
        x = xp(value)
        exponent = int(round(math.log10(value)))
        svg.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#edf1f7"/>')
        svg.append(f'<text x="{x:.2f}" y="{top + plot_h + 27}" text-anchor="middle" font-family="Arial" font-size="14">10^{exponent}</text>')
    svg.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#1f2937" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#1f2937" stroke-width="2"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="16">Target condition number kappa (log10 scale)</text>',
        f'<text x="28" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 28 {top + plot_h / 2})" font-family="Arial" font-size="16">{html.escape(ylabel)}</text>',
    ])
    for index, label in enumerate(labels):
        color = COLORS[index % len(COLORS)]
        rows = series[label]
        points = [(xp(row["kappa"]), yp(row[field])) for row in rows]
        svg.append('<polyline fill="none" stroke="{}" stroke-width="2.5" points="{}"/>'.format(
            color, " ".join(f"{x:.2f},{y:.2f}" for x, y in points)))
        for row, (x, y) in zip(rows, points):
            if show_iqr:
                y1, y2 = yp(row["q1"]), yp(row["q3"])
                svg.append(f'<line x1="{x:.2f}" y1="{y1:.2f}" x2="{x:.2f}" y2="{y2:.2f}" stroke="{color}"/>')
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix or args.summary_csv.with_name(
        args.summary_csv.stem.removesuffix("_summary"))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    by_n = read_summary(args.summary_csv)
    for n, series in sorted(by_n.items()):
        timing = Path(f"{prefix}_n{n}_timing_by_kappa_loglog.svg")
        error = Path(f"{prefix}_n{n}_solution_error_loglog.svg")
        residual = Path(f"{prefix}_n{n}_residual_loglog.svg")
        render(series, timing, f"Conditioning sweep timing at n={n}",
               "Median GPU elapsed time (ms, log scale)", "time", True)
        render(series, error, f"FP32 solution sensitivity at n={n}",
               "Maximum relative solution error (log scale)", "solution_error")
        render(series, residual, f"Normalized residual stability at n={n}",
               "Maximum normalized residual (log scale)", "residual")
        print(timing)
        print(error)
        print(residual)


if __name__ == "__main__":
    main()
