#!/usr/bin/env python3
"""
Regenerates the CPU-vs-GPU execution-time figure from results/timings.csv.

Usage:
    python3 plot.py results/timings.csv

The CSV must have columns: n, block_size, cpu_ms, gpu_ms, residual_max,
driver_version, timestamp.

Produces results/cpu_vs_gpu_execution_time.png at 150 dpi.

This script intentionally has minimal dependencies (matplotlib + pandas) and
no plotly/seaborn/etc., so the artifact stays portable across student
environments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    sys.stderr.write(
        f"Missing Python dependency: {e}.\n"
        f"Install with:  python3 -m pip install matplotlib pandas\n"
    )
    sys.exit(1)


def main(csv_path: Path) -> None:
    if not csv_path.exists():
        sys.stderr.write(
            f"Input not found: {csv_path}\n"
            f"Run the benchmark first (`make test` or `make reproduce`).\n"
        )
        sys.exit(2)

    df = pd.read_csv(csv_path)
    required_cols = {"n", "cpu_ms", "gpu_ms"}
    missing = required_cols - set(df.columns)
    if missing:
        sys.stderr.write(f"CSV is missing required columns: {missing}\n")
        sys.exit(3)

    df = df.sort_values("n")

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(df["n"], df["cpu_ms"], marker="o", label="CPU (sequential C++)")
    ax.plot(df["n"], df["gpu_ms"], marker="s", label="GPU (CUDA, block=512)")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Solve time (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("CPU vs GPU Gauss elimination on commodity Turing (GTX 1650)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    for _, row in df.iterrows():
        ax.annotate(
            f"{row['cpu_ms']:.1f}",
            xy=(row["n"], row["cpu_ms"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
        ax.annotate(
            f"{row['gpu_ms']:.2f}",
            xy=(row["n"], row["gpu_ms"]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    fig.tight_layout()
    out = csv_path.parent / "cpu_vs_gpu_execution_time.png"
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="?",
        default="results/timings.csv",
        help="Path to timings CSV (default: results/timings.csv)",
    )
    args = parser.parse_args()
    main(Path(args.csv))
