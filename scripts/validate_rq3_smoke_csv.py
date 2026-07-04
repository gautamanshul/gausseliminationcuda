#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path


EXPECTED_VARIANTS = {
    "V3f",
    "V4",
    "V5af_t32x32",
    "V6c_b64",
}


def parse_float(row, key):
    try:
        return float(row[key])
    except KeyError as exc:
        raise SystemExit(f"missing required CSV column: {key}") from exc
    except ValueError as exc:
        raise SystemExit(f"non-numeric value in column {key}: {row.get(key)!r}") from exc


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: validate_rq3_smoke_csv.py <csv-path>")

    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 8:
        raise SystemExit(f"expected 8 smoke rows, found {len(rows)}")

    seen_variants = {row["variant"] for row in rows}
    missing = EXPECTED_VARIANTS - seen_variants
    if missing:
        raise SystemExit(f"missing expected variants: {sorted(missing)}")

    for row in rows:
        gpu_ms = parse_float(row, "gpu_ms")
        residual = parse_float(row, "residual_norm2")
        solution_error = parse_float(row, "solution_error_norm2")
        if not math.isfinite(gpu_ms) or gpu_ms <= 0:
            raise SystemExit(f"invalid gpu_ms for {row['variant']} n={row['n']}: {gpu_ms}")
        if not math.isfinite(residual) or residual >= 1e-4:
            raise SystemExit(
                f"residual threshold failed for {row['variant']} n={row['n']}: {residual}"
            )
        if not math.isfinite(solution_error) or solution_error >= 1e-3:
            raise SystemExit(
                "solution-error threshold failed for "
                f"{row['variant']} n={row['n']}: {solution_error}"
            )

    print(
        "RQ3_CONTAINER_VALIDATE passed "
        f"rows={len(rows)} variants={','.join(sorted(seen_variants))}"
    )


if __name__ == "__main__":
    main()
