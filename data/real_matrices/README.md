# V10 Real-Matrix Inputs

This folder holds small documented real-matrix inputs for the V10 external-validity supplement.

Supported input formats:

- Matrix Market coordinate or array matrices (`.mtx`). Coordinate matrices are densified in row-major order before solving.
- Plain dense text/CSV matrices. Either provide exactly `n` rows with `n` numeric columns, or a first line containing `n` followed by `n` rows.

The V10 harness builds a deterministic reference solution `x_ref`, computes `b = A * x_ref`, and then runs the selected solver variant. This keeps residual and solution-error metrics comparable even when the downloaded matrix does not include a right-hand side.

Example:

```powershell
out\build\x64-Release\gauss_elim_bench.exe --ablation --variant V3f --real-matrix data\real_matrices\toy5.mtx --matrix-name toy5 --out results\v10_real_matrix_smoke.csv
```

For dissertation runs, replace `toy5.mtx` with a small, cited SuiteSparse subset that fits GTX 1650 memory. Record skipped matrices and the reason they were skipped.

The first bounded SuiteSparse pilot subset is stored under `suitesparse/`:

- `manifest.csv` documents selected matrices, source URLs, dimensions, rank/conditioning metadata, and selection reasons.
- `skipped.csv` documents candidates intentionally deferred from the first pilot.
- Matrix Market files are kept as `.mtx`; temporary download archives are not retained.
