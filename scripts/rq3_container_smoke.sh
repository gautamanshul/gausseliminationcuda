#!/usr/bin/env bash
set -euo pipefail

cd /app
mkdir -p results

timestamp="$(date -u +%Y%m%d_%H%M%S)"
out_path="results/rq3_container_smoke_${timestamp}.csv"
log_path="results/rq3_container_smoke_${timestamp}.log"

echo "RQ3_CONTAINER_SMOKE start timestamp=${timestamp}" | tee "${log_path}"
echo "RQ3_CONTAINER_SMOKE binary=/app/build/gauss_elim_bench" | tee -a "${log_path}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    {
        echo "RQ3_CONTAINER_ERROR nvidia-smi not found inside container"
        echo "This usually means the container was started without NVIDIA GPU access."
        echo "Run from the host with:"
        echo '  docker run --rm --gpus all -v "$PWD/results:/app/results" gausselim:rq3'
        echo "If using Docker Desktop's GUI, make sure GPU support is enabled or use the CLI command above."
    } | tee -a "${log_path}"
    exit 2
fi

nvidia-smi | tee -a "${log_path}"

./build/gauss_elim_bench --gtest_filter='GaussFastCorrectness.*:GaussV6cCorrectness.*' --gtest_color=yes \
    2>&1 | tee -a "${log_path}"

run_case() {
    local variant="$1"
    shift
    echo "RQ3_CONTAINER_CASE variant=${variant}" | tee -a "${log_path}"
    ./build/gauss_elim_bench --ablation --variant "${variant}" --n 512,1024 --block 512 "$@" --out "${out_path}" \
        2>&1 | tee -a "${log_path}"
}

run_case V3f
run_case V4
run_case V5af --tile-rows 32 --tile-cols 32
run_case V6c --panel-width 64

python3 scripts/validate_rq3_smoke_csv.py "${out_path}" | tee -a "${log_path}"

echo "RQ3_CONTAINER_SMOKE done csv=${out_path} log=${log_path}" | tee -a "${log_path}"
