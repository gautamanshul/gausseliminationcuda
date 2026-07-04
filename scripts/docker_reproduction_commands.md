# RQ3 Container Reproduction Commands

This is the supplemental Linux/NVIDIA container path for RQ3. The validated
defense baseline remains the Windows-native Visual Studio/CMake/CUDA workflow.
Use this container path to check whether another NVIDIA host can build the
artifact and reproduce the representative smoke rows.

## Host Requirements

- Docker or a Docker-compatible engine.
- NVIDIA driver visible on the host.
- NVIDIA Container Toolkit installed and configured.
- Internet access during image build unless the base image is already cached.

Verify GPU visibility from a container:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Build

From the repository root:

```bash
docker build -t gausselim:rq3 .
```

The default CUDA architecture is `75`, matching GTX 1650/Turing-class hardware.
For another GPU generation, override the architecture:

```bash
docker build --build-arg CUDA_ARCH=86 -t gausselim:rq3-sm86 .
```

## Run Smoke

From the repository root:

```bash
mkdir -p results
docker run --rm --gpus all -v "$PWD/results:/app/results" gausselim:rq3
```

Prefer the CLI command above. Starting the image from Docker Desktop's graphical
Run button may omit the required NVIDIA runtime flag. If the log shows:

```text
nvidia-smi: command not found
```

then the container was started without GPU access. Re-run with `--gpus all` from
PowerShell, Command Prompt, Git Bash, or WSL.

The container runs:

- focused correctness tests for fast custom variants and V6c;
- `V3f` at `n = 512, 1024`;
- `V4` at `n = 512, 1024`;
- `V5af_t32x32` at `n = 512, 1024`;
- `V6c_b64` at `n = 512, 1024`;
- CSV validation for finite `gpu_ms`, residual, and solution-error values.

Expected output files:

```text
results/rq3_container_smoke_YYYYMMDD_HHMMSS.csv
results/rq3_container_smoke_YYYYMMDD_HHMMSS.log
```

## Interpretation

The container smoke supports build and correctness reproducibility. Exact timing
equivalence with the Windows defense baseline is not claimed because the
container changes the operating system, compiler toolchain, CUDA toolkit
version, and possibly driver/runtime behavior.
