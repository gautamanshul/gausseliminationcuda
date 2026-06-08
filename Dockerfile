# Reproducible build environment for the A00 Software Mini-Artifact.
# Build: docker build -t gausselim:a00 .
# Run:   docker run --rm --gpus all -v "$PWD/results:/app/results" gausselim:a00
#
# Requires nvidia-container-toolkit on the host. Verify with:
#   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python deps for the plotting script (used when running plot.py inside the
# container; outside the container plot.py can be run on the host).
RUN python3 -m pip install --no-cache-dir matplotlib pandas

WORKDIR /app

# Copy source. .dockerignore prevents the build/ and results/ directories
# from being baked into the image.
COPY . /app

# Build with the CUDA architecture for Turing (GTX 16-series). Override with
#   --build-arg CUDA_ARCH=86  (Ampere)
#   --build-arg CUDA_ARCH=89  (Ada)
ARG CUDA_ARCH=75
RUN mkdir -p build && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
    && cmake --build . -j

# The default command runs the parameterised sweep and emits results/timings.csv.
# The plotting step is left to the host (plot.py) so the image stays minimal.
CMD ["bash", "-lc", "cd /app && ./build/gauss_elim_bench --gtest_color=yes 2>&1 | tee results/run.log"]
