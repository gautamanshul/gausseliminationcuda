# Reproducible NVIDIA/Linux build environment for the RQ3 smoke artifact.
# Build: docker build -t gausselim:rq3 .
# Run:   docker run --rm --gpus all -v "$PWD/results:/app/results" gausselim:rq3
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

# The default command runs the RQ3 container smoke. This is a supplemental
# Linux/NVIDIA reproducibility path; the defense baseline remains the validated
# Windows + Visual Studio + CUDA workflow documented under dissertation_work.
CMD ["bash", "scripts/rq3_container_smoke.sh"]
