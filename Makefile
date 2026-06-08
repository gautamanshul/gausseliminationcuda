# A00 Software Mini-Artifact convenience targets.
# Single-command reproduction:  make reproduce
# Host-native build only:        make build
# Run tests on host:             make test
# Clean build + results:         make clean

SHELL := /bin/bash

# Default CUDA architecture: 75 = Turing (GTX 16-series, RTX 20-series).
# Override at the command line, e.g.  make CUDA_ARCH=86 build  (Ampere)
CUDA_ARCH ?= 75

# Image tag for the Docker target.
IMAGE_TAG ?= gausselim:a00

.PHONY: help reproduce build test docker-build docker-run plot clean

help:
	@echo "Targets:"
	@echo "  reproduce      Build via Docker, run the sweep, plot the figure (one-shot)"
	@echo "  build          Host-native CMake build into ./build"
	@echo "  test           Run GoogleTest sweep on the host (requires build)"
	@echo "  docker-build   Build the Docker image $(IMAGE_TAG)"
	@echo "  docker-run     Run the sweep inside Docker; outputs to ./results"
	@echo "  plot           Regenerate ./results/cpu_vs_gpu_execution_time.png"
	@echo "  clean          Remove ./build and ./results"
	@echo ""
	@echo "Override CUDA architecture with CUDA_ARCH (default $(CUDA_ARCH))."

reproduce: results docker-build docker-run plot
	@echo ""
	@echo "Reproduction complete. See:"
	@echo "  results/timings.csv"
	@echo "  results/cpu_vs_gpu_execution_time.png"

build:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH)
	cd build && cmake --build . -j

test: build results
	cd build && ctest --output-on-failure

docker-build:
	docker build --build-arg CUDA_ARCH=$(CUDA_ARCH) -t $(IMAGE_TAG) .

docker-run: results
	docker run --rm --gpus all \
		-v "$$PWD/results:/app/results" \
		$(IMAGE_TAG)

plot: results
	python3 plot.py results/timings.csv

results:
	mkdir -p results

clean:
	rm -rf build results
