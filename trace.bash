#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --onnx    Run Torch to ONNX export only"
    echo "  --trt     Run ONNX to TensorRT export only"
    echo "  --all     Run both exports (default)"
    echo "  --help    Show this help message"
    exit 0
}

RUN_ONNX=false
RUN_TRT=false

case "${1:-all}" in
    --onnx) RUN_ONNX=true ;;
    --trt)  RUN_TRT=true ;;
    --all)  RUN_ONNX=true; RUN_TRT=true ;;
    --help) usage ;;
    *) echo "Unknown option: ${1}. Use --help for usage."; exit 1 ;;
esac

XAUTH="${XAUTHORITY:-${HOME}/.Xauthority}"

run_onnx() {
    local image="sam3-tensorrt:trace"

    echo "Checking for Docker image: ${image}..."
    if ! docker image inspect "${image}" &>/dev/null; then
        echo "Image '${image}' not found. Building via ./docker/build-trace.bash..."
        ./docker/build-trace.bash
        echo "Build complete."
    else
        echo "Image '${image}' already exists. Skipping build."
    fi

    echo "Starting Torch to ONNX export..."
    docker run --rm \
        --network=host \
        --gpus all \
        -v "/tmp/.X11-unix:/tmp/.X11-unix" \
        -v "./models:/home/$(whoami)/models" \
        -v "./scripts:/home/$(whoami)/scripts" \
        -e DISPLAY="${DISPLAY:-}" \
        -e QT_X11_NO_MITSHM=1 \
        -e XAUTHORITY="${XAUTH}" \
        -e HF_KEY="${HF_READ_KEY}" \
        --name sam3-tensorrt-trace \
        "${image}" \
        bash -c "cd scripts && python3 export_all.py"
}

run_trt() {
    local image="sam3-tensorrt:trt"

    echo "Checking for Docker image: ${image}..."
    if ! docker image inspect "${image}" &>/dev/null; then
        echo "Image '${image}' not found. Building via ./docker/build-trt.bash..."
        ./docker/build-trt.bash
        echo "Build complete."
    else
        echo "Image '${image}' already exists. Skipping build."
    fi

    echo "Starting ONNX to TensorRT export..."
    docker run --rm \
        --network=host \
        --gpus all \
        -v "/tmp/.X11-unix:/tmp/.X11-unix" \
        -v "./models:/home/$(whoami)/models" \
        -v "./scripts:/home/$(whoami)/scripts" \
        -e DISPLAY="${DISPLAY:-}" \
        -e QT_X11_NO_MITSHM=1 \
        -e XAUTHORITY="${XAUTH}" \
        --name sam3-tensorrt-trt \
        "${image}" \
        bash -c "cd scripts && ./onnx_to_trt.bash"
}

$RUN_ONNX && run_onnx
$RUN_TRT  && run_trt
