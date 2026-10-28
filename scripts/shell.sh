#!/usr/bin/env bash
# Open an interactive shell in the PyTorch container on one GPU (for short jobs and checks).
# Usage: bash scripts/shell.sh <gpu_id>      then type commands; 'exit' to leave.
HOME_DIR="$HOME"
docker run --rm -it --gpus "device=${1:-1}" --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME="$HOME_DIR" -e USER="$(whoami)" -e LOGNAME="$(whoami)" -e PYTHONPATH="$HOME_DIR/pylibs" \
  -v "$HOME_DIR:$HOME_DIR" -w "$(pwd)" nvcr.io/nvidia/pytorch:26.04-py3 bash
