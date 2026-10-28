#!/usr/bin/env bash
# Run any command on one GPU inside the NGC PyTorch container, in the background.
# It keeps running after you close PuTTY. Usage (on the server, from the project folder):
#   bash scripts/run.sh <gpu_id> <job_name> <command...>
#   bash scripts/run.sh 3 xlite_f0 python train.py --config configs/xlite.yaml --fold 0
# Follow the log:  tail -f logs/<job_name>.log      Stop it:  docker stop <job_name>
set -e
GPU=$1; NAME=$2; shift 2
HOME_DIR="$HOME"
PROJ=$(pwd)
mkdir -p logs
docker run -d --rm --name "$NAME" --gpus "device=$GPU" \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME="$HOME_DIR" -e USER="$(whoami)" -e LOGNAME="$(whoami)" -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="$HOME_DIR/pylibs" \
  -v "$HOME_DIR:$HOME_DIR" -w "$PROJ" \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  bash -c "$* > logs/$NAME.log 2>&1"
echo "started '$NAME' on GPU $GPU  ->  tail -f logs/$NAME.log"
