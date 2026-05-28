#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."

gpus=4
background=0
log_file="output.log"
wandb=1
wandb_project="ip-final-project"
wandb_run_name="unet-pretrain-003"
args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --background|--nohup)
      background=1
      shift
      ;;
    --log-file)
      log_file="$2"
      shift 2
      ;;
    --log-file=*)
      log_file="${1#*=}"
      shift
      ;;
    --gpus)
      gpus="$2"
      shift 2
      ;;
    --gpus=*)
      gpus="${1#*=}"
      shift
      ;;
    --wandb)
      wandb=1
      shift
      ;;
    --no-wandb)
      wandb=0
      shift
      ;;
    --wandb-project)
      wandb_project="$2"
      shift 2
      ;;
    --wandb-project=*)
      wandb_project="${1#*=}"
      shift
      ;;
    --wandb-run-name)
      wandb_run_name="$2"
      shift 2
      ;;
    --wandb-run-name=*)
      wandb_run_name="${1#*=}"
      shift
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

if [[ "$background" -eq 1 ]]; then
  cmd=(bash "$script_dir/run.sh" --gpus "$gpus" --log-file "$log_file")
  if [[ "$wandb" -eq 1 ]]; then
    cmd+=(--wandb --wandb-project "$wandb_project" --wandb-run-name "$wandb_run_name")
  else
    cmd+=(--no-wandb)
  fi
  cmd+=("${args[@]}")

  nohup "${cmd[@]}" > "$log_file" 2>&1 &
  echo "Started training in background: pid=$! log=$log_file"
  exit 0
fi

train_args=()
if [[ "$wandb" -eq 1 ]]; then
  train_args+=(--wandb --wandb-project "$wandb_project" --wandb-run-name "$wandb_run_name")
fi
train_args+=("${args[@]}")

if [[ "$gpus" -gt 1 ]]; then
  uv run python -m torch.distributed.run \
    --rdzv_backend=c10d \
    --rdzv_endpoint=127.0.0.1:0 \
    --nproc_per_node="$gpus" \
    unet_pretrain/train.py \
    "${train_args[@]}"
else
  uv run python unet_pretrain/train.py "${train_args[@]}"
fi
