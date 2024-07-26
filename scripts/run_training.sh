#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-gpu=62G  # A little less than 1/4 of node's memory
#SBATCH --cpus-per-gpu=10  # Exactly 1/4 of node's CPUs
#SBATCH --gpus-per-node=rtxa6000:4
#SBATCH --time=24:00:00
#SBATCH --job-name=TRAIN_MODEL
#SBATCH --output=data/slurm_logs/%x-%j.out  # `%x-%j` -> JOB_NAME-JOB_ID
#SBATCH --mail-type=all

set -euxo pipefail

# Determine directory script is stored in
# `sbatch` makes a temporary copy when the job is queued,
# so some complex logic is needed
# Currently does not support paths with spaces,
# but that is difficult to do without a Bash parser
# Based on https://stackoverflow.com/a/56991068/2445901
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  SCRIPT_FILE=$(
    scontrol show job "$SLURM_JOB_ID" |
      grep -E '^   Command' |
      cut -f 2- -d '=' |
      cut -f 1 -d ' '
  )
else
  SCRIPT_FILE=$0
fi
ROOT_DIR=$(dirname "$(dirname "$(realpath "$SCRIPT_FILE")")")

MODEL_NAME=$1
shift 1

CONFIG_DIR=$ROOT_DIR/config/$MODEL_NAME
export CONFIG_DIR

function convert_arg_file() {
  # Convert JSON argument file into format `transformers` can process
  # e.g., `{"block_size": 256}` -> `--block_size\n256`
  ARG_FILE=$(mktemp)
  jq -r 'to_entries | map("--\(.key)\n\(.value)") | .[]' \
    <"$CONFIG_DIR"/transformers_config.json \
    >"$ARG_FILE"
  echo "$ARG_FILE"
}
export -f convert_arg_file

# Use safe default to silence `torch.distributed.run` warning
export OMP_NUM_THREADS=1

# Strip away GPU type, leaving only the count
shopt -s extglob # Required for pattern matching syntax to work
NUM_GPUS=${SLURM_GPUS_PER_NODE/#+([[:alnum:]]):/}

# Handle multinode training
if ((SLURM_JOB_NUM_NODES > 1)); then
  multinode_train_args=(
    --deepspeed_multinode_launcher standard
    --machine_rank \$SLURM_PROCID
    --main_process_ip "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
    # Range from cluster documentation
    --main_process_port "$(shuf --input-range=60001-62000 -n 1)"
  )
else
  multinode_train_args=()
fi

MODELS_DIR=$(python -c 'from src.paths import MODELS_DIR; print(MODELS_DIR)')
OUTPUT_DIR=$MODELS_DIR/"$MODEL_NAME"_$(date -u +'%Y_%m_%d_%H_%M_%S')

CMD=(
  accelerate launch
  --config_file "$CONFIG_DIR"/accelerate_config.yaml
  "${multinode_train_args[@]}"
  --num_machines "$SLURM_JOB_NUM_NODES"
  --num_processes $((SLURM_JOB_NUM_NODES * NUM_GPUS))
  -m src.run_clm
  --arg_file "\$(convert_arg_file)"
  --output_dir "$OUTPUT_DIR"
  "$@"
)

time srun bash -xc "${CMD[*]}"
