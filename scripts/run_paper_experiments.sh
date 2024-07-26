#!/usr/bin/env bash

set -euo pipefail

# Create testing shim if Slurm is not available
if ! command -v sbatch >/dev/null 2>&1; then
  function sbatch() {
    # Based on https://stackoverflow.com/a/38865224/2445901
    shuf --input-range=100000-999999 -n 1
  }
  TEST_MODE=1
else
  TEST_MODE=
fi

model_types=(
  falcon-7b
  gemma-7b
  gemma-7b-it
  gpt2-large
  llama2-7b
  llama2-13b
  mistral-7b
)
declare -A is_chat_model_type
is_chat_model_type=(
  ["falcon-7b"]="False"
  ["gemma-7b"]="False"
  ["gemma-7b-it"]="True"
  ["gpt2-large"]="False"
  ["llama2-7b"]="False"
  ["llama2-13b"]="False"
  ["mistral-7b"]="True"
)
dataset_types=(
  rcqa
)
epochs=(
  # '05'
  '10'
  # '20'
  # '30'
)
lrs=(
  # '2.5e-5' # Default - 50%
  # '4.0e-5' # Default - 20%
  '5.0e-5' # Default value
  # '6.0e-5' # Default + 20%
  # '7.5e-5' # Default + 50%
)

declare -A models
models=(
  ['falcon-7b-lr5.0e-5-10epoch-rcqa']=data/models/falcon_7b_lr5.0e-5_10epoch_rcqa_2024_04_05_16_01_00/
  ['gemma-7b-lr5.0e-5-10epoch-rcqa']=data/models/gemma_7b_lr5.0e-5_10epoch_rcqa_2024_04_05_16_01_00/
  ['gemma-7b-it-lr5.0e-5-10epoch-rcqa']=data/models/gemma_7b_it_lr5.0e-5_10epoch_rcqa_2024_04_05_16_01_00/
  ['gpt2-large-lr5.0e-5-10epoch-rcqa']=data/models/gpt2_large_lr5.0e-5_10epoch_rcqa_2024_04_05_16_01_00/
  ['llama2-7b-lr5.0e-5-10epoch-rcqa']=data/models/llama2_7b_lr5.0e-5_10epoch_rcqa_2024_04_05_16_01_00/
  ['llama2-13b-lr5.0e-5-10epoch-rcqa']=data/models/llama2_13b_lr5.0e-5_10epoch_rcqa_2024_04_05_16_01_00/
  ['mistral-7b-lr5.0e-5-10epoch-rcqa']=data/models/mistral_7b_lr5.0e-5_10epoch_rcqa_2024_04_05_16_01_00/
)

rcqa_data_dir=rcqa_data
declare -A base_datasets
base_datasets=(
  ['rcqa-train']="$rcqa_data_dir"/prompts/train.jsonl
  ['rcqa-dev']="$rcqa_data_dir"/prompts/test.jsonl
)
declare -A datasets
datasets=(
  ['rcqa-train']="$rcqa_data_dir"/prompts/unified_train.jsonl
  ['rcqa-dev']="$rcqa_data_dir"/prompts/unified_test.jsonl
)
declare -A eval_datasets
eval_datasets=(
  ['rcqa-train']="$rcqa_data_dir"/datasets/train.all.jsonl
  ['rcqa-dev']="$rcqa_data_dir"/datasets/test.all.jsonl
)

# Given that some models are not trained directly (e.g., checkpoints),
# we just wait for all training jobs to complete
declare -a train_job_ids
train_job_ids=()

function train_model() {
  local base_model=$1
  local model=$2
  local train_dataset=$3
  local eval_dataset=$4
  shift 4

  base_model=$(echo "$base_model" | tr '-' '_')
  # Override needed because no `pb-neg-dev` set exists
  if [[ $eval_dataset == pb-neg-dev ]]; then
    eval_dataset=pb-dev
  fi

  local model_id=${models[$model]}
  local train_dataset_path=${datasets[$train_dataset]}
  local eval_dataset_path=${datasets[$eval_dataset]}

  if ! [[ -d config/$base_model ]]; then
    echo "Config for base model '$base_model' does not exist at path 'config/$base_model'" 1>&2
    return 1
  fi
  if [[ -d $model_id ]] && [[ -z $TEST_MODE ]]; then
    echo "Model '$model' already exists at path '$model_id'. Skipping training" 1>&2
    return
  fi
  if ! [[ -f $train_dataset_path ]] && [[ -z $TEST_MODE ]]; then
    echo "Train dataset '$train_dataset' does not exist at path '$train_dataset_path'" 1>&2
    return 1
  fi
  if ! [[ -f $eval_dataset_path ]] && [[ -z $TEST_MODE ]]; then
    echo "Eval dataset '$eval_dataset' does not exist at path '$eval_dataset_path'" 1>&2
    return 1
  fi

  local num_nodes
  num_nodes=$(cat config/"$base_model"/num_nodes.txt)

  cmd=(
    sbatch
    --mail-user "$USER_EMAIL"
    --nodes "$num_nodes"
    --parsable
    scripts/run_training.sh
    "$base_model"
    --train_file "$train_dataset_path"
    --validation_file "$eval_dataset_path"
    --output_dir "$model_id"
    "$@"
  )
  echo "${cmd[@]}"
  local slurm_job_id
  slurm_job_id=$("${cmd[@]}")
  train_job_ids+=("$slurm_job_id")
}

declare -a eval_job_ids
eval_job_ids=()
declare -a eval_names
eval_names=()

# For evaluating models without first fine-tuning.
function evaluate_base_model() {
  local model_id=$1
  local dataset=$2
  shift 2

  local dataset_path=${eval_datasets[$dataset]}

  id_with_underscores="$(echo "$model_id" | tr '-' '_')"
  if [[ -d config/"$id_with_underscores" ]]; then
    real_model_id=$(jq -r .model_name_or_path config/"$id_with_underscores"/transformers_config.json)
    echo "Using ID '$real_model_id' from config for base model '$model_id'" 1>&2
  else
    echo "Config for base model '$model_id' does not exist at path 'config/$model_id'; using model ID as-is" 1>&2
    real_model_id=$model_id
  fi
  if ! [[ -f $dataset_path ]] && [[ -z $TEST_MODE ]]; then
    echo "Dataset '$dataset' does not exist at path '$dataset_path'" 1>&2
    return 1
  fi

  local eval_name=${model_id}_base-$dataset

  dep_commands=()
  cmd=(
    sbatch
    --parsable
    "${dep_commands[@]}"
    scripts/run_evaluation_rcqa.sh
    --name "$eval_name"
    --model "$real_model_id"
    --eval-data "$dataset_path"
    "$@"
  )
  echo "${cmd[@]}"
  local slurm_job_id
  slurm_job_id=$("${cmd[@]}")
  eval_job_ids+=("$slurm_job_id")
  eval_names+=("$eval_name")
}

function evaluate_model() {
  local model=$1
  local dataset=$2
  shift 2

  local model_id=${models[$model]}
  local dataset_path=${eval_datasets[$dataset]}

  if ! [[ -d $model_id ]] && [[ -z $TEST_MODE ]]; then
    echo "Model '$model' does not exist at path '$model_id'" 1>&2
    # Non-fatal because model might already be queued for training
  fi
  if ! [[ -f $dataset_path ]] && [[ -z $TEST_MODE ]]; then
    echo "Dataset '$dataset' does not exist at path '$dataset_path'" 1>&2
    return 1
  fi

  local eval_name=$model-$dataset

  # Handle case where evaluation is called without training first
  if [[ ${#train_job_ids[@]} == 0 ]]; then
    dep_commands=()
  else
    dep_commands=(
      --kill-on-invalid-dep=yes
      "--dependency=afterok:$(echo "${train_job_ids[@]}" | tr ' ' ':')"
    )
  fi

  cmd=(
    sbatch
    --parsable
    "${dep_commands[@]}"
    scripts/run_evaluation_rcqa.sh
    --name "$eval_name"
    --model "$model_id"
    --eval-data "$dataset_path"
    "$@"
  )
  echo "${cmd[@]}"
  local slurm_job_id
  slurm_job_id=$("${cmd[@]}")
  eval_job_ids+=("$slurm_job_id")
  eval_names+=("$eval_name")
}

function collate_eval_output() {
  if [[ ${#eval_names[@]} == 0 ]]; then
    echo "Cannot run collation because no evaluation data is available" 1>&2
    return 1
  fi

  # Handle case where collation is called without evaluation first
  if [[ ${#eval_job_ids[@]} == 0 ]]; then
    dep_commands=()
  else
    dep_commands=(
      --kill-on-invalid-dep=yes
      # Using `afterany` because a failure of
      # one eval job should not stop collation from running
      "--dependency=afterany:$(echo "${eval_job_ids[@]}" | tr ' ' ':')"
    )
  fi

  cmd=(
    sbatch
    --parsable
    "${dep_commands[@]}"
    scripts/collate_paper_eval_output.sh "${eval_names[@]}"
  )
  echo "${cmd[@]}"
  local slurm_job_id
  slurm_job_id=$("${cmd[@]}")
  echo "View results in 'data/slurm_logs/COLLATE_EVAL_OUTPUT-$slurm_job_id.out'"
}

USER_EMAIL=$USER@$(hostname --fqdn | sed -E 's/.*\.(.*\..*)/\1/g')

# Preprocess data to get chat data
for dataset_type in "${dataset_types[@]}"; do
  train_key=$dataset_type-train
  python -m src.process_rcqa_chat \
    --input-file "${base_datasets[$train_key]}" \
    --output-file "${datasets[$train_key]}"
  dev_key=$dataset_type-dev
  python -m src.process_rcqa_chat \
    --input-file "${base_datasets[$dev_key]}" \
    --output-file "${datasets[$dev_key]}"
done

# Evaluate base models
for model_type in "${model_types[@]}"; do
  for dataset_type in "${dataset_types[@]}"; do
    evaluate_base_model \
      "$model_type" \
      "$dataset_type"-dev \
      --run-as-chat "${is_chat_model_type[$model_type]}"
  done
done

# Train models
for model_type in "${model_types[@]}"; do
  for dataset_type in "${dataset_types[@]}"; do
    for epoch in "${epochs[@]}"; do
      for lr in "${lrs[@]}"; do
        train_model \
          "$model_type" \
          "$model_type"-lr"$lr"-"$epoch"epoch-"$dataset_type" \
          "$dataset_type"-train \
          "$dataset_type"-dev \
          --learning_rate "$lr" \
          --num_train_epochs "$epoch" \
          --run_as_chat="${is_chat_model_type[$model_type]}"
      done
    done
  done
done

# Training data, train set
for model_type in "${model_types[@]}"; do
  for dataset_type in "${dataset_types[@]}"; do
    for epoch in "${epochs[@]}"; do
      for lr in "${lrs[@]}"; do
        evaluate_model \
          "$model_type"-lr"$lr"-"$epoch"epoch-"$dataset_type" \
          "$dataset_type"-train \
          --run-as-chat "${is_chat_model_type[$model_type]}"
      done
    done
  done
done

# Training data, dev set
for model_type in "${model_types[@]}"; do
  for dataset_type in "${dataset_types[@]}"; do
    for epoch in "${epochs[@]}"; do
      for lr in "${lrs[@]}"; do
        evaluate_model \
          "$model_type"-lr"$lr"-"$epoch"epoch-"$dataset_type" \
          "$dataset_type"-dev \
          --run-as-chat "${is_chat_model_type[$model_type]}"
      done
    done
  done
done

collate_eval_output
