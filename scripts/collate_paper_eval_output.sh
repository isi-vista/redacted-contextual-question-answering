#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1
#SBATCH --time=30:00
#SBATCH --job-name=COLLATE_EVAL_OUTPUT
#SBATCH --output=data/slurm_logs/%x-%j.out  # `%x-%j` -> JOB_NAME-JOB_ID
#SBATCH --mail-type=all

set -euo pipefail

RCQA_EVAL_DIR=$(python -c 'from src.paths import RCQA_EVAL_DIR; print(RCQA_EVAL_DIR)')

# Arguments are the "evaluation keys" that serve as the output file names
eval_keys=("$@")

echo '### Evaluation report (text)'
for eval_key in "${eval_keys[@]}"; do
  file="$RCQA_EVAL_DIR"/"$eval_key".jsonl
  echo "## $eval_key"
  if [[ -f $file ]]; then
    cat "$file"
  else
    echo 'No evaluation results available'
  fi
  echo
done

echo '### Evaluation report (CSV)'
collated_report="$RCQA_EVAL_DIR"/collated_eval.csv
for eval_key in "${eval_keys[@]}"; do
  file="$RCQA_EVAL_DIR"/"$eval_key".jsonl
  as_csv_file="$RCQA_EVAL_DIR"/"$eval_key".csv
  case "$eval_key" in
    *-lr-*epoch-rcqa-rcqa-*)
      echo -n "$eval_key," |
        sed -E 's/(falcon-7b|llama2-7b|llama2-13b|gemma-7b|gemma-7b-it|mistral-7b)-lr([0-9.]+e-?[0-9]+)-([0-9]+)epoch-(rcqa)-((rcqa)-(train|dev))/\1,\2,\3,\4,\5/g' |
        tee -a "$collated_report"
      ;;
    *-rcqa-rcqa-*)
      echo -n "$eval_key," |
        sed -E 's/(falcon-7b|llama2-7b|llama2-13b|gemma-7b|gemma-7b-it|mistral-7b)-(rcqa)-((rcqa)-(train|dev))/\1,,,\2,\3/g' |
        tee -a "$collated_report"
      ;;
    *)
      printf '%s' "$eval_key,,,," | tee -a "$collated_report"
      ;;
  esac
  printf '\n' | tee -a "$collated_report"
  if [[ -f $file ]]; then
    # Thanks to Stack Overflow user `user3899165`, https://stackoverflow.com/a/32965227
    #
    # jac: Shellcheck gets confused about the jq $variables and thinks they are
    # supposed to be shell variable expansions
    # shellcheck disable=SC2016
    filter='map(del(.generation_metadata)) |
      (map(keys) | add | unique) as $cols |
      map(. as $row | $cols | map($row[.])) as $rows |
      $cols, $rows[] |
      @csv'
    jq -rs "$filter" "$file" | tee "$collated_report" | tee "$as_csv_file"
  else
    echo 'eval failed,eval failed,eval failed,eval failed,eval failed'
  fi
done
