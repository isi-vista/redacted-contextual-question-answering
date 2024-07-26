#!/usr/bin/env bash

#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=DOWNLOAD_MODELS
#SBATCH --output=data/slurm_logs/%x-%j.out  # `%x-%j` -> JOB_NAME-JOB_ID
#SBATCH --mail-type=all

set -euxo pipefail

PYTHONPATH=$(dirname "$(dirname "$(realpath "$0")")")
export PYTHONPATH

# Download `transformers` models
# There does not appear to be a direct way to query the cache's contents,
# so we check the expected cache directory name to see if a model has been downloaded.
TRANSFORMERS_CACHE_DIR=$(python -c 'from src.paths import TRANSFORMERS_CACHE_DIR; print(TRANSFORMERS_CACHE_DIR)')
# Parse all model names directly from configuration files
model_name_string=$(
  find config/ -name transformers_config.json |
    sort |
    xargs -I {} jq -r .model_name_or_path {} |
    sort
)
readarray -t models <<<"$model_name_string"
for model in "${models[@]}"; do
  if ! [[ -d $TRANSFORMERS_CACHE_DIR/models--${model/\//--} ]]; then
    # Continue running if download fails, e.g., if user is not authorized
    transformers-cli download --cache-dir "$TRANSFORMERS_CACHE_DIR" "$model" || true
  fi
done
