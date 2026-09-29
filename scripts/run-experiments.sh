#!/usr/bin/env bash
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY_ROOT"

if [[ ! -x .venv/bin/python ]]; then
    echo "Missing .venv. Run scripts/create-env.sh first." >&2
    exit 1
fi

PYTHON="$REPOSITORY_ROOT/.venv/bin/python"
BASE_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct" #"google/gemma-4-E2B-it"
TEACHER_MODEL="HuggingFaceTB/SmolLM2-360M-Instruct" #"google/gemma-4-12B-it"
SFT_RUN="${BASE_MODEL}-sft"
OPD_RUN="${SFT_RUN}-opd"
SFT_MODEL="$REPOSITORY_ROOT/outputs/$SFT_RUN/final_model"
OPD_MODEL="$REPOSITORY_ROOT/outputs/$OPD_RUN/final_model"

echo "Generating data..."
"$PYTHON" generate_data.py

echo "Evaluating base model..."
"$PYTHON" -m src.evaluation.evaluate_local model="$BASE_MODEL"

echo "Training SFT model..."
"$PYTHON" train_sft.py \
    train.model_name="$BASE_MODEL" \
    train.run_name="$SFT_RUN" \
    train.final_model.output_dir=outputs

echo "Evaluating SFT model..."
"$PYTHON" -m src.evaluation.evaluate_local model="$SFT_MODEL"

echo "Training OPD model..."
"$PYTHON" train_opd.py \
    train.model_name="$SFT_MODEL" \
    train.teacher_model="$TEACHER_MODEL" \
    train.run_name="$OPD_RUN" \
    train.final_model.output_dir=outputs

echo "Evaluating OPD model..."
"$PYTHON" -m src.evaluation.evaluate_local model="$OPD_MODEL"
