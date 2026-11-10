#!/bin/bash
# End-to-end test on example-data: binarize -> train stage 1 -> train stage 2 -> inference,
# for both approaches. Uses 50% of the example training data (--keep_frac 0.5).
# Run from the repo root, on a GPU node.
set -euo pipefail

export REPO_ROOT=$(realpath "$(dirname "$0")")
cd "$REPO_ROOT/metrics"

OUT=test_outputs
CFG=../configs/data_configs
TEST_CSV=$REPO_ROOT/example-data/wmt25/dev/wmt25_en-cs_CZ_esa-merged_dev.csv
TRAIN_ARGS="--num_train_epochs 1 --frozen_epochs 0 --warmup_steps 10 --eval_steps 50 --save_steps 50 --logging_steps 10 --report_to none --batch_size 2"


# ---------------- GemmaItXComet ----------------
GEMMA=google/gemma-3-1b-it
# data preparation
for stage in stage1_1.5 stage2; do
  python binarize_data_xcomet.py \
  --model_id $GEMMA \
  --save_path test_$stage \
  --keep_frac 0.5 \
  --files_yaml $CFG/example_$stage.yaml \
  --user_prompt ../configs/prompts/xcomet_user_prompt.txt
done

# stage 1
python train_GemmaItXComet.py \
  --model_id $GEMMA $TRAIN_ARGS \
  --primary_eval_dataset dev_paraf \
  --saved_ds_path databins/google_gemma-3-1b-it/test_stage1_1.5 --output_dir $OUT/gemma_stage1

# stage 2
python train_GemmaItXComet.py \
  --model_id $GEMMA $TRAIN_ARGS \
  --primary_eval_dataset dev_paraf \
  --saved_ds_path databins/google_gemma-3-1b-it/test_stage2 --output_dir $OUT/gemma_stage2 \
  --init_from $OUT/gemma_stage1

# inference
python inference/inference_regression-model.py \
  --archi GemmaItXComet \
  --ckpt $OUT/gemma_stage2 \
  --csv $TEST_CSV \
  --src_lang en \
  --tgt_lang cs \
  --output $OUT/gemma_predicted-scores.csv

# ---------------- QwenXComet ----------------
# QWEN=Qwen/Qwen3-Embedding-4B   # model used in our experiments (needs a larger GPU)
QWEN=Qwen/Qwen3-Embedding-0.6B
QWEN_DB=databins/${QWEN//\//_}

# data preparation
for stage in stage1_1.5 stage2; do
  python binarize_data_qwen.py \
    --model_id $QWEN \
    --save_path test_$stage \
    --keep_frac 0.5 \
    --files_yaml $CFG/example_$stage.yaml
done

# stage 1
python train_QwenXComet.py \
  --model_id $QWEN $TRAIN_ARGS \
  --primary_eval_dataset dev_paraf \
  --saved_ds_path $QWEN_DB/test_stage1_1.5 \
  --output_dir $OUT/qwen_stage1

# stage 2
python train_QwenXComet.py \
  --model_id $QWEN $TRAIN_ARGS \
  --primary_eval_dataset dev_paraf \
  --saved_ds_path $QWEN_DB/test_stage2 \
  --output_dir $OUT/qwen_stage2 \
  --init_from $OUT/qwen_stage1

# inference
python inference/inference_regression-model.py \
  --archi QwenXComet \
  --ckpt $OUT/qwen_stage2 \
  --csv $TEST_CSV \
  --src_lang en \
  --tgt_lang cs \
  --output $OUT/qwen_predicted-scores.csv

echo "All steps finished. Predicted quality scores: metrics/$OUT/{gemma,qwen}_predicted-scores.csv"
