#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/data0/zhongxiang/LLM-ERI-repro
DSV_ROOT=/data0/zhongxiang/DSV-LFS-main
PY=/data0/zhongxiang/conda/envs/gla/bin/python
DEEPSPEED=/data0/zhongxiang/conda/envs/gla/bin/deepspeed
DS_NUM_GPUS=${DS_NUM_GPUS:-2}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DSV_ROOT
export PATH="/data0/zhongxiang/conda/envs/gla/bin:${PATH}"
export PYTHONPATH="${DSV_ROOT}:${DSV_ROOT}/model:${PYTHONPATH:-}"

cd "${DSV_ROOT}"

OUT_DIR="${PROJECT_ROOT}/runs/voc_stage1_full_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

nohup "${DEEPSPEED}" --num_gpus="${DS_NUM_GPUS}" "${PROJECT_ROOT}/scripts/train_weakly_compat.py" \
  --version "${DSV_ROOT}/llava-v1.5-7b" \
  --dataset_dir /data0/zhongxiang/MCT+/VOC2012 \
  --vision_pretrained "${DSV_ROOT}/sam_vit_h_4b8939.pth" \
  --vision-tower "${DSV_ROOT}/clip-vit-large-patch14-336" \
  --benchmark pascal \
  --exp_name "$(basename "${OUT_DIR}")" \
  --log_base_dir "${PROJECT_ROOT}/runs" \
  --epochs 20 \
  --steps_per_epoch 5291 \
  --batch_size 1 \
  --grad_accumulation_steps 1 \
  --workers 4 \
  --precision bf16 \
  --lr 2e-5 \
  --lora_r 0 \
  --image_size 512 \
  --print_freq 20 > "${OUT_DIR}/train.log" 2>&1 &

PID=$!
echo "${PID}" > "${OUT_DIR}/pid.txt"
echo "${OUT_DIR}" > "${PROJECT_ROOT}/outputs/voc_stage1_full_latest.txt"
echo "started ${PID} ${OUT_DIR}"
