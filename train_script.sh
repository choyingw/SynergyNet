#!/usr/bin/env bash

LOG_ALIAS=$1
LOG_DIR="ckpts/logs"
mkdir -p ${LOG_DIR}

LOG_FILE="${LOG_DIR}/`date +'%Y-%m-%d_%H:%M.%S'`.log"

python3 main_train.py \
    --start-epoch=1 \
    --snapshot="ckpts/SynergyNet" \
    --param-fp-train='./3dmm_data/param_all_norm_v201.pkl' \
    --warmup=5 \
    --base-lr=0.027 \
    --epochs=50 \
    --milestones=30,40 \
    --print-freq=50 \
    --devices-id=0 \
    --workers=8 \
    --filelists-train="./3dmm_data/train_aug_120x120.list.train" \
    --root="./train_aug_120x120" \
    --log-file="${LOG_FILE}" \
    --test_initial=True \
    --save_val_freq=5 \
    --resume="" \
    --batch-size=16 \
    --arch="dcnv2" \
    --arch="dcnv1" \
    --snapshot="ckpts/dcnv2_SynergyNet" \
    --snapshot="ckpts/dcnv1_SynergyNet" \
