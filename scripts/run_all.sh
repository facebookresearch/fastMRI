#!/usr/bin/env bash
set -e
DATA_ROOT=data
CKPT_DIR=checkpoints
LOG_DIR=logs

# 0. 预创建目录
mkdir -p ${CKPT_DIR} ${LOG_DIR} results

# 1. 五折交叉验证训练 & 评估
for MODEL in mri_only clin_only early_fusion mlp_fusion; do
  python src/train.py --model ${MODEL} --csv ${DATA_ROOT}/clinical.csv --img_dir ${DATA_ROOT}/mri --fold -1 --save_dir ${CKPT_DIR}
done

# 2. Late-Fusion 基于前四个模型输出训练 XGBoost
python src/train.py --model late_fusion --csv ${DATA_ROOT}/clinical.csv --img_dir ${DATA_ROOT}/mri --save_dir ${CKPT_DIR}

# 3. 综合评估并绘图
python src/evaluate.py --model mri_only --ckpt ${CKPT_DIR}/fold0_mri_only.pt --csv ${DATA_ROOT}/clinical.csv --img_dir ${DATA_ROOT}/mri
