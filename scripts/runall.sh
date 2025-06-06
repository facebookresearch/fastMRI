#!/bin/bash
set -e

# 定义路径（可根据需要调整）
DATA_ROOT=data
NIFTI_DIR=${DATA_ROOT}/nifti
BIAS_DIR=${DATA_ROOT}/bias_corrected
ROI_DIR=${DATA_ROOT}/roi
AUG_DIR=${DATA_ROOT}/augmented
CLINICAL_CSV=clinical_data/subjects.csv
CLEANED_CSV=${DATA_ROOT}/cleaned_clinical.csv
CKPT_DIR=checkpoints
RESULT_DIR=results

echo "=========================="
echo "1. DICOM → NIfTI 转换"
echo "=========================="
python scripts/dicom_to_nifti.py --input ${DATA_ROOT}/dicom --output ${NIFTI_DIR}

echo "=========================="
echo "2. Bias Field N4 校正"
echo "=========================="
python scripts/bias_correction.py ${NIFTI_DIR} ${BIAS_DIR}

echo "=========================="
echo "3. 提取 ROI 区域"
echo "=========================="
python scripts/extract_roi.py ${BIAS_DIR} ${ROI_DIR}

echo "=========================="
echo "4. MRI 数据增强"
echo "=========================="
python scripts/augment_nifti.py ${ROI_DIR} ${AUG_DIR}

echo "=========================="
echo "5. 临床数据预处理"
echo "=========================="
python scripts/preprocess_clinical.py --csv ${CLINICAL_CSV} --output ${CLEANED_CSV}

echo "=========================="
echo "6. 训练 Early-Fusion Transformer"
echo "=========================="
python src/train.py --model early_fusion \
    --csv ${CLEANED_CSV} --img_dir ${AUG_DIR} --save_dir ${CKPT_DIR}/early

echo "=========================="
echo "7. 提取 Late-Fusion 特征 + 训练 XGBoost"
echo "=========================="
python src/train_late_fusion.py \
    --csv ${CLEANED_CSV} --img_dir ${AUG_DIR} --save_dir ${CKPT_DIR}/late

echo "=========================="
echo "8. 模型评估 + 可视化（SHAP / DCA）"
echo "=========================="
python src/evaluate.py --model early_fusion \
    --ckpt ${CKPT_DIR}/early/fold0.pt \
    --csv ${CLEANED_CSV} --img_dir ${AUG_DIR} --save_dir ${RESULT_DIR}/early

python scripts/plot_shap_dca.py \
    --csv ${CLEANED_CSV} \
    --img_dir ${AUG_DIR} \
    --save_dir ${RESULT_DIR}/early

echo "🎉 全部流程完成！结果已保存至 ${RESULT_DIR}"
