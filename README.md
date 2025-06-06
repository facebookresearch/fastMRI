# 多模态 MRI + 临床特征在 AS 早期诊断中的应用

**作者**：<ZoyaHuo>

## 摘要
利用 ResNet50 等模型对 MRI 图像以及临床表型进行融合，评估其在强直脊柱炎（AS）早期诊断中的价值。提供了数据加载、模型训练、评估与绘图等脚本，可一键完成五折交叉验证及测试。

## 目录结构
```
.
├── LICENSE
├── README.md
├── requirements.txt
├── .gitignore
├── data/                # 数据目录（未纳入版本控制）
├── src/                 # 源码
│   ├── dataset.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│   └── __init__.py
├── scripts/
│   └── run_all.sh
├── notebooks/
└── archive/
```

## 依赖安装
```bash
pip install -r requirements.txt
```

## 数据准备
- `clinical.csv`：包含 `patient_id`、`label` 及若干临床变量。
- `mri/`：以 `patient_id.png` 命名的图像文件，或相应 `.h5` 格式（参见转换脚本）。

## 一键运行
```bash
bash scripts/run_all.sh
```

## 输出
- `checkpoints/`：各模型训练得到的权重文件。
- `results/`：交叉验证平均结果及测试指标、ROC/PR、校准曲线、DCA 图。

## 引用
- [fastMRI: An Open Dataset and Benchmarks for Accelerated MRI](https://arxiv.org/abs/1811.08839)
- [fastMRI GitHub](https://github.com/facebookresearch/fastMRI)
