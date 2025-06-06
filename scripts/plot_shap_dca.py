import argparse
import os
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

def plot_shap(csv_file, save_dir):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=["patient_id", "label"], errors="ignore")
    y = df["label"] if "label" in df.columns else np.random.randint(0, 2, size=len(X))  # fallback

    model = LogisticRegression().fit(X, y)
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    plt.figure(figsize=(8, 4))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "shap_bar.png"))
    print("✅ SHAP 图已保存 →", os.path.join(save_dir, "shap_bar.png"))


def plot_dca(csv_file, save_dir):
    df = pd.read_csv(csv_file)
    X = df.drop(columns=["patient_id", "label"], errors="ignore")
    y = df["label"]

    model = LogisticRegression().fit(X, y)
    probs = model.predict_proba(X)[:, 1]

    thresholds = np.linspace(0.01, 0.99, 100)
    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = [0] * len(thresholds)

    for t in thresholds:
        pred = probs >= t
        TP = ((pred == 1) & (y == 1)).sum()
        FP = ((pred == 1) & (y == 0)).sum()
        n = len(y)
        nb = TP / n - FP / n * (t / (1 - t))
        net_benefit_model.append(nb)
        nb_all = y.mean() - (1 - y.mean()) * (t / (1 - t))
        net_benefit_all.append(nb_all)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, net_benefit_model, label="Model", color="blue")
    plt.plot(thresholds, net_benefit_all, label="Treat All", linestyle="--", color="gray")
    plt.plot(thresholds, net_benefit_none, label="Treat None", linestyle="--", color="black")
    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dca_curve.png"))
    print("✅ DCA 图已保存 →", os.path.join(save_dir, "dca_curve.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="临床特征 + label 的 CSV 路径")
    parser.add_argument("--save_dir", required=True, help="图像保存目录")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("📊 开始生成 SHAP 图")
    plot_shap(args.csv, args.save_dir)

    print("📈 开始生成 DCA 曲线图")
    plot_dca(args.csv, args.save_dir)
