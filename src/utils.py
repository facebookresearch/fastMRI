"""Utility functions for training and evaluation."""

import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.calibration import calibration_curve


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_roc_pr(y_true, prob, out_prefix):
    fpr, tpr, _ = roc_curve(y_true, prob)
    RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig(f"{out_prefix}_roc.png")
    plt.close()
    precision, recall, _ = precision_recall_curve(y_true, prob)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.savefig(f"{out_prefix}_pr.png")
    plt.close()


def plot_calibration_curve(y_true, prob, out_png):
    prob_true, prob_pred = calibration_curve(y_true, prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0, 1], [0, 1], '--')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(out_png)
    plt.close()


def decision_curve_analysis(y_true, prob, out_png):
    thresholds = np.linspace(0.01, 0.99, 50)
    net_benefit = []
    prevalence = y_true.mean()
    for t in thresholds:
        tp = ((prob >= t) & (y_true == 1)).sum()
        fp = ((prob >= t) & (y_true == 0)).sum()
        nb = tp / len(y_true) - fp / len(y_true) * t / (1 - t)
        net_benefit.append(nb)
    plt.plot(thresholds, net_benefit)
    plt.xlabel('Threshold')
    plt.ylabel('Net Benefit')
    plt.savefig(out_png)
    plt.close()


def bootstrap_ci(metric_fn, y_true, prob, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        scores.append(metric_fn(y_true[idx], prob[idx]))
    scores = np.array(scores)
    return np.percentile(scores, [2.5, 97.5])
