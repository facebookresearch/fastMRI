"""Evaluation utilities."""

import argparse
from pathlib import Path
import json
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    recall_score,
    confusion_matrix,
)

from .dataset import create_dataloader, MRIDataset
from .models import get_model
from .utils import (
    plot_roc_pr,
    plot_calibration_curve,
    decision_curve_analysis,
    bootstrap_ci,
    save_json,
)


def evaluate_checkpoint(model_name: str, ckpt_path: Path, csv: Path, img_dir: Path, mode: str, out_dir: Path):
    loader = create_dataloader(csv, img_dir, batch_size=16, train=False, augment=False, mode=mode)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(model_name)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    y_true, prob = [], []
    with torch.no_grad():
        for img, clin, label in loader:
            img, clin = img.to(device), clin.to(device)
            if len(model.forward.__code__.co_varnames) == 3:
                output = model(img, clin)
            else:
                output = model(img)
            prob.append(torch.softmax(output, 1)[:, 1].cpu())
            y_true.append(label)
    prob = torch.cat(prob).numpy()
    y_true = torch.cat(y_true).numpy()

    roc = roc_auc_score(y_true, prob)
    pr = average_precision_score(y_true, prob)
    brier = brier_score_loss(y_true, prob)
    acc = accuracy_score(y_true, prob > 0.5)
    sens = recall_score(y_true, prob > 0.5)
    tn, fp, fn, tp = confusion_matrix(y_true, prob > 0.5).ravel()
    spec = tn / (tn + fp)
    roc_ci = bootstrap_ci(roc_auc_score, y_true, prob)
    pr_ci = bootstrap_ci(average_precision_score, y_true, prob)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_pr(y_true, prob, str(out_dir / 'curve'))
    plot_calibration_curve(y_true, prob, str(out_dir / 'calibration.png'))
    decision_curve_analysis(y_true, prob, str(out_dir / 'dca.png'))

    metrics = {
        'roc_auc': roc,
        'pr_auc': pr,
        'brier': brier,
        'accuracy': acc,
        'sensitivity': sens,
        'specificity': spec,
        'roc_auc_ci': roc_ci.tolist(),
        'pr_auc_ci': pr_ci.tolist(),
    }
    save_json(metrics, out_dir / 'test_metrics.json')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--csv', type=Path, default=Path('data/clinical.csv'))
    parser.add_argument('--img_dir', type=Path, default=Path('data/mri'))
    parser.add_argument('--mode', type=str, default='png')
    parser.add_argument('--out_dir', type=Path, default=Path('results'))
    args = parser.parse_args()

    evaluate_checkpoint(args.model, args.ckpt, args.csv, args.img_dir, args.mode, args.out_dir)
