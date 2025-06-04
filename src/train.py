"""Training script."""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .dataset import MRIDataset, split_patient_kfold, create_dataloader
from .models import get_model
from .utils import seed_everything, save_json


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    for img, clin, label in loader:
        img, clin, label = img.to(device), clin.to(device), label.to(device)
        optimizer.zero_grad()
        if isinstance(model, nn.Module) and len(model.forward.__code__.co_varnames) == 3:
            output = model(img, clin)
        else:
            output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for img, clin, label in loader:
            img, clin = img.to(device), clin.to(device)
            if isinstance(model, nn.Module) and len(model.forward.__code__.co_varnames) == 3:
                output = model(img, clin)
            else:
                output = model(img)
            preds.append(torch.softmax(output, 1)[:, 1].cpu())
            labels.append(label)
    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = accuracy_score(labels, preds > 0.5)
    return acc


def run_fold(fold, train_idx, test_idx, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = MRIDataset(args.csv, args.img_dir, mode=args.mode, train=True, augment=True).df
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]
    train_df.to_csv('fold_train.csv', index=False)
    test_df.to_csv('fold_test.csv', index=False)
    train_loader = create_dataloader('fold_train.csv', args.img_dir, args.batch_size, True, True, args.mode)
    val_loader = create_dataloader('fold_test.csv', args.img_dir, args.batch_size, False, False, args.mode)
    model = get_model(args.model)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, val_loader, device)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), Path(args.save_dir) / f'fold{fold}_{args.model}.pt')
    return best_acc


def main(args):
    seed_everything()
    df = MRIDataset(args.csv, args.img_dir, mode=args.mode, train=True, augment=True).df
    folds = split_patient_kfold(df, 5)
    if args.fold >= 0:
        train_idx, test_idx = folds[args.fold]
        acc = run_fold(args.fold, train_idx, test_idx, args)
        print({'accuracy': acc})
    else:
        accs = []
        for i, (train_idx, test_idx) in enumerate(folds):
            acc = run_fold(i, train_idx, test_idx, args)
            accs.append(acc)
        save_json({'accuracy': sum(accs)/len(accs)}, Path('results')/ 'cv_results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mri_only')
    parser.add_argument('--csv', type=Path, default=Path('data/clinical.csv'))
    parser.add_argument('--img_dir', type=Path, default=Path('data/mri'))
    parser.add_argument('--mode', type=str, default='png')
    parser.add_argument('--fold', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=Path, default=Path('checkpoints'))
    args = parser.parse_args()
    main(args)
