"""Dataset utilities for MRI and clinical data."""

from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class MRIDataset(Dataset):
    """Dataset for MRI images stored as PNG or fastMRI H5 files."""

    def __init__(self, csv_path: Path, img_dir: Path, mode: str = "png", train: bool = True, augment: bool = True):
        """Initialize dataset.

        Parameters
        ----------
        csv_path: Path
            Path to clinical CSV with at least columns ['patient_id', 'label', ...].
        img_dir: Path
            Directory containing MRI images or H5 files.
        mode: str
            Either 'png' or 'fastmri_h5'.
        train: bool
            Whether dataset is for training.
        augment: bool
            If True, apply augmentation when in training mode.
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.mode = mode
        self.train = train
        self.augment = augment and train
        self.transform = self._build_transform()

    def _build_transform(self):
        tfms = [transforms.ToTensor()]
        if self.augment:
            tfms = [
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(p=0.5),
            ] + tfms
        return transforms.Compose(tfms)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pid = row['patient_id']
        label = row['label']
        if self.mode == 'png':
            img_path = self.img_dir / f"{pid}.png"
            img = Image.open(img_path).convert('RGB')
        else:
            # Placeholder: load slice from fastMRI H5 file
            img_path = self.img_dir / f"{pid}.h5"
            import h5py
            with h5py.File(img_path, 'r') as f:
                img = Image.fromarray(f['image'][()])
        img = self.transform(img)
        clinical = row.drop(['patient_id', 'label']).values.astype('float32')
        return img, clinical, label


def split_patient_kfold(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[List[int], List[int]]]:
    """Return indices for patient-level k-fold cross validation."""
    patients = df['patient_id'].unique()
    patients.sort()
    folds = []
    for i in range(n_splits):
        test_pids = patients[i::n_splits]
        train_pids = [p for p in patients if p not in test_pids]
        train_idx = df[df['patient_id'].isin(train_pids)].index.tolist()
        test_idx = df[df['patient_id'].isin(test_pids)].index.tolist()
        folds.append((train_idx, test_idx))
    return folds


def create_dataloader(csv_path: Path, img_dir: Path, batch_size: int, train: bool = True, augment: bool = True, mode: str = 'png') -> DataLoader:
    """Utility to create dataloader with default options."""
    dataset = MRIDataset(csv_path, img_dir, mode=mode, train=train, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=4)
