"""Model definitions for MRI and clinical fusion."""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50_MRI(nn.Module):
    """ResNet50 backbone for MRI images."""

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        """Forward pass returning logits."""
        return self.backbone(x)


class ClinicalMLP(nn.Module):
    """Simple MLP for clinical variables."""

    def __init__(self, in_features: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


class EarlyFusionTransformer(nn.Module):
    """Placeholder Early Fusion Transformer."""

    def __init__(self, in_features: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        self.fc_clin = nn.Linear(in_features, 16)
        self.head = nn.Linear(16 * 112 * 112 + 16, 2)

    def forward(self, img, clin):
        img_feat = torch.flatten(self.conv(img), 1)
        clin_feat = self.fc_clin(clin)
        feat = torch.cat([img_feat, clin_feat], dim=1)
        return self.head(feat)


class SimpleFusionMLP(nn.Module):
    """Baseline late fusion MLP combining MRI and clinical features."""

    def __init__(self, in_features: int = 2048, clin_features: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features + clin_features, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, img_feat, clin_feat):
        x = torch.cat([img_feat, clin_feat], dim=1)
        return self.net(x)


def get_model(name: str, **kwargs) -> nn.Module:
    """Factory method to build models by name."""
    if name == 'mri_only':
        return ResNet50_MRI(**kwargs)
    if name == 'clin_only':
        return ClinicalMLP(**kwargs)
    if name == 'early_fusion':
        return EarlyFusionTransformer(**kwargs)
    if name == 'mlp_fusion':
        return SimpleFusionMLP(**kwargs)
    raise ValueError(f'Unknown model {name}')
