"""Model definitions for MRI and clinical fusion."""

import torch
import torch.nn as nn
import torchvision.models as models


# ========= 1️⃣ 图像模型（用于 MRI-only 或提取图像特征） =========
class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=128):
        super().__init__()
        backbone = models.resnet18(pretrained=pretrained)
        modules = list(backbone.children())[:-1]  # 去除 fc 层
        self.feature_extractor = nn.Sequential(*modules)
        self.projector = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)      # [B, 512]
        return self.projector(x)       # [B, output_dim]


# ========= 2️⃣ 临床模型（用于提取结构化临床变量） =========
class ClinicalMLP(nn.Module):
    def __init__(self, in_features: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)  # [B, hidden]


# ========= 3️⃣ ✅ Early-Fusion Transformer 模型 =========
class EarlyFusionTransformer(nn.Module):
    def __init__(self, clin_feat_dim=4, hidden_dim=128, nhead=4, num_layers=2):
        super().__init__()

        self.img_encoder = ResNet18Encoder(pretrained=True, output_dim=hidden_dim)
        self.clin_encoder = ClinicalMLP(in_features=clin_feat_dim, hidden=hidden_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, 2, hidden_dim))  # 2 tokens: image, clinical

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="relu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, img, clin):
        img_feat = self.img_encoder(img)   # [B, hidden]
        clin_feat = self.clin_encoder(clin)  # [B, hidden]
        tokens = torch.stack([img_feat, clin_feat], dim=1)  # [B, 2, hidden]
        tokens = tokens + self.pos_embedding
        fused = self.transformer(tokens)  # [B, 2, hidden]
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


# ========= 4️⃣ ✅ Late-Fusion XGBoost 支持模型结构 =========
class FeatureExtractorLateFusion(nn.Module):
    """
    用于 Late-Fusion 的模型，不直接分类，而是提取两个特征分支：
    - img_feat: ResNet18 图像特征
    - clin_feat: 临床变量 MLP 特征
    然后你可以将两个拼接 → 输入 XGBoost 分类器
    """

    def __init__(self, clin_feat_dim=4, output_dim=64):
        super().__init__()
        self.img_encoder = ResNet18Encoder(pretrained=True, output_dim=output_dim)
        self.clin_encoder = ClinicalMLP(in_features=clin_feat_dim, hidden=output_dim)

    def forward(self, img, clin):
        img_feat = self.img_encoder(img)     # [B, D]
        clin_feat = self.clin_encoder(clin)  # [B, D]
        return img_feat, clin_feat  # 由外部代码拼接后喂给 XGBoost

def get_model(name: str, **kwargs) -> nn.Module:
    if name == 'mri_only':
        return ResNet18Encoder(**kwargs)
    if name == 'clin_only':
        return ClinicalMLP(**kwargs)
    if name == 'early_fusion':
        return EarlyFusionTransformer(**kwargs)
    if name == 'late_fusion_features':
        return FeatureExtractorLateFusion(**kwargs)
    raise ValueError(f'Unknown model {name}')

