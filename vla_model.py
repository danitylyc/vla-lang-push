"""VLA-Push model: a from-scratch vision-language-action policy.

Architecture (all modules written from scratch in PyTorch):
  * vision: 4-layer CNN encoder (64x64 RGB -> 256-d visual feature)
  * language: small Transformer encoder (token ids -> 64-d instruction feature)
  * fusion: concat(visual, language) -> MLP -> 3-D end-effector delta action

Regression target is the (normalized [-1, 1]) 3-D action used by panda-gym.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# vision encoder (CNN)
# ---------------------------------------------------------------------------
class VisionEncoder(nn.Module):
    def __init__(self, img: int = 64, feat_dim: int = 256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),  # 64 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(inplace=True),
        )
        h = img // 16
        self.fc = nn.Linear(256 * h * h, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,3,64,64) -> (B,feat)
        x = self.convs(x)
        x = x.flatten(1)
        return F.relu(self.fc(x))


# ---------------------------------------------------------------------------
# language encoder (small Transformer)
# ---------------------------------------------------------------------------
class LanguageEncoder(nn.Module):
    def __init__(self, vocab_size: int, max_len: int = 8, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, ff_dim: int = 128):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
            batch_first=True, dropout=0.0, activation="relu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.cls_proj = nn.Linear(d_model, d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:  # (B,L) -> (B,d_model)
        B, L = ids.shape
        x = self.embed(ids) * math.sqrt(self.d_model) + self.pos[:, :L]
        x = self.encoder(x)
        cls = x[:, 0]  # <cls> token output
        return F.relu(self.cls_proj(cls))


# ---------------------------------------------------------------------------
# VLA policy: vision + language -> action
# ---------------------------------------------------------------------------
class VLAPolicy(nn.Module):
    def __init__(self, vocab_size: int, img: int = 64, vis_dim: int = 256,
                 lang_dim: int = 64, hidden: int = 256, action_dim: int = 3):
        super().__init__()
        self.vision = VisionEncoder(img=img, feat_dim=vis_dim)
        self.language = LanguageEncoder(vocab_size=vocab_size, max_len=8, d_model=lang_dim)
        self.fusion = nn.Sequential(
            nn.Linear(vis_dim + lang_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, rgb: torch.Tensor, instr: torch.Tensor) -> torch.Tensor:
        """rgb: (B,3,H,W) uint8-normalized float; instr: (B,L) int64 token ids."""
        v = self.vision(rgb)
        l = self.language(instr)
        return self.fusion(torch.cat([v, l], dim=-1))
