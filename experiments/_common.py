#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Shared helpers for the ExplainableCropAI revision experiments.

These utilities are used by all `run_*.py` scripts in this directory:
  - run_explanation_comparison.py   (reviewer point #7)
  - run_calibration.py              (reviewer point #8)
  - run_baselines_ablation.py       (reviewer point #11)

Design notes
------------
The project model (`src/models/convnext_v2.py`) returns a 2-tuple
``(logits, additional_outputs)`` from ``forward``. Off-the-shelf
attribution / calibration libraries (Captum, LIME, SHAP, pytorch-grad-cam)
expect a model that returns a single logits tensor. ``LogitsOnly`` wraps any
such model so those libraries work unchanged.

All randomness is seeded (default 42) so every script is reproducible, which
directly answers reviewer point #14 (reproducibility).
"""

import os
import sys
import random
import argparse

import numpy as np
import torch
import torch.nn as nn

# Make `src` importable regardless of where the script is launched from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

PADDY_DISEASE_CLASSES = [
    'bacterial_leaf_blight',
    'bacterial_leaf_streak',
    'bacterial_panicle_blight',
    'blast',
    'brown_spot',
    'dead_heart',
    'downy_mildew',
    'hispa',
    'normal',
    'tungro',
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def set_seed(seed: int = 42):
    """Seed every RNG we touch so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slightly slower, but reproducible).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class LogitsOnly(nn.Module):
    """Wrap a (logits, extras)-returning model so it returns logits only.

    Captum / LIME / SHAP / pytorch-grad-cam all assume ``model(x) -> Tensor``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            return out[0]
        return out


def build_model(backbone: str = 'convnextv2_tiny', num_classes: int = 10,
                drop_rate: float = 0.3, pretrained: bool = True):
    """Construct a classifier. ConvNeXt V2-Tiny is the project default;
    other timm backbones are used as baselines in #11.

    The returned model keeps the project's ``(logits, extras)`` contract so the
    repo's Grad-CAM++/LayerCAM code and MCDropout work unchanged. Wrap with
    ``LogitsOnly`` for third-party libraries.
    """
    if backbone == 'convnextv2_tiny':
        from src.models.convnext_v2 import create_model
        return create_model({'num_classes': num_classes,
                             'pretrained': pretrained,
                             'drop_rate': drop_rate})

    if backbone == 'resnet50':
        # The original project backbone, kept as a baseline.
        from src.base_model import PaddyDiseaseClassifier
        return PaddyDiseaseClassifier(num_classes=num_classes, pretrained=pretrained)

    # Generic timm backbone wrapped to honour the (logits, extras) contract.
    return _TimmClassifier(backbone, num_classes, drop_rate, pretrained)


class _TimmClassifier(nn.Module):
    """Minimal timm wrapper for the additional baselines (#11).

    Uses timm's native classifier head, which is correct across all
    architectures -- including VGG-style models that have a pre-logits MLP
    (where ``forward_features`` channels != ``num_features``). Returns
    ``(logits, extras)`` to honour the project's forward contract; the
    baselines only consume the logits.
    """

    def __init__(self, backbone: str, num_classes: int, drop_rate: float, pretrained: bool):
        super().__init__()
        import timm
        self.model = timm.create_model(
            backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)

    def forward(self, x):
        logits = self.model(x)
        return logits, {}


def load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load weights saved by the training script.

    Accepts checkpoints saved as a raw state_dict, or under
    'model_state_dict' / 'state_dict'.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load_checkpoint] WARNING missing keys: {missing}")
    if unexpected:
        print(f"[load_checkpoint] WARNING unexpected keys: {unexpected}")
    model.to(device).eval()
    return model


def _paddy_collate(batch):
    """Collate that keeps image/label tensors and path/class_name as lists,
    dropping the PIL 'original_image' field returned by PaddyDiseaseDataset
    (not needed for training/eval and not collatable by torch.default_collate)."""
    return {
        'image': torch.stack([b['image'] for b in batch]),
        'label': torch.stack([b['label'] for b in batch]),
        'path': [b['path'] for b in batch],
        'class_name': [b['class_name'] for b in batch],
    }


def build_loader(csv_file: str, split: str, batch_size: int = 32,
                 num_workers: int = 4, input_size: int = 224, shuffle: bool = False):
    """Return a DataLoader for one split of a splits.csv produced by
    experiments/create_splits.py (columns: image_path,label,split).

    The train split gets augmentation transforms; val/test get deterministic
    resize/center-crop.
    """
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.data_loader import PaddyDiseaseDataset, get_transforms

    df = pd.read_csv(csv_file)
    if 'split' in df.columns:
        df = df[df['split'] == split]
    tmp = csv_file.replace('.csv', f'_{split}.csv')
    df.to_csv(tmp, index=False)
    tf_mode = 'train' if split == 'train' else 'test'
    ds = PaddyDiseaseDataset(csv_file=tmp, transform=get_transforms(tf_mode, input_size), mode=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=_paddy_collate)


def common_args(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to trained model checkpoint (.pth)')
    p.add_argument('--splits-csv', type=str, default='experiments/outputs/splits.csv',
                   help='CSV with image_path,label,split columns')
    p.add_argument('--backbone', type=str, default='convnextv2_tiny')
    p.add_argument('--num-classes', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output-dir', type=str, default='experiments/outputs')
    p.add_argument('--cpu', action='store_true', help='Force CPU')
    return p
