#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline comparison and component ablation.
Addresses reviewer point #11 (stronger experimental comparisons & ablations).

Two modes:

  --mode baselines
      Train + evaluate several backbones under IDENTICAL training settings:
      ConvNeXt V2-Tiny (ours), ResNet50, EfficientNet-B0, Inception-v3,
      plus lightweight (MobileViT, EfficientFormer) and a fine-grained option.
      Reports accuracy / macro-F1 / params / inference latency.

  --mode ablation
      Take the chosen backbone and toggle pipeline components, reporting the
      measurable benefit of each:
        backbone-only -> +explanation-loss -> +MC-dropout uncertainty
      (NLG and recommendation are evaluated on their own metrics; see notes.)

Each configuration is trained with the same optimiser, schedule, epochs and
augmentation so differences are attributable to the component under test.

Usage
-----
# Baselines
python experiments/run_baselines_ablation.py --mode baselines \
    --splits-csv experiments/outputs/splits.csv --epochs 40 \
    --backbones convnextv2_tiny resnet50 efficientnet_b0 inception_v3 mobilevit_s

# Ablation
python experiments/run_baselines_ablation.py --mode ablation \
    --splits-csv experiments/outputs/splits.csv --epochs 40 \
    --backbone convnextv2_tiny
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, build_loader, LogitsOnly)


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6  # millions


@torch.no_grad()
def measure_latency(model, device, input_size=224, iters=50):
    model.eval()
    x = torch.randn(1, 3, input_size, input_size, device=device)
    for _ in range(5):  # warmup
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000.0  # ms


def explanation_consistency_loss(activations):
    """Lightweight 'explanation loss' regulariser: encourages spatially smooth,
    concentrated activation maps (total-variation penalty). Stand-in for the
    explanation-guided term ablated in the paper; replace with the project's
    actual loss if different."""
    if activations is None or activations.dim() != 4:
        return torch.tensor(0.0)
    a = activations.mean(1)  # (B,H,W)
    tv = (a[:, 1:, :] - a[:, :-1, :]).abs().mean() + \
         (a[:, :, 1:] - a[:, :, :-1]).abs().mean()
    return tv


def train_one(model, train_loader, val_loader, device, epochs, lr,
              use_explanation_loss=False, expl_weight=0.05):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    best_val = 0.0
    best_state = None
    for ep in range(epochs):
        model.train()
        for batch in train_loader:
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            opt.zero_grad()
            out = model(x)
            logits, extras = (out if isinstance(out, tuple) else (out, {}))
            loss = ce(logits, y)
            if use_explanation_loss:
                loss = loss + expl_weight * explanation_consistency_loss(
                    extras.get('activations')).to(loss.device)
            loss.backward()
            opt.step()
        sched.step()
        val_acc = evaluate(model, val_loader, device)['accuracy']
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {ep+1}/{epochs}  val_acc={val_acc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate(model, loader, device, mc_samples=0):
    model.eval()
    preds, labels = [], []
    if mc_samples > 0:
        for mod in model.modules():
            if isinstance(mod, nn.Dropout):
                mod.train()
    for batch in loader:
        x = batch['image'].to(device)
        if mc_samples > 0:
            ps = []
            for _ in range(mc_samples):
                out = model(x)
                logits = out[0] if isinstance(out, tuple) else out
                ps.append(F.softmax(logits, 1))
            probs = torch.stack(ps).mean(0)
        else:
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, 1)
        preds.append(probs.argmax(1).cpu().numpy())
        labels.append(batch['label'].numpy())
    preds = np.concatenate(preds); labels = np.concatenate(labels)
    return {'accuracy': accuracy_score(labels, preds),
            'macro_f1': f1_score(labels, preds, average='macro')}


def run_baselines(args, device):
    train_loader = build_loader(args.splits_csv, 'train', args.batch_size, shuffle=True)
    val_loader = build_loader(args.splits_csv, 'val', args.batch_size)
    test_loader = build_loader(args.splits_csv, 'test', args.batch_size)

    rows = []
    for bb in args.backbones:
        print(f"\n=== Baseline: {bb} ===")
        try:
            set_seed(args.seed)
            model = build_model(bb, args.num_classes, drop_rate=0.3, pretrained=True)
            params = count_params(model)
            model = train_one(model, train_loader, val_loader, device, args.epochs, args.lr)
            test = evaluate(model, test_loader, device)
            lat = measure_latency(model, device, args.input_size)
            rows.append({'backbone': bb, 'params_M': round(params, 1),
                         'latency_ms': round(lat, 2), **test})
            torch.save({'model_state_dict': model.state_dict(), 'config': {'backbone': bb}},
                       os.path.join(args.output_dir, f'baseline_{bb}.pth'))
            print(f"  test: {test}")
            # incremental save so a later failure never loses earlier results
            pd.DataFrame(rows).sort_values('accuracy', ascending=False).to_csv(
                os.path.join(args.output_dir, 'baselines.csv'), index=False)
        except Exception as e:
            print(f"  [SKIP] {bb} failed: {e}")
            continue

    df = pd.DataFrame(rows).sort_values('accuracy', ascending=False)
    df.to_csv(os.path.join(args.output_dir, 'baselines.csv'), index=False)
    print("\n=== Baseline comparison ===")
    print(df.to_string(index=False))


def run_ablation(args, device):
    train_loader = build_loader(args.splits_csv, 'train', args.batch_size, shuffle=True)
    val_loader = build_loader(args.splits_csv, 'val', args.batch_size)
    test_loader = build_loader(args.splits_csv, 'test', args.batch_size)

    configs = [
        {'name': 'backbone_only', 'expl': False, 'mc': 0},
        {'name': '+explanation_loss', 'expl': True, 'mc': 0},
        {'name': '+explanation_loss+mc_dropout', 'expl': True, 'mc': args.mc_samples},
    ]
    rows = []
    for cfg in configs:
        print(f"\n=== Ablation: {cfg['name']} ===")
        set_seed(args.seed)
        model = build_model(args.backbone, args.num_classes, drop_rate=0.3, pretrained=True)
        model = train_one(model, train_loader, val_loader, device, args.epochs, args.lr,
                          use_explanation_loss=cfg['expl'])
        test = evaluate(model, test_loader, device, mc_samples=cfg['mc'])
        rows.append({'config': cfg['name'], **test})
        print(f"  test: {test}")

    df = pd.DataFrame(rows)
    # delta vs backbone-only
    base_acc = df.iloc[0]['accuracy']
    df['delta_acc'] = df['accuracy'] - base_acc
    df.to_csv(os.path.join(args.output_dir, 'ablation.csv'), index=False)
    print("\n=== Ablation (note: NLG & recommendation are evaluated on their own "
          "metrics in run_explanation_comparison, not classification accuracy) ===")
    print(df.to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['baselines', 'ablation'], required=True)
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--backbones', nargs='+',
                    default=['convnextv2_tiny', 'resnet50', 'efficientnet_b0',
                             'inception_v3', 'mobilevit_s'])
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--mc-samples', type=int, default=30)
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    if args.mode == 'baselines':
        run_baselines(args, device)
    else:
        run_ablation(args, device)


if __name__ == '__main__':
    main()
