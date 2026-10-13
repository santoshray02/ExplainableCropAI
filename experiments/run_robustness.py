#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Robustness to image perturbations (Round-2 point R2.17; R2.19 for baselines).

Applies eight corruptions at five severities to every test image, before the
standard test transform, and reports accuracy per (corruption, severity) for
each model, plus for the primary model: temperature-scaled ECE, AUROC of
confidence for correctness, and the referral rate at the moderate-certainty
threshold (confidence < 0.9), which tests whether the uncertainty mechanism
responds to degraded inputs.

Corruptions: brightness, contrast, gaussian_blur, jpeg, gaussian_noise,
rotation, occlusion, colour_temperature.

Outputs: robustness.csv (long format), robustness_summary.csv (mean corruption
error per model).

Usage (deepbeastx, SUPIR env):
  python experiments/run_robustness.py --backbones convnextv2_tiny resnet50 ...
"""
import os
import io
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, LogitsOnly, build_loader,
    PADDY_DISEASE_CLASSES)
from experiments.run_calibration import (  # noqa: E402
    expected_calibration_error, fit_temperature, collect_logits)
from experiments.run_baseline_eval import auroc  # noqa: E402
from src.data_loader import get_transforms  # noqa: E402

SEVERITIES = [1, 2, 3, 4, 5]
CORRUPTIONS = ['brightness', 'contrast', 'gaussian_blur', 'jpeg', 'gaussian_noise',
               'rotation', 'occlusion', 'colour_temperature']


def corrupt(img, name, s, rng):
    """Apply corruption `name` at severity s in 1..5 to a PIL image."""
    if name == 'brightness':
        f = [1.3, 1.6, 0.7, 0.5, 0.35][s - 1]
        return ImageEnhance.Brightness(img).enhance(f)
    if name == 'contrast':
        f = [0.8, 0.6, 0.45, 0.3, 0.2][s - 1]
        return ImageEnhance.Contrast(img).enhance(f)
    if name == 'gaussian_blur':
        r = [1, 2, 3, 4, 6][s - 1]
        return img.filter(ImageFilter.GaussianBlur(radius=r))
    if name == 'jpeg':
        q = [60, 40, 25, 15, 8][s - 1]
        buf = io.BytesIO(); img.save(buf, format='JPEG', quality=q); buf.seek(0)
        return Image.open(buf).convert('RGB')
    if name == 'gaussian_noise':
        sigma = [8, 16, 24, 32, 48][s - 1]
        a = np.asarray(img).astype(np.float32)
        a = a + rng.normal(0, sigma, a.shape)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if name == 'rotation':
        deg = [15, 30, 45, 90, 180][s - 1]
        return img.rotate(deg, resample=Image.BILINEAR, expand=False, fillcolor=(128, 128, 128))
    if name == 'occlusion':
        frac = [0.1, 0.2, 0.3, 0.4, 0.5][s - 1]
        a = np.asarray(img).copy(); h, w = a.shape[:2]
        area = frac * h * w
        side = int(np.sqrt(area / 4))  # four square patches
        for _ in range(4):
            y = rng.randint(0, max(1, h - side)); x = rng.randint(0, max(1, w - side))
            a[y:y + side, x:x + side] = 128
        return Image.fromarray(a)
    if name == 'colour_temperature':
        # warm (odd severities) / cool (even) shift of increasing strength
        k = [0.08, 0.12, 0.18, 0.24, 0.32][s - 1]
        gain = np.array([1 + k, 1.0, 1 - k]) if s % 2 else np.array([1 - k, 1.0, 1 + k])
        a = np.asarray(img).astype(np.float32) * gain
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    raise ValueError(name)


class CorruptedTest(Dataset):
    def __init__(self, rows, name, severity, input_size, seed):
        self.rows, self.name, self.s, self.seed = rows, name, severity, seed
        self.tf = get_transforms('test', input_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, y = self.rows[i]
        img = Image.open(p).convert('RGB')
        if self.name != 'clean':
            rng = np.random.RandomState(self.seed * 1000 + i)
            img = corrupt(img, self.name, self.s, rng)
        return self.tf(img), y


@torch.no_grad()
def run(model, loader, device, T):
    probs, ys = [], []
    for x, y in loader:
        out = model(x.to(device))
        logits = out[0] if isinstance(out, tuple) else out
        probs.append(F.softmax(logits / T, 1).cpu()); ys.append(y)
    probs = torch.cat(probs).numpy(); ys = torch.cat(ys).numpy()
    pred = probs.argmax(1); conf = probs.max(1); correct = pred == ys
    ece, _, _ = expected_calibration_error(conf, pred, ys)
    return {'accuracy': float(correct.mean()), 'ece': float(ece),
            'auroc_conf_correct': auroc(conf, correct),
            'referral_rate_conf<0.9': float((conf < 0.9).mean()),
            'referral_rate_conf<0.5': float((conf < 0.5).mean()),
            'mean_conf': float(conf.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--checkpoint-dir', default='experiments/outputs')
    ap.add_argument('--backbones', nargs='+', default=['convnextv2_tiny'])
    ap.add_argument('--corruptions', nargs='+', default=CORRUPTIONS)
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.splits_csv)
    test = df[df['split'] == 'test']
    rows = [(p, PADDY_DISEASE_CLASSES.index(l)) for p, l in zip(test['image_path'], test['label'])]
    print(f'{len(rows)} test images', flush=True)
    val_loader = build_loader(args.splits_csv, 'val', args.batch_size)

    out_path = os.path.join(args.output_dir, 'robustness.csv')
    results = []
    for bb in args.backbones:
        ckpt = os.path.join(args.checkpoint_dir, f'baseline_{bb}.pth')
        raw = load_checkpoint(build_model(bb, args.num_classes, pretrained=False), ckpt, device)
        model = LogitsOnly(raw).to(device).eval()
        vl, vy = collect_logits(model, val_loader, device)
        T = fit_temperature(vl, vy, device)
        clean = run(model, DataLoader(CorruptedTest(rows, 'clean', 0, args.input_size, args.seed),
                                      batch_size=args.batch_size, num_workers=args.num_workers), device, T)
        results.append({'backbone': bb, 'corruption': 'clean', 'severity': 0, **clean})
        print(results[-1], flush=True)
        for name in args.corruptions:
            for s in SEVERITIES:
                loader = DataLoader(CorruptedTest(rows, name, s, args.input_size, args.seed),
                                    batch_size=args.batch_size, num_workers=args.num_workers)
                r = run(model, loader, device, T)
                results.append({'backbone': bb, 'corruption': name, 'severity': s, **r})
                print(results[-1], flush=True)
            pd.DataFrame(results).to_csv(out_path, index=False)
        del raw, model; torch.cuda.empty_cache()

    res = pd.DataFrame(results)
    res.to_csv(out_path, index=False)
    summ = []
    for bb in args.backbones:
        sub = res[res['backbone'] == bb]
        clean_acc = float(sub[sub['corruption'] == 'clean']['accuracy'].iloc[0])
        cor = sub[sub['corruption'] != 'clean']
        row = {'backbone': bb, 'clean_accuracy': round(clean_acc * 100, 2),
               'mean_corrupted_accuracy': round(cor['accuracy'].mean() * 100, 2),
               'mean_corruption_error': round((1 - cor['accuracy']).mean() * 100, 2),
               'relative_drop_points': round((clean_acc - cor['accuracy'].mean()) * 100, 2)}
        for name in args.corruptions:
            row[f'acc_{name}'] = round(cor[cor['corruption'] == name]['accuracy'].mean() * 100, 2)
        summ.append(row)
    pd.DataFrame(summ).to_csv(os.path.join(args.output_dir, 'robustness_summary.csv'), index=False)
    print(pd.DataFrame(summ).to_string(index=False))


if __name__ == '__main__':
    main()
