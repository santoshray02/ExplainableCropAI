#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model-based faithfulness comparison of attribution methods (NO expert masks).

Reviewer point #7, model-based variant: since no expert symptom masks were
collected, we compare the six attribution methods on automatic, model-based
metrics that need no human annotation -- insertion / deletion AUC
(Petsiuk et al., 2018) and wall-clock runtime -- over a sample of test images.
Produces Table `tab:visual-explanation` (Method | Insertion | Deletion | Runtime).

Usage
-----
python experiments/run_explanation_faithfulness.py \
    --checkpoint experiments/outputs/baseline_convnextv2_tiny.pth \
    --splits-csv experiments/outputs/splits.csv \
    --backbone convnextv2_tiny --num-classes 10 --max-images 50
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import cv2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, LogitsOnly)
from src.data_loader import get_transforms  # noqa: E402
from src.evaluation.explanation_metrics import VisualExplanationEvaluator  # noqa: E402
from experiments.run_explanation_comparison import (  # noqa: E402
    heatmap_gradcampp, heatmap_layercam,
    heatmap_ig, heatmap_lime, heatmap_shap)
from src.explainers.gradcam_plus import GradCAMPlusPlus  # noqa: E402

NAME = {'gradcam': 'Grad-CAM', 'gradcampp': 'Grad-CAM++', 'layercam': 'LayerCAM',
        'ig': 'Integrated Gradients', 'lime': 'LIME', 'shap': 'SHAP'}


class _VanillaGradCAM(GradCAMPlusPlus):
    """Standard Grad-CAM: weights = global-average-pooled gradients (no second
    order). Reuses the working Grad-CAM++ hook machinery."""
    def _compute_weights(self, gradients, activations, eps=1e-8):
        return gradients.mean(dim=(2, 3), keepdim=True)


def heatmap_gradcam(raw_model, x, class_idx, device):
    out = _VanillaGradCAM(raw_model).generate(x, class_idx)
    return out['cam'], out['class_idx']


def gen_cam(m, wrapped, raw, img, device, background):
    if m == 'gradcam':
        return heatmap_gradcam(raw, img, None, device)
    if m == 'gradcampp':
        return heatmap_gradcampp(raw, img, None, device)
    if m == 'layercam':
        return heatmap_layercam(raw, img, None, device)
    if m == 'ig':
        return heatmap_ig(wrapped, img, None, device)
    if m == 'lime':
        return heatmap_lime(wrapped, img, None, device)
    if m == 'shap':
        return heatmap_shap(wrapped, img, None, device, background)
    raise ValueError(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--methods', nargs='+',
                    default=['gradcam', 'gradcampp', 'layercam', 'ig', 'lime', 'shap'])
    ap.add_argument('--max-images', type=int, default=50)
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)

    raw = build_model(args.backbone, args.num_classes, pretrained=False)
    raw = load_checkpoint(raw, args.checkpoint, device)
    wrapped = LogitsOnly(raw).to(device).eval()
    evaluator = VisualExplanationEvaluator(model=raw)  # raw returns (logits, extras)

    tfm = get_transforms('test', args.input_size)
    df = pd.read_csv(args.splits_csv)
    df = df[df['split'] == 'test'] if 'split' in df.columns else df
    df = df.sample(n=min(args.max_images, len(df)), random_state=args.seed).reset_index(drop=True)
    print(f"[#7] faithfulness on {len(df)} test images; methods={args.methods}")

    background = None
    if 'shap' in args.methods:
        bg = [tfm(Image.open(p).convert('RGB')) for p in df['image_path'].head(16)]
        background = torch.stack(bg).to(device)

    per = []
    for i, row in df.iterrows():
        p = row['image_path']
        img = tfm(Image.open(p).convert('RGB')).unsqueeze(0).to(device)
        for m in args.methods:
            t0 = time.time()
            try:
                cam, cidx = gen_cam(m, wrapped, raw, img, device, background)
            except Exception as e:
                if i == 0:
                    print(f"[{m}] CAM FAILED: {e}")
                continue
            dt = time.time() - t0
            cam = cv2.resize(np.asarray(cam, dtype=np.float32),
                             (args.input_size, args.input_size))
            try:
                f = evaluator.evaluate_faithfulness(img, cam, class_idx=int(cidx))
                ins, dele = f.get('insertion_auc'), f.get('deletion_auc')
            except Exception as e:
                if i == 0:
                    print(f"[{m}] FAITHFULNESS FAILED: {e}")
                ins = dele = np.nan
            per.append({'method': m, 'image': os.path.basename(p),
                        'time_s': dt, 'insertion': ins, 'deletion': dele})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(df)} images done")

    pi = pd.DataFrame(per)
    pi.to_csv(os.path.join(args.output_dir, 'explanation_per_image.csv'), index=False)

    rows = []
    for m in args.methods:
        s = pi[pi['method'] == m]
        if s.empty:
            continue
        rows.append({'method': m, 'name': NAME.get(m, m), 'n': len(s),
                     'insertion': round(float(s['insertion'].mean()), 3),
                     'deletion': round(float(s['deletion'].mean()), 3),
                     'runtime_s': round(float(s['time_s'].mean()), 2)})
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(args.output_dir, 'explanation_comparison.csv'), index=False)
    print("\n=== Explanation faithfulness (mean over test sample) ===")
    print(summ.to_string(index=False))

    with open(os.path.join(args.output_dir, 'table_explanation_comparison.tex'), 'w') as f:
        f.write("% auto-generated by run_explanation_faithfulness.py (reviewer #7)\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Method} & \\textbf{Insertion} & \\textbf{Deletion} & "
                "\\textbf{Runtime (s)} \\\\\n\\midrule\n")
        for _, r in summ.iterrows():
            nm = r['name'] + ' (ours)' if r['method'] == 'gradcampp' else r['name']
            f.write(f"{nm} & {r['insertion']:.2f} & {r['deletion']:.2f} & "
                    f"{r['runtime_s']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("wrote table_explanation_comparison.tex")


if __name__ == '__main__':
    main()
