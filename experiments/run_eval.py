#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full test-set evaluation for a trained classifier.

Produces the headline §3.1 numbers and the confusion-matrix data that the
baselines script does not: overall accuracy / macro precision-recall-F1,
per-class precision/recall/F1, top-2 / top-3 accuracy, the confusion matrix,
and a per-image predictions CSV (for regenerating the confusion-matrix figure
and for the explanation/calibration scripts).

Usage
-----
python experiments/run_eval.py \
    --checkpoint experiments/outputs/baseline_convnextv2_tiny.pth \
    --splits-csv experiments/outputs/splits.csv \
    --backbone convnextv2_tiny --num-classes 10
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, build_loader, load_checkpoint,
    PADDY_DISEASE_CLASSES)
from sklearn.metrics import (  # noqa: E402
    accuracy_score, precision_recall_fscore_support, confusion_matrix)


def topk_accuracy(probs, labels, k):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--split', default='test')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)

    model = build_model(args.backbone, args.num_classes, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device)
    loader = build_loader(args.splits_csv, args.split, args.batch_size,
                          input_size=args.input_size, shuffle=False)

    classes = PADDY_DISEASE_CLASSES
    all_probs, all_labels, all_paths = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            all_probs.append(F.softmax(logits, 1).cpu().numpy())
            all_labels.append(batch['label'].numpy())
            all_paths.extend(batch['path'])
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(1)

    acc = accuracy_score(labels, preds)
    mp, mr, mf, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0)
    pc_p, pc_r, pc_f, pc_s = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(classes))), zero_division=0)
    top2 = topk_accuracy(probs, labels, 2)
    top3 = topk_accuracy(probs, labels, 3)
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    cm_pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100.0

    # Per-image predictions (for the confusion-matrix figure + downstream scripts)
    pred_df = pd.DataFrame({
        'image_path': all_paths,
        'true_label': [classes[i] for i in labels],
        'pred_label': [classes[i] for i in preds],
        'correct': (labels == preds),
    })
    for ci, c in enumerate(classes):
        pred_df[f'prob_{c}'] = probs[:, ci]
    pred_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)

    # Per-class metrics
    pcm = pd.DataFrame({
        'class': classes,
        'precision': np.round(pc_p * 100, 1),
        'recall': np.round(pc_r * 100, 1),
        'f1': np.round(pc_f * 100, 1),
        'support': pc_s,
    })
    pcm.to_csv(os.path.join(args.output_dir, 'per_class_metrics.csv'), index=False)

    # Confusion matrix (counts + row-normalised %)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(args.output_dir, 'confusion_matrix.csv'))
    pd.DataFrame(np.round(cm_pct, 1), index=classes, columns=classes).to_csv(
        os.path.join(args.output_dir, 'confusion_matrix_pct.csv'))

    metrics = {
        'accuracy': round(acc * 100, 1),
        'macro_precision': round(mp * 100, 1),
        'macro_recall': round(mr * 100, 1),
        'macro_f1': round(mf * 100, 1),
        'top2_accuracy': round(top2 * 100, 1),
        'top3_accuracy': round(top3 * 100, 1),
        'n_test': int(len(labels)),
    }
    with open(os.path.join(args.output_dir, 'headline_metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)
    print(json.dumps(metrics, indent=2))
    print(pcm.to_string(index=False))


if __name__ == '__main__':
    main()
