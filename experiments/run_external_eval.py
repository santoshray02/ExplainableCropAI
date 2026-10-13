#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Zero-shot external-dataset evaluation (Round-2 reviewer point R2.6).

Runs the trained Paddy Doctor classifier, WITHOUT fine-tuning, on an
independently collected rice-disease dataset arranged as one sub-folder per
class, and reports performance on the classes the two label sets share.

Two protocols are reported:
  * closed-set  : argmax restricted to the shared classes (the fair comparison
                  for a classifier that was trained on a superset of labels);
  * open-set    : argmax over all 10 Paddy Doctor classes; an image predicted
                  as a class absent from the external dataset counts as wrong.
Also reported: per-class recall, macro-F1, 2,000-sample bootstrap 95% CIs,
confusion matrix, and the fraction of images the referral policy would send
to an expert at several confidence thresholds.

Usage (deepbeastx, SUPIR env):
  python experiments/run_external_eval.py \
      --checkpoint experiments/outputs/baseline_convnextv2_tiny.pth \
      --data-dir "/home/santosh/data/external/riceleafbd/Original Images" \
      --class-map '{"Bacterial Leaf Blight":"bacterial_leaf_blight","Brown Spot":"brown_spot","Healthy Leaf":"normal","Tungro Virus":"tungro"}' \
      --name riceleafbd
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, PADDY_DISEASE_CLASSES)
from src.data_loader import get_transforms  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def list_images(data_dir, class_map):
    items = []
    for folder, paddy_cls in class_map.items():
        d = os.path.join(data_dir, folder)
        if not os.path.isdir(d):
            raise FileNotFoundError(d)
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXT):
                items.append((os.path.join(d, fn), paddy_cls))
    return items


def bootstrap_ci(fn, n, B=2000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(B):
        idx = rng.randint(0, n, n)
        vals.append(fn(idx))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def macro_f1(y, p, classes):
    f1s = []
    for c in classes:
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--class-map', required=True,
                    help='JSON dict: external folder name -> Paddy Doctor class name')
    ap.add_argument('--name', required=True, help='short dataset tag for output files')
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--temperature', type=float, default=1.423,
                    help='temperature fitted on the Paddy Doctor validation split')
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    class_map = json.loads(args.class_map)
    classes = PADDY_DISEASE_CLASSES
    shared = sorted(set(class_map.values()), key=classes.index)
    shared_idx = [classes.index(c) for c in shared]

    model = build_model(args.backbone, args.num_classes, pretrained=False)
    model = load_checkpoint(model, args.checkpoint, device)
    tf = get_transforms('test', args.input_size)

    items = list_images(args.data_dir, class_map)
    print(f'{len(items)} images in {len(class_map)} folders -> shared classes {shared}')

    probs_all, logits_all = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(items), args.batch_size):
            batch = items[i:i + args.batch_size]
            xs = torch.stack([tf(Image.open(p).convert('RGB')) for p, _ in batch]).to(device)
            out = model(xs)
            logits = out[0] if isinstance(out, tuple) else out
            logits_all.append(logits.cpu())
    logits = torch.cat(logits_all)
    probs = F.softmax(logits, 1).numpy()
    probs_T = F.softmax(logits / args.temperature, 1).numpy()
    y = np.array([classes.index(c) for _, c in items])

    pred_open = probs.argmax(1)
    sub = probs[:, shared_idx]
    pred_closed = np.array(shared_idx)[sub.argmax(1)]
    conf_closed = probs_T[:, shared_idx].max(1) / np.maximum(probs_T[:, shared_idx].sum(1), 1e-9)
    conf_open = probs_T.max(1)

    n = len(y)
    res = {'dataset': args.name, 'n': int(n), 'shared_classes': shared,
           'checkpoint': os.path.basename(args.checkpoint)}
    for tag, pred in (('closed', pred_closed), ('open', pred_open)):
        acc = float(np.mean(pred == y))
        mf = macro_f1(y, pred, shared_idx)
        acc_ci = bootstrap_ci(lambda idx: np.mean(pred[idx] == y[idx]), n)
        mf_ci = bootstrap_ci(lambda idx: macro_f1(y[idx], pred[idx], shared_idx), n)
        res[f'{tag}_accuracy'] = round(acc * 100, 1)
        res[f'{tag}_accuracy_ci95'] = [round(v * 100, 1) for v in acc_ci]
        res[f'{tag}_macro_f1'] = round(mf * 100, 1)
        res[f'{tag}_macro_f1_ci95'] = [round(v * 100, 1) for v in mf_ci]
        res[f'{tag}_per_class_recall'] = {
            classes[c]: round(float(np.mean(pred[y == c] == c)) * 100, 1) for c in shared_idx}
    # where do open-set predictions go?
    res['open_pred_distribution'] = {
        classes[c]: int(np.sum(pred_open == c)) for c in range(len(classes)) if np.sum(pred_open == c)}
    # referral policy on external images (closed-set confidence)
    for thr in (0.5, 0.7, 0.9):
        refer = conf_closed < thr
        kept = ~refer
        res[f'referral_rate_conf<{thr}'] = round(float(refer.mean()) * 100, 1)
        res[f'accuracy_on_kept_conf>={thr}'] = (
            round(float(np.mean(pred_closed[kept] == y[kept])) * 100, 1) if kept.any() else None)
    errs = pred_closed != y
    if errs.any() and (~errs).any():
        pos, neg = conf_closed[~errs], conf_closed[errs]
        auroc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos[:2000] for b in neg[:2000]]))
        res['auroc_conf_vs_correct_closed'] = round(auroc, 3)
    res['mean_conf_closed'] = round(float(conf_closed.mean()), 3)

    cm = pd.crosstab(pd.Series([classes[i] for i in y], name='true'),
                     pd.Series([classes[i] for i in pred_closed], name='pred_closed'))
    cm.to_csv(os.path.join(args.output_dir, f'external_{args.name}_confusion_closed.csv'))
    cm_open = pd.crosstab(pd.Series([classes[i] for i in y], name='true'),
                          pd.Series([classes[i] for i in pred_open], name='pred_open'))
    cm_open.to_csv(os.path.join(args.output_dir, f'external_{args.name}_confusion_open.csv'))

    df = pd.DataFrame({'image_path': [p for p, _ in items],
                       'true_label': [classes[i] for i in y],
                       'pred_closed': [classes[i] for i in pred_closed],
                       'pred_open': [classes[i] for i in pred_open],
                       'conf_closed': conf_closed, 'conf_open': conf_open})
    for ci, c in enumerate(classes):
        df[f'prob_{c}'] = probs[:, ci]
    df.to_csv(os.path.join(args.output_dir, f'external_{args.name}_predictions.csv'), index=False)

    with open(os.path.join(args.output_dir, f'external_{args.name}_metrics.json'), 'w') as fp:
        json.dump(res, fp, indent=2)
    print(json.dumps(res, indent=2))
    print(cm.to_string())


if __name__ == '__main__':
    main()
