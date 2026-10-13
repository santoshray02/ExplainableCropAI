#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extended baseline comparison (Round-2 points R1.4, R2.19).

For every saved baseline checkpoint, evaluate on the held-out test split and
report, beyond accuracy: macro-F1, bootstrap 95% CIs, temperature-scaled ECE
and Brier score, AUROC of confidence as a score for correctness, and an exact
McNemar test of each baseline against ConvNeXt V2-Tiny on the paired test
predictions (plus a bootstrap CI on the accuracy difference).

Outputs (in --output-dir):
  predictions_<backbone>.csv   per-image predictions for each model
  baselines_extended.csv       one row per model
  mcnemar.csv                  paired comparison of every baseline vs. ConvNeXt V2-Tiny

Usage (deepbeastx, SUPIR env):
  python experiments/run_baseline_eval.py --splits-csv experiments/outputs/splits.csv
"""
import os
import sys
import math
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, LogitsOnly, build_loader,
    PADDY_DISEASE_CLASSES)
from experiments.run_calibration import (  # noqa: E402
    expected_calibration_error, brier_score, fit_temperature, collect_logits)

BACKBONES = ['convnextv2_tiny', 'resnet50', 'vgg16', 'efficientformer_l1',
             'mobilevit_s', 'efficientnet_b0', 'inception_v3']


def macro_f1(y, p, k):
    f = []
    for c in range(k):
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c))
        pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def auroc(score, positive):
    """AUROC of `score` for predicting the boolean `positive` (rank based)."""
    order = np.argsort(score)
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = positive.sum(); n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap(fn, n, B=2000, seed=42):
    rng = np.random.RandomState(seed)
    vals = [fn(rng.randint(0, n, n)) for _ in range(B)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--checkpoint-dir', default='experiments/outputs')
    ap.add_argument('--backbones', nargs='+', default=BACKBONES)
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    val_loader = build_loader(args.splits_csv, 'val', args.batch_size)
    test_loader = build_loader(args.splits_csv, 'test', args.batch_size)
    k = args.num_classes

    rows, correct_by_model, paths = [], {}, None
    for bb in args.backbones:
        ckpt = os.path.join(args.checkpoint_dir, f'baseline_{bb}.pth')
        if not os.path.exists(ckpt):
            print(f'[skip] {ckpt} not found'); continue
        print(f'=== {bb} ===', flush=True)
        raw = build_model(bb, k, pretrained=False)
        raw = load_checkpoint(raw, ckpt, device)
        model = LogitsOnly(raw).to(device).eval()
        vl, vy = collect_logits(model, val_loader, device)
        T = fit_temperature(vl, vy, device)
        tl, ty = collect_logits(model, test_loader, device)
        if paths is None:
            paths = []
            for batch in test_loader:
                paths.extend(batch['path'])
        probs = F.softmax(torch.tensor(tl) / T, dim=1).numpy()
        probs_raw = F.softmax(torch.tensor(tl), dim=1).numpy()
        pred = probs.argmax(1); conf = probs.max(1)
        correct = pred == ty
        correct_by_model[bb] = correct
        n = len(ty)
        acc = correct.mean()
        acc_ci = bootstrap(lambda idx: correct[idx].mean(), n)
        mf1 = macro_f1(ty, pred, k)
        mf1_ci = bootstrap(lambda idx: macro_f1(ty[idx], pred[idx], k), n)
        ece_raw, _, _ = expected_calibration_error(probs_raw.max(1), probs_raw.argmax(1), ty)
        ece_T, mce_T, _ = expected_calibration_error(conf, pred, ty)
        row = {'backbone': bb, 'n_test': n, 'temperature': round(T, 3),
               'accuracy': round(acc * 100, 2), 'acc_ci_lo': round(acc_ci[0] * 100, 2),
               'acc_ci_hi': round(acc_ci[1] * 100, 2),
               'macro_f1': round(mf1 * 100, 2), 'f1_ci_lo': round(mf1_ci[0] * 100, 2),
               'f1_ci_hi': round(mf1_ci[1] * 100, 2),
               'ece_uncalibrated': round(ece_raw, 4), 'ece_temp_scaled': round(ece_T, 4),
               'mce_temp_scaled': round(mce_T, 4),
               'brier_temp_scaled': round(brier_score(probs, ty, k), 4),
               'auroc_conf_correct': round(auroc(conf, correct), 3),
               'n_errors': int((~correct).sum())}
        rows.append(row); print(row, flush=True)
        df = pd.DataFrame({'image_path': paths, 'true_label': [PADDY_DISEASE_CLASSES[i] for i in ty],
                           'pred_label': [PADDY_DISEASE_CLASSES[i] for i in pred],
                           'correct': correct, 'confidence_T': conf})
        for ci, c in enumerate(PADDY_DISEASE_CLASSES):
            df[f'prob_{c}'] = probs[:, ci]
        df.to_csv(os.path.join(args.output_dir, f'predictions_{bb}.csv'), index=False)
        del raw, model; torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'baselines_extended.csv'), index=False)

    # McNemar vs ConvNeXt V2-Tiny
    ref = 'convnextv2_tiny'
    if ref in correct_by_model:
        a = correct_by_model[ref]
        mrows = []
        for bb, c in correct_by_model.items():
            if bb == ref:
                continue
            b_only = int(np.sum(a & ~c))   # ConvNeXt right, baseline wrong
            c_only = int(np.sum(~a & c))   # baseline right, ConvNeXt wrong
            diff = a.mean() - c.mean()
            d_ci = bootstrap(lambda idx: a[idx].mean() - c[idx].mean(), len(a))
            mrows.append({'baseline': bb, 'convnext_acc': round(a.mean() * 100, 2),
                          'baseline_acc': round(c.mean() * 100, 2),
                          'acc_diff_points': round(diff * 100, 2),
                          'diff_ci_lo': round(d_ci[0] * 100, 2), 'diff_ci_hi': round(d_ci[1] * 100, 2),
                          'convnext_only_correct': b_only, 'baseline_only_correct': c_only,
                          'mcnemar_p_exact': round(mcnemar_exact(b_only, c_only), 4)})
        mdf = pd.DataFrame(mrows)
        mdf.to_csv(os.path.join(args.output_dir, 'mcnemar.csv'), index=False)
        print(mdf.to_string(index=False))


if __name__ == '__main__':
    main()
