#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uncertainty-versus-error analysis with MC Dropout (Round-2 points R2.10, R2.11, R1.6).

Runs 30-sample MC Dropout on the test split and reports how well each
uncertainty signal identifies the incorrect predictions:
  * AUROC of (1 - MC mean confidence), predictive entropy and mutual
    information as scores for "prediction is wrong";
  * referral curves (accuracy on retained, errors captured) when referring by
    each signal;
  * bootstrap 95% CIs for ECE, Brier score and accuracy-on-retained at 10% and
    30% referral.

Outputs: uncertainty_error.csv, referral_by_signal.csv, calibration_ci.csv,
         mc_predictions.csv (per image: mean probs, entropy, MI).

Usage (deepbeastx, SUPIR env):
  python experiments/run_uncertainty_error.py --checkpoint experiments/outputs/baseline_convnextv2_tiny.pth
"""
import os
import sys
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
    expected_calibration_error, brier_score, fit_temperature, collect_logits,
    mc_dropout_decompose)
from experiments.run_baseline_eval import auroc, bootstrap  # noqa: E402


def referral_rows(score, correct, label):
    """Refer the highest-`score` fraction; report accuracy on kept and errors captured."""
    order = np.argsort(-score)  # most uncertain first
    n = len(score); n_err = int((~correct).sum()); rows = []
    for r in [0.0, 0.05, 0.10, 0.20, 0.30]:
        k = int(r * n); ref, kept = order[:k], order[k:]
        rows.append({'signal': label, 'referral_rate': r,
                     'accuracy_on_kept': round(float(correct[kept].mean()) * 100, 2),
                     'errors_captured': int((~correct[ref]).sum()), 'total_errors': n_err})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--mc-samples', type=int, default=30)
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    raw = load_checkpoint(build_model(args.backbone, args.num_classes, pretrained=False),
                          args.checkpoint, device)
    model = LogitsOnly(raw).to(device).eval()
    val_loader = build_loader(args.splits_csv, 'val', args.batch_size)
    test_loader = build_loader(args.splits_csv, 'test', args.batch_size)
    k = args.num_classes

    # Deterministic (dropout off) temperature-scaled predictions
    vl, vy = collect_logits(model, val_loader, device)
    T = fit_temperature(vl, vy, device)
    tl, ty = collect_logits(model, test_loader, device)
    probs_T = F.softmax(torch.tensor(tl) / T, 1).numpy()
    pred_T = probs_T.argmax(1); conf_T = probs_T.max(1); correct_T = pred_T == ty
    n = len(ty)

    # MC Dropout
    set_seed(args.seed)
    mean_p, mi, ale, my = mc_dropout_decompose(raw, test_loader, device, args.mc_samples)
    assert np.array_equal(my, ty)
    entropy = mi + ale
    pred_mc = mean_p.argmax(1); conf_mc = mean_p.max(1); correct_mc = pred_mc == ty

    signals = {
        'softmax_conf_T (1-conf)': (1 - conf_T, correct_T),
        'mc_mean_conf (1-conf)': (1 - conf_mc, correct_mc),
        'mc_predictive_entropy': (entropy, correct_mc),
        'mc_mutual_information': (mi, correct_mc),
        'mc_aleatoric_entropy': (ale, correct_mc),
    }
    rows, ref_rows = [], []
    for name, (score, corr) in signals.items():
        a = auroc(score, ~corr)
        a_ci = bootstrap(lambda idx: auroc(score[idx], ~corr[idx]), n, B=500)
        rows.append({'signal': name, 'accuracy': round(float(corr.mean()) * 100, 2),
                     'n_errors': int((~corr).sum()), 'auroc_error': round(a, 3),
                     'auroc_ci_lo': round(a_ci[0], 3), 'auroc_ci_hi': round(a_ci[1], 3),
                     'mean_score_correct': round(float(score[corr].mean()), 4),
                     'mean_score_wrong': round(float(score[~corr].mean()), 4)})
        ref_rows.extend(referral_rows(score, corr, name))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, 'uncertainty_error.csv'), index=False)
    pd.DataFrame(ref_rows).to_csv(os.path.join(args.output_dir, 'referral_by_signal.csv'), index=False)

    # Bootstrap CIs for calibration numbers (temperature-scaled deterministic predictions)
    def ece_fn(idx):
        return expected_calibration_error(conf_T[idx], pred_T[idx], ty[idx])[0]

    def brier_fn(idx):
        return brier_score(probs_T[idx], ty[idx], k)

    order = np.argsort(conf_T)

    def kept_acc(idx, r):
        o = idx[np.argsort(conf_T[idx])]
        return correct_T[o[int(r * len(o)):]].mean()

    cal = {'temperature': round(T, 3),
           'ece': round(ece_fn(np.arange(n)), 4), 'brier': round(brier_fn(np.arange(n)), 4),
           'acc_kept_10pct_referral': round(kept_acc(np.arange(n), 0.10) * 100, 2),
           'acc_kept_30pct_referral': round(kept_acc(np.arange(n), 0.30) * 100, 2)}
    for name, fn in [('ece', ece_fn), ('brier', brier_fn),
                     ('acc_kept_10pct_referral', lambda i: kept_acc(i, 0.10) * 100),
                     ('acc_kept_30pct_referral', lambda i: kept_acc(i, 0.30) * 100)]:
        lo, hi = bootstrap(fn, n, B=1000)
        cal[f'{name}_ci_lo'] = round(lo, 4); cal[f'{name}_ci_hi'] = round(hi, 4)
    pd.DataFrame([cal]).to_csv(os.path.join(args.output_dir, 'calibration_ci.csv'), index=False)
    print(cal)

    paths = []
    for batch in test_loader:
        paths.extend(batch['path'])
    mc = pd.DataFrame({'image_path': paths, 'true_label': [PADDY_DISEASE_CLASSES[i] for i in ty],
                       'pred_mc': [PADDY_DISEASE_CLASSES[i] for i in pred_mc], 'correct_mc': correct_mc,
                       'mc_conf': conf_mc, 'entropy': entropy, 'mutual_info': mi, 'aleatoric': ale,
                       'softmax_conf_T': conf_T, 'correct_T': correct_T})
    mc.to_csv(os.path.join(args.output_dir, 'mc_predictions.csv'), index=False)
    print(pd.DataFrame(ref_rows).to_string(index=False))


if __name__ == '__main__':
    main()
