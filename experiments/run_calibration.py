#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Uncertainty calibration and decomposition experiment.
Addresses reviewer point #8 (uncertainty estimation & calibration).

Produces, on the held-out TEST split:
  - Temperature scaling   : optimal T fit on the VALIDATION split (NLL / LBFGS)
  - Calibration metrics    : ECE, MCE, Brier score, NLL  (before & after scaling)
  - Reliability diagram     : figure saved to outputs/reliability_diagram.png
  - Uncertainty decomposition (MC Dropout): epistemic (mutual information) vs
    aleatoric (expected entropy), and their share of total predictive entropy.
  - Referral policy table   : accuracy vs coverage as we abstain on the most
    uncertain predictions -> shows how uncertainty drives expert referral.

Usage
-----
python experiments/run_calibration.py \
    --checkpoint runs/convnextv2_best.pth \
    --splits-csv experiments/outputs/splits.csv \
    --mc-samples 30
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, LogitsOnly, build_loader)


# --------------------------------------------------------------------------- #
# Calibration metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(confidences, predictions, labels, n_bins=15):
    """ECE and MCE with equal-width confidence bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    N = len(labels)
    bin_stats = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_stats.append((0.5 * (lo + hi), np.nan, np.nan, 0))
            continue
        acc = (predictions[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        gap = abs(acc - conf)
        ece += (mask.sum() / N) * gap
        mce = max(mce, gap)
        bin_stats.append((0.5 * (lo + hi), acc, conf, int(mask.sum())))
    return ece, mce, bin_stats


def brier_score(probs, labels, num_classes):
    onehot = np.eye(num_classes)[labels]
    return np.mean(np.sum((probs - onehot) ** 2, axis=1))


def nll(probs, labels):
    return -np.mean(np.log(probs[np.arange(len(labels)), labels] + 1e-12))


# --------------------------------------------------------------------------- #
# Temperature scaling (Guo et al., 2017)
# --------------------------------------------------------------------------- #
def fit_temperature(logits, labels, device):
    """Optimise a single scalar T to minimise NLL on validation logits."""
    logits = torch.tensor(logits, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    T = torch.nn.Parameter(torch.ones(1, device=device))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=200)
    ce = nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / T, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().cpu().item())


@torch.no_grad()
def collect_logits(model, loader, device):
    all_logits, all_labels = [], []
    for batch in loader:
        x = batch['image'].to(device)
        out = model(x)
        logits = out[0] if isinstance(out, tuple) else out
        all_logits.append(logits.cpu().numpy())
        all_labels.append(batch['label'].numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def mc_dropout_decompose(raw_model, loader, device, n_samples):
    """Batched MC-Dropout: returns mean probs + epistemic/aleatoric per sample."""
    # Enable dropout layers.
    for mod in raw_model.modules():
        if isinstance(mod, nn.Dropout):
            mod.train()

    mean_probs, epi, ale, labels = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['image'].to(device)
            probs_samples = []
            for _ in range(n_samples):
                out = raw_model(x)
                logits = out[0] if isinstance(out, tuple) else out
                probs_samples.append(F.softmax(logits, dim=1).cpu().numpy())
            probs_samples = np.stack(probs_samples)            # (S, B, C)
            mean_p = probs_samples.mean(0)                      # (B, C)
            total_entropy = -np.sum(mean_p * np.log(mean_p + 1e-10), axis=1)
            expected_entropy = -np.mean(
                np.sum(probs_samples * np.log(probs_samples + 1e-10), axis=2), axis=0)
            mutual_info = total_entropy - expected_entropy
            mean_probs.append(mean_p)
            epi.append(mutual_info)            # epistemic
            ale.append(expected_entropy)       # aleatoric
            labels.append(batch['label'].numpy())
    return (np.concatenate(mean_probs), np.concatenate(epi),
            np.concatenate(ale), np.concatenate(labels))


def plot_reliability(bin_stats_before, bin_stats_after, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, stats, title in [(axes[0], bin_stats_before, 'Before scaling'),
                             (axes[1], bin_stats_after, 'After temperature scaling')]:
        centers = [c for c, a, _, n in stats if n > 0]
        accs = [a for c, a, _, n in stats if n > 0]
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.bar(centers, accs, width=1.0 / len(stats), alpha=0.7,
               edgecolor='black', label='Accuracy')
        ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
        ax.set_title(title); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()


def referral_table(confidences, predictions, labels):
    """Accuracy at increasing referral (abstention) rates."""
    order = np.argsort(confidences)  # least confident first -> referred first
    rows = []
    for referral in [0.0, 0.05, 0.10, 0.20, 0.30]:
        n_referred = int(referral * len(labels))
        kept = order[n_referred:]
        acc = (predictions[kept] == labels[kept]).mean() if len(kept) else float('nan')
        rows.append({'referral_rate': referral, 'coverage': 1 - referral,
                     'accuracy_on_kept': acc})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--splits-csv', default='experiments/outputs/splits.csv')
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--mc-samples', type=int, default=30)
    ap.add_argument('--n-bins', type=int, default=15)
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)

    raw_model = build_model(args.backbone, args.num_classes, pretrained=False)
    raw_model = load_checkpoint(raw_model, args.checkpoint, device)
    model = LogitsOnly(raw_model).to(device).eval()

    val_loader = build_loader(args.splits_csv, 'val', args.batch_size)
    test_loader = build_loader(args.splits_csv, 'test', args.batch_size)

    # 1) Temperature scaling fit on validation, applied to test.
    val_logits, val_labels = collect_logits(model, val_loader, device)
    T = fit_temperature(val_logits, val_labels, device)
    print(f"[#8] fitted temperature T = {T:.4f}")

    test_logits, test_labels = collect_logits(model, test_loader, device)

    def metrics_from_logits(logits, temp=1.0):
        probs = F.softmax(torch.tensor(logits) / temp, dim=1).numpy()
        conf = probs.max(1); pred = probs.argmax(1)
        ece, mce, bins = expected_calibration_error(conf, pred, test_labels, args.n_bins)
        return {
            'accuracy': (pred == test_labels).mean(),
            'ece': ece, 'mce': mce,
            'brier': brier_score(probs, test_labels, args.num_classes),
            'nll': nll(probs, test_labels),
        }, bins, conf, pred

    before, bins_before, _, _ = metrics_from_logits(test_logits, 1.0)
    after, bins_after, conf_after, pred_after = metrics_from_logits(test_logits, T)

    cal = pd.DataFrame([{'setting': 'uncalibrated', **before},
                        {'setting': f'temperature(T={T:.3f})', **after}])
    cal.to_csv(os.path.join(args.output_dir, 'calibration_metrics.csv'), index=False)
    print("\n=== Calibration metrics (test) ===")
    print(cal.to_string(index=False))

    plot_reliability(bins_before, bins_after,
                     os.path.join(args.output_dir, 'reliability_diagram.png'))

    # 2) MC-Dropout uncertainty decomposition.
    mean_p, epi, ale, mc_labels = mc_dropout_decompose(
        raw_model, test_loader, device, args.mc_samples)
    epi_share = epi.sum() / (epi.sum() + ale.sum() + 1e-12)
    decomp = {
        'mc_samples': args.mc_samples,
        'mean_total_entropy': float((epi + ale).mean()),
        'mean_epistemic_MI': float(epi.mean()),
        'mean_aleatoric_H': float(ale.mean()),
        'epistemic_share': float(epi_share),
        'aleatoric_share': float(1 - epi_share),
    }
    pd.DataFrame([decomp]).to_csv(
        os.path.join(args.output_dir, 'uncertainty_decomposition.csv'), index=False)
    print("\n=== Uncertainty decomposition (MC Dropout) ===")
    for k, v in decomp.items():
        print(f"  {k}: {v}")

    # 3) Referral policy: abstain on least-confident (highest epistemic) cases.
    ref = referral_table(conf_after, pred_after, test_labels)
    ref.to_csv(os.path.join(args.output_dir, 'referral_policy.csv'), index=False)
    print("\n=== Referral policy (abstain on most uncertain) ===")
    print(ref.to_string(index=False))


if __name__ == '__main__':
    main()
