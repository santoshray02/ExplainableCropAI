#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Few-shot domain adaptation on an external rice-disease dataset (R2.6 follow-up).

Question answered: how many *local* images are needed to recover the accuracy
lost under cross-dataset shift (see run_external_eval.py)?

Protocol
--------
* Fixed, stratified hold-out: 50% of every shared class is the external TEST set
  (seed 42) and is never touched by training.
* From the other 50%, draw k images per class (k in --shots) as the adaptation
  set; k=0 is the zero-shot baseline, k='all' uses the whole 50%.
* Fine-tune the Paddy Doctor checkpoint end-to-end (AdamW, lr 1e-4, cosine,
  --epochs epochs, train-mode augmentation) and report closed-set accuracy and
  macro-F1 on the external TEST set, with bootstrap 95% CIs.
* Optional --compare-imagenet repeats the k='all' run from ImageNet weights
  (no Paddy pre-training) to show what the Paddy Doctor training contributes.

Usage (deepbeastx, SUPIR env):
  python experiments/run_external_finetune.py \
      --checkpoint experiments/outputs/baseline_convnextv2_tiny.pth \
      --data-dir "/home/santosh/data/external/riceleafbd/Original Images" \
      --class-map '{"Bacterial Leaf Blight":"bacterial_leaf_blight","Brown Spot":"brown_spot","Healthy Leaf":"normal","Tungro Virus":"tungro"}' \
      --name riceleafbd --shots 0 5 10 25 50 all --epochs 15
"""
import os
import sys
import json
import copy
import math
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments._common import (  # noqa: E402
    set_seed, get_device, build_model, load_checkpoint, PADDY_DISEASE_CLASSES)
from experiments.run_external_eval import list_images, bootstrap_ci, macro_f1  # noqa: E402
from src.data_loader import get_transforms  # noqa: E402


class ListDataset(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert('RGB')), y


def stratified_holdout(items, classes, frac=0.5, seed=42, group_fn=None):
    """Hold out `frac` of each class as the external test set.

    If `group_fn(path) -> group id` is given, whole groups (e.g. capture
    sessions / augmentation families identified from the filename) are kept on
    one side of the split, so near-duplicates cannot leak between adaptation and
    test sets (reviewer point R2.5 applied to the external data too).
    """
    rng = random.Random(seed)
    pool, test = [], []
    for c in classes:
        xs = [it for it in items if it[1] == c]
        if group_fn is None:
            rng.shuffle(xs)
            n_test = int(round(len(xs) * frac))
            test += xs[:n_test]
            pool += xs[n_test:]
        else:
            groups = sorted({group_fn(p) for p, _ in xs})
            rng.shuffle(groups)
            n_test_g = max(1, int(round(len(groups) * frac)))
            test_g = set(groups[:n_test_g])
            test += [it for it in xs if group_fn(it[0]) in test_g]
            pool += [it for it in xs if group_fn(it[0]) not in test_g]
            print(f'    class {c}: {len(groups)} groups -> {n_test_g} held out as test '
                  f'({len([1 for it in xs if group_fn(it[0]) in test_g])} images)')
    return pool, test


def sample_shots(pool, classes, k, seed):
    if k == 'all':
        return list(pool)
    rng = random.Random(seed)
    out = []
    for c in classes:
        xs = [it for it in pool if it[1] == c]
        rng.shuffle(xs)
        out += xs[:int(k)]
    return out


@torch.no_grad()
def predict(model, loader, device, shared_idx):
    model.eval()
    probs = []
    for x, _ in loader:
        out = model(x.to(device))
        logits = out[0] if isinstance(out, tuple) else out
        probs.append(F.softmax(logits, 1).cpu())
    probs = torch.cat(probs).numpy()
    return np.array(shared_idx)[probs[:, shared_idx].argmax(1)]


def finetune(model, train_items, device, epochs, lr, batch_size, input_size, seed):
    set_seed(seed)
    ds = ListDataset(train_items, get_transforms('train', input_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = max(1, epochs * len(loader))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    model.train()
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        print(f'    epoch {ep + 1}/{epochs} loss={tot / max(n, 1):.4f}', flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--backbone', default='convnextv2_tiny')
    ap.add_argument('--num-classes', type=int, default=10)
    ap.add_argument('--data-dir', required=True)
    ap.add_argument('--class-map', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--shots', nargs='+', default=['0', '5', '10', '25', '50', 'all'])
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--input-size', type=int, default=224)
    ap.add_argument('--holdout-frac', type=float, default=0.5)
    ap.add_argument('--group-regex', default=None,
                    help='regex whose first group, matched on the file name, identifies images '
                         'that must stay on one side of the hold-out (e.g. "^([A-Za-z]+[0-9]+)_" '
                         'for the Sethy et al. files BLAST1_001.jpg ...)')
    ap.add_argument('--compare-imagenet', action='store_true',
                    help='also fine-tune from ImageNet weights (no Paddy Doctor training) as a control')
    ap.add_argument('--imagenet-shots', nargs='+', default=['all'],
                    help='k values for the ImageNet-init control')
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device(prefer_gpu=not args.cpu)
    os.makedirs(args.output_dir, exist_ok=True)
    classes = PADDY_DISEASE_CLASSES
    class_map = json.loads(args.class_map)
    shared = sorted(set(class_map.values()), key=classes.index)
    shared_idx = [classes.index(c) for c in shared]

    items = [(p, classes.index(c)) for p, c in list_images(args.data_dir, class_map)]
    group_fn = None
    if args.group_regex:
        import re
        rx = re.compile(args.group_regex)

        def group_fn(path):
            m = rx.search(os.path.basename(path))
            return m.group(1) if m else os.path.basename(path)
    pool, test = stratified_holdout(items, shared_idx, args.holdout_frac, args.seed, group_fn)
    print(f'{args.name}: {len(items)} images; adaptation pool {len(pool)}, held-out test {len(test)}')
    y_test = np.array([y for _, y in test])
    test_loader = DataLoader(ListDataset(test, get_transforms('test', args.input_size)),
                             batch_size=64, shuffle=False, num_workers=4)

    base = build_model(args.backbone, args.num_classes, pretrained=False)
    base = load_checkpoint(base, args.checkpoint, device)
    base_state = copy.deepcopy(base.state_dict())

    rows = []

    def evaluate(model, tag, n_train, init):
        pred = predict(model, test_loader, device, shared_idx)
        n = len(y_test)
        acc = float(np.mean(pred == y_test))
        mf = macro_f1(y_test, pred, shared_idx)
        acc_ci = bootstrap_ci(lambda idx: np.mean(pred[idx] == y_test[idx]), n)
        mf_ci = bootstrap_ci(lambda idx: macro_f1(y_test[idx], pred[idx], shared_idx), n)
        row = {'dataset': args.name, 'init': init, 'shots_per_class': tag, 'n_train': n_train,
               'n_test': n, 'accuracy': round(acc * 100, 1),
               'acc_ci_lo': round(acc_ci[0] * 100, 1), 'acc_ci_hi': round(acc_ci[1] * 100, 1),
               'macro_f1': round(mf * 100, 1),
               'f1_ci_lo': round(mf_ci[0] * 100, 1), 'f1_ci_hi': round(mf_ci[1] * 100, 1)}
        for c in shared_idx:
            row[f'recall_{classes[c]}'] = round(float(np.mean(pred[y_test == c] == c)) * 100, 1)
        rows.append(row)
        print(json.dumps(row), flush=True)

    for k in args.shots:
        model = build_model(args.backbone, args.num_classes, pretrained=False).to(device)
        model.load_state_dict(base_state)
        if k == '0':
            evaluate(model, 0, 0, 'paddy')
            continue
        tr = sample_shots(pool, shared_idx, k, args.seed)
        print(f'  fine-tuning with k={k} per class ({len(tr)} images)', flush=True)
        model = finetune(model, tr, device, args.epochs, args.lr, args.batch_size,
                         args.input_size, args.seed)
        evaluate(model, k, len(tr), 'paddy')
        del model; torch.cuda.empty_cache()

    if args.compare_imagenet:
        for k in args.imagenet_shots:
            model = build_model(args.backbone, args.num_classes, pretrained=True).to(device)
            tr = sample_shots(pool, shared_idx, k, args.seed)
            print(f'  ImageNet-init control, k={k} ({len(tr)} images)', flush=True)
            model = finetune(model, tr, device, args.epochs, args.lr, args.batch_size,
                             args.input_size, args.seed)
            evaluate(model, k, len(tr), 'imagenet')
            del model; torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = os.path.join(args.output_dir, f'external_{args.name}_finetune.csv')
    df.to_csv(out, index=False)
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
