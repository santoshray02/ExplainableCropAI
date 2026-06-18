#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reproducible, leakage-controlled train/val/test split for Paddy Doctor.
Addresses reviewer point #6 (data splitting & leakage).

What this does
--------------
1. Scans the dataset directory (one sub-folder per class) and lists every image.
2. Detects near-duplicate images with a perceptual hash (pHash). Near-duplicate
   GROUPS are kept together and assigned to a single split, so the same plant /
   near-identical photo can never appear in two splits (prevents leakage).
3. Performs a STRATIFIED, IMAGE-LEVEL split (70/15/15 by default) with a fixed
   seed, preserving per-class proportions in every split.
4. Writes:
     - splits.csv          (image_path,label,split)  -> consumed by training & all experiments
     - split_counts.csv    (per-class counts per split)  -> paste into the paper
     - split_summary.txt   (human-readable report + dedup stats)

Note on split level: Paddy Doctor provides one image per labelled sample with no
patient/field/farm identifier, so the natural unit is IMAGE-LEVEL. We additionally
collapse perceptual-hash duplicate groups to remove the main residual leakage risk.
Document this honestly in the manuscript.

Usage
-----
python experiments/create_splits.py \
    --data-dir /data/paddy-doctor/train_images \
    --output-dir experiments/outputs \
    --train 0.70 --val 0.15 --test 0.15 --seed 42 \
    --phash-threshold 5
"""

import os
import sys
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_images(data_dir):
    """Return (paths, labels) by walking class sub-directories."""
    paths, labels = [], []
    classes = sorted(d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d)))
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fn in sorted(os.listdir(cls_dir)):
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(cls_dir, fn))
                labels.append(cls)
    return paths, labels


def compute_phash_groups(paths, threshold):
    """Group near-duplicate images by perceptual hash Hamming distance.

    Returns a list of group ids (one per path). Images whose pHash differ by
    <= threshold bits are placed in the same group. Falls back to one-group-per
    -image if the `imagehash` package is unavailable.
    """
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("[create_splits] imagehash/Pillow not installed -> skipping dedup "
              "(every image is its own group). `pip install imagehash Pillow`")
        return list(range(len(paths))), 0

    hashes = []
    for p in paths:
        try:
            hashes.append(imagehash.phash(Image.open(p).convert('RGB')))
        except Exception as e:
            print(f"[create_splits] could not hash {p}: {e}")
            hashes.append(None)

    group_of = [-1] * len(paths)
    reps = []  # (hash, group_id)
    next_group = 0
    n_dupe = 0
    for i, h in enumerate(hashes):
        if h is None:
            group_of[i] = next_group
            next_group += 1
            continue
        assigned = None
        for rep_hash, gid in reps:
            if (h - rep_hash) <= threshold:
                assigned = gid
                break
        if assigned is None:
            group_of[i] = next_group
            reps.append((h, next_group))
            next_group += 1
        else:
            group_of[i] = assigned
            n_dupe += 1
    return group_of, n_dupe


def stratified_group_split(labels, groups, ratios, seed):
    """Stratified split that keeps duplicate groups intact.

    For each class we shuffle its groups and slice by the target ratios. Because
    a group could in principle span classes (rare for pHash near-dupes), each
    group is assigned to the split of its first-seen class.
    """
    rng = np.random.RandomState(seed)
    train_r, val_r, _ = ratios

    # Map group -> its (majority) class and member indices.
    group_indices = defaultdict(list)
    for idx, g in enumerate(groups):
        group_indices[g].append(idx)
    group_class = {g: Counter(labels[i] for i in idxs).most_common(1)[0][0]
                   for g, idxs in group_indices.items()}

    by_class = defaultdict(list)
    for g, cls in group_class.items():
        by_class[cls].append(g)

    split_of_index = {}
    for cls, grps in by_class.items():
        grps = list(grps)
        rng.shuffle(grps)
        n = len(grps)
        n_train = int(round(train_r * n))
        n_val = int(round(val_r * n))
        for j, g in enumerate(grps):
            if j < n_train:
                s = 'train'
            elif j < n_train + n_val:
                s = 'val'
            else:
                s = 'test'
            for i in group_indices[g]:
                split_of_index[i] = s
    return [split_of_index[i] for i in range(len(labels))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data-dir', required=True,
                    help='Dataset root with one sub-folder per class')
    ap.add_argument('--output-dir', default='experiments/outputs')
    ap.add_argument('--train', type=float, default=0.70)
    ap.add_argument('--val', type=float, default=0.15)
    ap.add_argument('--test', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--phash-threshold', type=int, default=5,
                    help='Hamming distance <= this => near-duplicate (0 disables grouping)')
    args = ap.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "ratios must sum to 1"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[create_splits] scanning {args.data_dir} ...")
    paths, labels = list_images(args.data_dir)
    labels = np.array(labels)
    print(f"[create_splits] found {len(paths)} images across {len(set(labels))} classes")

    if args.phash_threshold > 0:
        print("[create_splits] computing perceptual-hash duplicate groups ...")
        groups, n_dupe = compute_phash_groups(paths, args.phash_threshold)
    else:
        groups, n_dupe = list(range(len(paths))), 0
    print(f"[create_splits] near-duplicate images folded into groups: {n_dupe}")

    splits = stratified_group_split(labels, groups,
                                    (args.train, args.val, args.test), args.seed)

    df = pd.DataFrame({'image_path': paths, 'label': labels, 'split': splits})
    out_csv = os.path.join(args.output_dir, 'splits.csv')
    df.to_csv(out_csv, index=False)

    # Per-class x per-split count table.
    counts = (df.groupby(['label', 'split']).size()
              .unstack(fill_value=0)
              .reindex(columns=['train', 'val', 'test'], fill_value=0))
    counts['total'] = counts.sum(axis=1)
    counts.loc['TOTAL'] = counts.sum(axis=0)
    counts_csv = os.path.join(args.output_dir, 'split_counts.csv')
    counts.to_csv(counts_csv)

    summary = os.path.join(args.output_dir, 'split_summary.txt')
    with open(summary, 'w') as f:
        f.write("Reproducible split summary (reviewer point #6)\n")
        f.write("=" * 50 + "\n")
        f.write(f"data_dir            : {args.data_dir}\n")
        f.write(f"ratios (tr/val/te)  : {args.train}/{args.val}/{args.test}\n")
        f.write(f"seed                : {args.seed}\n")
        f.write(f"split level         : image-level (pHash duplicate groups kept together)\n")
        f.write(f"phash threshold     : {args.phash_threshold}\n")
        f.write(f"total images        : {len(paths)}\n")
        f.write(f"near-dup folded     : {n_dupe}\n\n")
        f.write(counts.to_string())
        f.write("\n")
    print(f"[create_splits] wrote:\n  {out_csv}\n  {counts_csv}\n  {summary}")
    print("\n" + counts.to_string())


if __name__ == '__main__':
    main()
