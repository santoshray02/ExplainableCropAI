#!/bin/bash
cd ~/projects/ExplainableCropAI
P=~/miniconda3/envs/SUPIR/bin/python
CK=experiments/outputs/baseline_convnextv2_tiny.pth
$P experiments/run_external_finetune.py --checkpoint $CK \
  --data-dir "/home/santosh/data/external/riceleafbd/Original Images" \
  --class-map '{"Bacterial Leaf Blight":"bacterial_leaf_blight","Brown Spot":"brown_spot","Healthy Leaf":"normal","Tungro Virus":"tungro"}' \
  --name riceleafbd --shots 0 5 10 25 50 all --epochs 15 --compare-imagenet
$P experiments/run_external_finetune.py --checkpoint $CK \
  --data-dir "/home/santosh/data/external/sethy/Rice Leaf Disease Images" \
  --class-map '{"Bacterialblight":"bacterial_leaf_blight","Blast":"blast","Brownspot":"brown_spot","Tungro":"tungro"}' \
  --name sethy --shots 0 5 10 25 50 all --epochs 15 --compare-imagenet
echo ALLDONE
