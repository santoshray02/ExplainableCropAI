"""Regenerate the confusion-matrix figure (Fig 4) from REAL test predictions."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
cm_csv = os.path.join(HERE, 'outputs', 'confusion_matrix_pct.csv')
out_dir = os.path.join(ROOT, 'paper', 'figures')

LABELS = {
    'bacterial_leaf_blight': 'Bacterial Leaf Blight',
    'bacterial_leaf_streak': 'Bacterial Leaf Streak',
    'bacterial_panicle_blight': 'Bacterial Panicle Blight',
    'blast': 'Leaf Blast',
    'brown_spot': 'Brown Spot',
    'dead_heart': 'Dead Heart',
    'downy_mildew': 'Downy Mildew',
    'hispa': 'Hispa',
    'normal': 'Healthy',
    'tungro': 'Tungro',
}

df = pd.read_csv(cm_csv, index_col=0)
classes = list(df.index)
disp = [LABELS[c] for c in classes]
cm = df.values

fig, ax = plt.subplots(figsize=(9, 7.6))
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=100)
ax.set_xticks(range(len(disp)))
ax.set_xticklabels(disp, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(disp)))
ax.set_yticklabels(disp, fontsize=9)
ax.set_xlabel('Predicted label', fontsize=11)
ax.set_ylabel('True label', fontsize=11)
for i in range(len(disp)):
    for j in range(len(disp)):
        v = cm[i, j]
        if v > 0:
            ax.text(j, i, f'{v:.1f}', ha='center', va='center',
                    fontsize=7.5, color='white' if v > 55 else '#222222')
cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Row-normalised %')
ax.set_title('Confusion matrix: ConvNeXt V2-Tiny (overall accuracy 97.5%)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'confusion_matrix.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
print('saved confusion_matrix.pdf/.png to', out_dir)
