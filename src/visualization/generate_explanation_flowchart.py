#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate the ExplainableCropAI multi-level explanation process flowchart.

Revision notes
--------------
This version is a process / architecture diagram, not a worked example.
Each box describes the *role* of the component; no specific disease,
treatment, or confidence values appear inside the figure (those belong
to the case-study figures elsewhere in the paper). Novel contributions
share the same red double-border convention as Figure 1.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Output path -----------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
output_dir = os.path.join(project_root, "paper", "figures")
os.makedirs(output_dir, exist_ok=True)

# Canvas ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Palette (mirrors the TikZ colours in main.tex for visual consistency) -------
COLORS = {
    "input":          "#E6E6E6",
    "backbone":       "#C8E6C8",
    "fpn":            "#B4DCB4",
    "gradcam":        "#C8DCFF",
    "uncertainty":    "#FFDCB4",
    "llm":            "#E6C8FF",
    "recommender":    "#FFFFC8",
    "kb":             "#FFF6BE",
    "output":         "#C8F0F0",
    "outchannel":     "#A0DCDC",
    "title_bar":      "#37474F",
    "arrow":          "#546E7A",
    "text":           "#212121",
    "novel_border":   "#C83232",
}

# Component drawing helpers ---------------------------------------------------
def draw_box(x, y, w, h, fill, title, description=None,
             novel=False, title_size=13, desc_size=11):
    """Draw a titled rounded box. Novel components get a red double border."""
    edge_color = COLORS["novel_border"] if novel else "#455A64"
    line_w = 2.2 if novel else 1.2

    body = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        facecolor=fill, edgecolor=edge_color,
        linewidth=line_w, zorder=2,
    )
    ax.add_patch(body)

    # Inner second line for the "double border" cue on novel boxes
    if novel:
        inner = FancyBboxPatch(
            (x + 0.35, y + 0.35), w - 0.7, h - 0.7,
            boxstyle="round,pad=0.02,rounding_size=0.5",
            facecolor="none", edgecolor=edge_color,
            linewidth=0.7, zorder=3,
        )
        ax.add_patch(inner)

    # Title bar
    title_h = 3.2
    title_bar = FancyBboxPatch(
        (x, y + h - title_h), w, title_h,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        facecolor=COLORS["title_bar"], edgecolor=COLORS["title_bar"],
        linewidth=1, zorder=4,
    )
    ax.add_patch(title_bar)
    ax.text(x + w / 2, y + h - title_h / 2, title,
            ha="center", va="center", fontsize=title_size,
            fontweight="bold", color="white", zorder=5)

    # Novel badge (top-right of the box)
    if novel:
        badge_w, badge_h = 2.6, 1.6
        bx = x + w - badge_w - 0.3
        by = y + h - title_h - badge_h - 0.3
        badge = FancyBboxPatch(
            (bx, by), badge_w, badge_h,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            facecolor="#FFE6E6", edgecolor=COLORS["novel_border"],
            linewidth=1, zorder=6,
        )
        ax.add_patch(badge)
        ax.text(bx + badge_w / 2, by + badge_h / 2, "NOVEL",
                ha="center", va="center", fontsize=7.5,
                fontweight="bold", color=COLORS["novel_border"], zorder=7)

    # Description (multi-line, role-only — no example values)
    if description:
        text_y = y + h - title_h - 1.4
        for i, line in enumerate(description):
            ax.text(x + w / 2, text_y - i * 2.0, line,
                    ha="center", va="top", fontsize=desc_size,
                    color=COLORS["text"], zorder=5)

    return (x + w / 2, y + h / 2)


def draw_arrow(p1, p2, label=None, rad=0.0, label_offset=(0, 0)):
    """Connector with optional mid-label on a white background."""
    arrow = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", linewidth=1.4,
        color=COLORS["arrow"], shrinkA=4, shrinkB=4,
        mutation_scale=14, zorder=1,
    )
    ax.add_patch(arrow)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center",
                fontsize=9.5, style="italic", color="#37474F",
                bbox=dict(facecolor="white", edgecolor="none",
                          alpha=0.95, pad=1.5), zorder=8)


# =============================================================================
# Row 1: Input  ->  Backbone (ConvNeXt V2-Tiny, NOVEL)
# =============================================================================
draw_box(3, 80, 18, 12, COLORS["input"], "Input Image",
         ["Field-captured", "paddy leaf RGB", "640 × 640 × 3"])

draw_box(26, 80, 22, 12, COLORS["backbone"], "ConvNeXt V2-Tiny",
         ["Modern CNN backbone", "for feature extraction"],
         novel=True)

# =============================================================================
# Row 2: Visual Explanation (Grad-CAM++, NOVEL) and Uncertainty (MC Dropout)
# =============================================================================
draw_box(3, 60, 22, 13, COLORS["gradcam"], "Visual Explanation",
         ["Grad-CAM++ attribution", "Second-order gradients",
          "produce heatmaps over", "disease-relevant regions"],
         novel=True)

draw_box(28, 60, 20, 13, COLORS["uncertainty"], "Uncertainty",
         ["Monte Carlo Dropout", "30 stochastic passes",
          "split into epistemic", "and aleatoric components"])

# =============================================================================
# Row 3: LLM (Phi-3-mini, NOVEL) and Recommendation (+ Knowledge Base)
# =============================================================================
draw_box(3, 36, 22, 16, COLORS["llm"], "Phi-3-mini LLM",
         ["Audience-adaptive natural-",
          "language generation",
          "fed with prediction,",
          "uncertainty and visual",
          "evidence"],
         novel=True)

draw_box(28, 36, 22, 16, COLORS["recommender"], "Recommendation Engine",
         ["Context-aware treatment",
          "guidance, conditioned on",
          "diagnosis, uncertainty",
          "and agronomic priors"])

# Knowledge Base — distinct "data store" cylinder-like shape
kb_x, kb_y, kb_w, kb_h = 55, 39, 12, 10
draw_box(kb_x, kb_y, kb_w, kb_h, COLORS["kb"], "Knowledge Base",
         ["Agronomic rules,",
          "treatments, severity",
          "thresholds"])

# =============================================================================
# Row 4: Decision Support Output  ->  End User
# =============================================================================
draw_box(8, 10, 38, 14, COLORS["output"], "Decision Support Output",
         ["Three synchronised channels:",
          "prediction + confidence  │  visual explanation",
          "│  NL explanation + recommendations"],
         desc_size=9.5)

draw_box(56, 12, 16, 10, COLORS["outchannel"], "End User / Mobile App",
         ["Farmer or", "extension officer"])

# =============================================================================
# Arrows
# =============================================================================
# Row 1: input -> backbone
draw_arrow((21, 86), (26, 86), "RGB image")

# Backbone -> Visual Explanation (down-left) and Uncertainty (down)
draw_arrow((33, 80), (14, 73), "Multi-scale features (P2–P5)",
           rad=0.15, label_offset=(2, 2))
draw_arrow((40, 80), (38, 73), "Top features (P5)",
           label_offset=(3, 0))

# Visual + Uncertainty -> LLM
draw_arrow((14, 60), (14, 52), "Visual evidence",
           label_offset=(4, 0))
draw_arrow((38, 60), (22, 52), "Prediction + uncertainty",
           rad=-0.15, label_offset=(0, 0))

# LLM -> Recommendation Engine
draw_arrow((25, 44), (28, 44), "Diagnosis context",
           label_offset=(0, 1.2))

# Knowledge Base -> Recommendation Engine
draw_arrow((kb_x, kb_y + kb_h / 2), (50, 44),
           "Agronomic rules", label_offset=(0, 1.2))

# LLM -> Output, Recommendation -> Output
draw_arrow((14, 36), (20, 24), "Explanation",
           rad=0.10, label_offset=(-1, 0))
draw_arrow((39, 36), (34, 24), "Treatment",
           rad=-0.10, label_offset=(1, 0))

# Output -> End User
draw_arrow((46, 17), (56, 17), "Deployed insights")

# =============================================================================
# Title (no decorative subtitle inside canvas)
# =============================================================================
ax.text(40, 97, "ExplainableCropAI: Multi-level Explanation Process",
        ha="center", va="center", fontsize=17,
        fontweight="bold", color="#1A237E")

# =============================================================================
# Legend (bottom-right, includes the NOVEL convention)
# =============================================================================
legend_x, legend_y = 76, 92
legend_w, legend_h = 22, 38
ax.add_patch(FancyBboxPatch(
    (legend_x, legend_y - legend_h), legend_w, legend_h,
    boxstyle="round,pad=0.02,rounding_size=0.5",
    facecolor="white", edgecolor="#90A4AE", linewidth=0.8, zorder=1,
))
ax.text(legend_x + legend_w / 2, legend_y - 2.5, "Components",
        ha="center", va="center", fontsize=12,
        fontweight="bold", color=COLORS["text"])

legend_items = [
    ("Input / Output",          COLORS["input"]),
    ("CNN Backbone",            COLORS["backbone"]),
    ("Visual Explanation",      COLORS["gradcam"]),
    ("Uncertainty",             COLORS["uncertainty"]),
    ("LLM Module",              COLORS["llm"]),
    ("Recommender",             COLORS["recommender"]),
    ("Knowledge Base",          COLORS["kb"]),
]
for i, (label, fill) in enumerate(legend_items):
    yy = legend_y - 6 - i * 3.5
    ax.add_patch(FancyBboxPatch(
        (legend_x + 1.5, yy - 1.1), 3.5, 2.2,
        boxstyle="round,pad=0.01,rounding_size=0.3",
        facecolor=fill, edgecolor="#90A4AE", linewidth=0.7, zorder=2,
    ))
    ax.text(legend_x + 6, yy, label,
            ha="left", va="center", fontsize=11,
            color=COLORS["text"], zorder=3)

# Novel-marker entry
yy = legend_y - 6 - len(legend_items) * 3.5
ax.add_patch(FancyBboxPatch(
    (legend_x + 1.5, yy - 1.1), 3.5, 2.2,
    boxstyle="round,pad=0.01,rounding_size=0.3",
    facecolor="#FFE6E6", edgecolor=COLORS["novel_border"],
    linewidth=1.4, zorder=2,
))
ax.text(legend_x + 2.5, yy + 0.05, "N",
        ha="center", va="center", fontsize=7,
        fontweight="bold", color=COLORS["novel_border"], zorder=3)
ax.text(legend_x + 6, yy, "Novel contribution",
        ha="left", va="center", fontsize=8,
        color=COLORS["text"], zorder=3)

# =============================================================================
# Save
# =============================================================================
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "explanation_flowchart.png"),
            dpi=300, bbox_inches="tight",
            facecolor="white", edgecolor="none")
plt.savefig(os.path.join(output_dir, "explanation_flowchart.pdf"),
            format="pdf", bbox_inches="tight",
            facecolor="white", edgecolor="none")

print(f"Explanation flowchart saved to {output_dir}")
plt.close()
