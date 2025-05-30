#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate system architecture diagram for ExplainableCropAI.

This script creates a visual representation of the ExplainableCropAI system
architecture, showing the main components and data flow.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 10))

# Define colors
colors = {
    'bg': '#f8f9fa',
    'model': '#d5e8d4',
    'explanation': '#dae8fc',
    'uncertainty': '#ffe6cc',
    'nlg': '#e1d5e7',
    'recommendation': '#fff2cc',
    'input': '#f5f5f5',
    'output': '#f5f5f5',
    'arrow': '#666666',
    'border': '#666666',
    'text': '#333333'
}

# Set background color
ax.set_facecolor(colors['bg'])
fig.patch.set_facecolor(colors['bg'])

# Helper function to create a box
def create_box(x, y, width, height, color, label, fontsize=11):
    box = patches.Rectangle((x, y), width, height, linewidth=2, 
                            edgecolor=colors['border'], facecolor=color, alpha=0.8)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color=colors['text'])
    return box

# Helper function to create an arrow
def create_arrow(start, end, label=None, position=0.5, fontsize=10, offset=0):
    # Calculate the arrow path
    start_x, start_y = start
    end_x, end_y = end
    
    # Add offset to avoid overlapping with boxes
    if start_x == end_x:  # Vertical arrow
        start_y += offset if start_y < end_y else -offset
        end_y -= offset if start_y < end_y else -offset
    else:  # Horizontal arrow
        start_x += offset if start_x < end_x else -offset
        end_x -= offset if start_x < end_x else -offset
    
    arrow = patches.FancyArrowPatch((start_x, start_y), (end_x, end_y),
                                    connectionstyle="arc3,rad=0.1", 
                                    arrowstyle="simple,head_width=10,head_length=10",
                                    linewidth=2, color=colors['arrow'])
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        # Calculate the position along the arrow
        mid_x = start_x + position * (end_x - start_x)
        mid_y = start_y + position * (end_y - start_y)
        
        # Adjust label position for vertical arrows
        if start_x == end_x:
            ax.text(mid_x + 0.3, mid_y, label, ha='left', va='center', 
                    fontsize=fontsize, color=colors['text'])
        else:
            ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom', 
                    fontsize=fontsize, color=colors['text'])

# Create boxes for main components

# Input
input_box = create_box(1, 8, 2, 1, colors['input'], 'Input Image')

# Base Model
model_box = create_box(4, 8, 3, 2, colors['model'], 'Disease Detection\nModel')

# Visual Explanation
visual_box = create_box(2, 5, 2.5, 1.5, colors['explanation'], 'Visual Explanation\n(Grad-CAM)')

# Uncertainty Quantification
uncertainty_box = create_box(5, 5, 3, 1.5, colors['uncertainty'], 'Uncertainty\nQuantification\n(MC Dropout)')

# Natural Language Explanation
nlg_box = create_box(3.5, 3, 3, 1.5, colors['nlg'], 'Natural Language\nExplanation\nGenerator')

# Recommendation Engine
recommend_box = create_box(7.5, 3, 3, 1.5, colors['recommendation'], 'Recommendation\nEngine')

# Output
output_box = create_box(5, 0.5, 4, 1.5, colors['output'], 'Multi-level Explanations\nand Recommendations')

# Create arrows showing data flow
create_arrow((3, 8.5), (4, 8.5), label='', offset=0.1)
create_arrow((5.5, 8), (5.5, 6.5), label='Features', offset=0.1)
create_arrow((5.5, 8), (3.25, 5.75), label='Activations', position=0.3, offset=0.1)
create_arrow((3.25, 5), (5, 3.75), label='Visual Evidence', position=0.6, offset=0.1)
create_arrow((6.5, 5), (5, 3.75), label='Uncertainty\nMetrics', position=0.6, offset=0.1)
create_arrow((6.5, 3.75), (7.5, 3.75), label='Prediction', offset=0.1)
create_arrow((5, 3), (7, 1.25), label='Textual\nExplanation', position=0.7, offset=0.1)
create_arrow((9, 3), (7, 1.25), label='Treatment\nRecommendations', position=0.7, offset=0.1)

# Additional arrows
create_arrow((7.5, 3.75), (9, 3.75), label='Context', offset=0.1)  # Context to recommendation

# Set axis limits and remove ticks
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.set_xticks([])
ax.set_yticks([])
ax.axis('off')

# Add title
plt.suptitle('ExplainableCropAI System Architecture', fontsize=18, fontweight='bold', y=0.98)

# Add legend
legend_elements = [
    patches.Patch(facecolor=colors['model'], edgecolor=colors['border'], label='Disease Detection'),
    patches.Patch(facecolor=colors['explanation'], edgecolor=colors['border'], label='Visual Explanation'),
    patches.Patch(facecolor=colors['uncertainty'], edgecolor=colors['border'], label='Uncertainty Quantification'),
    patches.Patch(facecolor=colors['nlg'], edgecolor=colors['border'], label='Natural Language Generation'),
    patches.Patch(facecolor=colors['recommendation'], edgecolor=colors['border'], label='Recommendation Engine')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

# Add explanatory text
fig.text(0.05, 0.02, 'Data Flow: Input Image → Disease Detection → Multi-level Explanations → Actionable Recommendations', 
         fontsize=10, style='italic')

# Save the figure
output_dir = '../../results/figures'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'system_architecture.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'system_architecture.pdf'), format='pdf', bbox_inches='tight')

print(f"System architecture diagram saved to {output_dir}")

# Close the figure
plt.close(fig)
