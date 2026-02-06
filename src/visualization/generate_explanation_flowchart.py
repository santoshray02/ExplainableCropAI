#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate flowchart visualization of the ExplainableCropAI explanation process.

This script creates a detailed visualization showing the step-by-step process
of how an input image is processed through the system to generate multi-level
explanations and actionable recommendations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as path_effects

# Create output directory
output_dir = '../../paper/figures'
os.makedirs(output_dir, exist_ok=True)

# Set up the figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Define colors
colors = {
    'background': '#FFFFFF',
    'input': '#E1F5FE',
    'model': '#E8F5E9',
    'visual_exp': '#FFF3E0',
    'uncertainty': '#F3E5F5',
    'nlg': '#E8EAF6',
    'recommendation': '#FFF8E1',
    'arrow': '#546E7A',
    'text': '#263238',
    'highlight': '#FF5722',
    'outline': '#455A64'
}

# Set background color
fig.patch.set_facecolor(colors['background'])
ax.set_facecolor(colors['background'])

# Helper function to create a box with a title and content
def create_box(x, y, width, height, color, title, content=None, image_path=None, bold_words=None):
    # Main box
    box = Rectangle((x, y), width, height, facecolor=color, edgecolor=colors['outline'], 
                    linewidth=2, alpha=0.9, zorder=1)
    ax.add_patch(box)
    
    # Title bar
    title_bar = Rectangle((x, y + height - 4), width, 4, facecolor=colors['outline'], 
                          edgecolor=colors['outline'], linewidth=1, alpha=0.9, zorder=2)
    ax.add_patch(title_bar)
    
    # Title text
    title_text = ax.text(x + width/2, y + height - 2, title, 
                     ha='center', va='center', fontsize=12, fontweight='bold',
                     color='white', zorder=3)
    title_text.set_path_effects([path_effects.withStroke(linewidth=2, foreground='black')])
    
    # Add content text if provided
    if content:
        if bold_words:
            # Split content into parts to make certain words bold
            parts = []
            remaining_text = content
            for word in bold_words:
                if word in remaining_text:
                    before, word_and_after = remaining_text.split(word, 1)
                    if before:
                        parts.append((before, 'normal'))
                    parts.append((word, 'bold'))
                    remaining_text = word_and_after
            if remaining_text:
                parts.append((remaining_text, 'normal'))
            
            # Add text with bold parts
            y_offset = 0
            for text, style in parts:
                weight = 'bold' if style == 'bold' else 'normal'
                color = colors['highlight'] if style == 'bold' else colors['text']
                
                # Handle multiple lines
                lines = text.split('\n')
                for line in lines:
                    ax.text(x + 2, y + height - 6 - y_offset, line,
                           ha='left', va='top', fontsize=10, fontweight=weight, 
                           color=color, wrap=True)
                    y_offset += 4
        else:
            ax.text(x + 2, y + height - 6, content, 
                   ha='left', va='top', fontsize=10, wrap=True,
                   color=colors['text'])
    
    # Add image if provided
    if image_path:
        try:
            img = plt.imread(image_path)
            img_box = ax.imshow(img, extent=[x+2, x+width-2, y+2, y+height-6], 
                              aspect='auto', zorder=2)
        except:
            # If image loading fails, just create a placeholder
            placeholder = Rectangle((x+2, y+2), width-4, height-8, 
                                  facecolor='lightgray', edgecolor='gray',
                                  linewidth=1, alpha=0.7, zorder=2)
            ax.add_patch(placeholder)
            ax.text(x + width/2, y + height/2 - 3, "Image\nPlaceholder", 
                   ha='center', va='center', fontsize=10, color='gray')
    
    return (x + width/2, y + height/2)

# Helper function to create an arrow between components
def create_arrow(start, end, label=None, color=colors['arrow'], style='arc3,rad=0.1'):
    arrow = FancyArrowPatch(start, end, connectionstyle=style, 
                           arrowstyle='-|>', linewidth=2, 
                           color=color, shrinkA=5, shrinkB=5, zorder=0)
    ax.add_patch(arrow)
    
    if label:
        # Calculate the midpoint of the arrow
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Add label with a white background for better readability
        text = ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=9,
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=2))

# Create the flowchart components

# 1. Input Image
input_center = create_box(5, 75, 15, 15, colors['input'], "Input Image",
                       "Paddy leaf image\ncaptured in the field")

# 2. Disease Detection Model
model_center = create_box(25, 75, 20, 15, colors['model'], "Disease Detection Model",
                        "Modified ResNet50\nPrediction: Brown Spot\nConfidence: 92%")

# 3. Visual Explanation (Grad-CAM)
visual_exp_center = create_box(5, 55, 20, 15, colors['visual_exp'], "Visual Explanation",
                            "Grad-CAM highlighting\nimportant regions\ncontributing to prediction")

# 4. Uncertainty Quantification
uncertainty_center = create_box(30, 55, 20, 15, colors['uncertainty'], "Uncertainty Quantification",
                             "Monte Carlo Dropout\nEpistemic: Low\nAleatoric: Medium\nEntropy: 0.32")

# 5. Natural Language Explanation
nlg_center = create_box(15, 35, 25, 20, colors['nlg'], "Natural Language Explanation",
                     "This image shows a paddy plant affected by Brown Spot disease, as indicated by the circular brown lesions with yellow halos visible on the leaf surface. The system is moderately confident (92%) in this diagnosis based on the distinctive pattern of spots. Brown Spot typically develops during periods of high humidity and can be distinguished from Bacterial Leaf Streak by its circular rather than elongated lesions.",
                     bold_words=["Brown Spot", "circular brown lesions with yellow halos", "moderately confident (92%)", "high humidity"])

# 6. Context-Aware Recommendation
recommendation_center = create_box(45, 35, 25, 20, colors['recommendation'], "Context-Aware Recommendation",
                                "TREATMENT RECOMMENDATION:\n1. Apply fungicide containing propiconazole (e.g., Tilt) at 1ml/L water.\n2. Ensure proper field drainage to reduce humidity.\n3. Monitor adjacent plants for early symptoms.\n\nSEVERITY: Moderate (30% leaf area affected)\nALTERNATIVE: For organic farming, apply neem oil solution at 2-3% concentration.",
                                bold_words=["TREATMENT RECOMMENDATION:", "propiconazole", "field drainage", "SEVERITY: Moderate", "ALTERNATIVE:"])

# 7. Decision Support Output
output_center = create_box(30, 10, 40, 20, colors['input'], "Decision Support Output",
                        "INTEGRATED ANALYSIS & RECOMMENDATION\n\nDisease: Brown Spot (92% confidence)\nVisual Evidence: Circular lesions with yellow halos\nSeverity: Moderate (30% leaf area affected)\nRecommendation: Apply propiconazole fungicide\nUncertainty Note: Medium uncertainty in severity assessment; consider expert consultation if symptoms worsen",
                        bold_words=["INTEGRATED ANALYSIS & RECOMMENDATION", "Brown Spot", "Circular lesions", "Moderate", "propiconazole", "Medium uncertainty"])

# Create connections between components
create_arrow(input_center, model_center, "Image input")
create_arrow(model_center, visual_exp_center, "Feature maps")
create_arrow(model_center, uncertainty_center, "Multiple forward passes")
create_arrow(visual_exp_center, nlg_center, "Visual evidence")
create_arrow(uncertainty_center, nlg_center, "Confidence level")
create_arrow(nlg_center, recommendation_center, "Diagnosis + Uncertainty")
create_arrow(recommendation_center, output_center, "Treatment options")
create_arrow(nlg_center, output_center, "Explanation text")

# Add system user (farmer) at the end
farmer_x, farmer_y = 75, 15
farmer_triangle = Polygon([(farmer_x, farmer_y), 
                         (farmer_x - 5, farmer_y - 10),
                         (farmer_x + 5, farmer_y - 10)],
                         facecolor='#795548', edgecolor='black', zorder=2)
ax.add_patch(farmer_triangle)
farmer_body = Rectangle((farmer_x - 3, farmer_y - 20), 6, 10, 
                       facecolor='#795548', edgecolor='black', zorder=2)
ax.add_patch(farmer_body)

# Add phone/device in hands
device = Rectangle((farmer_x - 5, farmer_y - 15), 10, 6,
                  facecolor='lightgray', edgecolor='black', zorder=3)
ax.add_patch(device)
ax.text(farmer_x, farmer_y - 12, "ExplainableCropAI", ha='center', va='center', 
        fontsize=6, color='black', zorder=4)

# Connect output to farmer
create_arrow(output_center, (farmer_x - 5, farmer_y - 12), "Actionable insights")

# Create a legend
legend_elements = [
    Rectangle((0, 0), 1, 1, facecolor=colors['input'], edgecolor='black', label='Input/Output'),
    Rectangle((0, 0), 1, 1, facecolor=colors['model'], edgecolor='black', label='Model Component'),
    Rectangle((0, 0), 1, 1, facecolor=colors['visual_exp'], edgecolor='black', label='Visual Explanation'),
    Rectangle((0, 0), 1, 1, facecolor=colors['uncertainty'], edgecolor='black', label='Uncertainty Quantification'),
    Rectangle((0, 0), 1, 1, facecolor=colors['nlg'], edgecolor='black', label='Natural Language Explanation'),
    Rectangle((0, 0), 1, 1, facecolor=colors['recommendation'], edgecolor='black', label='Recommendation Engine'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=colors['highlight'], label='Key Information', markersize=10)
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=10, 
         title="ExplainableCropAI Components", title_fontsize=12)

# Add title and subtitle
title = ax.text(50, 96, "ExplainableCropAI: Multi-level Explanation Process", 
             ha='center', va='center', fontsize=18, fontweight='bold', color='black')
title.set_path_effects([path_effects.withStroke(linewidth=3, foreground='white')])

subtitle = ax.text(50, 92, "From Input Image to Actionable Agricultural Decision Support", 
                ha='center', va='center', fontsize=14, fontstyle='italic', color='#546E7A')
subtitle.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

# Add benefit annotations
benefits = [
    (15, 65, "Visual transparency\nfor verification"),
    (40, 65, "Honest uncertainty\nfor appropriate trust"),
    (28, 25, "Domain-specific language\nfor accessibility"),
    (58, 25, "Context-aware advice\nfor practical utility"),
    (55, 10, "Integrated support\nfor decision-making")
]

for x, y, text in benefits:
    ax.text(x, y, text, ha='center', va='center', fontsize=8, color='#1B5E20',
           bbox=dict(facecolor='white', edgecolor='#388E3C', alpha=0.7, boxstyle='round,pad=0.5'))

# Save the figure
plt.tight_layout(rect=[0.02, 0, 0.98, 0.95])
plt.savefig(os.path.join(output_dir, 'explanation_flowchart.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'explanation_flowchart.pdf'), format='pdf', bbox_inches='tight')

print(f"Explanation flowchart saved to {output_dir}")

# Close the figure
plt.close()
