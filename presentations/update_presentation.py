#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Update PowerPoint presentation for ExplainableCropAI project.

This script updates the existing PowerPoint presentation to incorporate
the new high-quality visualizations created for the project.
"""

import os
from pptx import Presentation

# Path to the existing presentation
input_path = 'ExplainableCropAI_Presentation.pptx'
# Path for saving the updated presentation
output_path = 'ExplainableCropAI_Presentation_Updated.pptx'

# Load the existing presentation
prs = Presentation(input_path)

# Function to replace an image in a slide
def replace_image(slide, new_image_path, image_index=0):
    """Replace an image in a slide with a new image."""
    # Get all shape objects in the slide
    shapes = slide.shapes
    
    # Find image shapes
    image_shapes = [shape for shape in shapes if hasattr(shape, 'image')]
    
    if len(image_shapes) > image_index:
        # Get the shape to replace
        shape = image_shapes[image_index]
        # Get the shape index in the slide's shapes collection
        shape_idx = [i for i, s in enumerate(shapes) if s == shape][0]
        # Get shape dimensions
        left = shape.left
        top = shape.top
        width = shape.width
        height = shape.height
        
        # Remove the old shape
        slide.shapes._spTree.remove(shape._element)
        
        # Insert the new image with same dimensions
        slide.shapes.add_picture(new_image_path, left, top, width, height)
        
        return True
    
    return False

# Function to add a new slide with an image
def add_image_slide(title_text, image_path, caption_text=None):
    """Add a new slide with an image and caption."""
    # Use the layout with title and content
    slide_layout = prs.slide_layouts[5]  # Title and Content layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title = slide.shapes.title
    title.text = title_text
    
    # Add image
    left = int(prs.slide_width * 0.1)
    top = int(prs.slide_height * 0.25)
    width = int(prs.slide_width * 0.8)
    slide.shapes.add_picture(image_path, left, top, width=width)
    
    # Add caption if provided
    if caption_text:
        left = int(prs.slide_width * 0.1)
        top = int(prs.slide_height * 0.8)
        width = int(prs.slide_width * 0.8)
        height = int(prs.slide_height * 0.1)
        
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        p = tf.add_paragraph()
        p.text = caption_text
        p.font.size = 1000000  # Size in EMU (914400 EMU = 1 point)
        p.font.italic = True
    
    return slide

# New visualizations to add
new_visualizations = [
    {
        'title': 'Explanation Process Flow',
        'image_path': '../results/figures/explanation_flowchart.png',
        'caption': 'End-to-end explanation process in ExplainableCropAI: from input image to multi-level explanations and actionable recommendations.'
    },
    {
        'title': 'Detailed Confusion Matrix',
        'image_path': '../results/figures/confusion_matrix.png',
        'caption': 'Classification performance across disease categories, highlighting common misclassifications and per-class accuracy.'
    },
    {
        'title': 'User Study Results',
        'image_path': '../results/figures/user_study_results.png',
        'caption': 'Impact of explanations across different stakeholder groups on decision accuracy, trust, time, and workload.'
    },
    {
        'title': 'Explanation Effectiveness by Task',
        'image_path': '../results/figures/explanation_effectiveness.png',
        'caption': 'Effectiveness of different explanation types for various agricultural tasks.'
    }
]

# Update the system architecture slide (usually slide index 5 based on previous script)
system_arch_idx = 4  # 0-indexed, corresponds to Slide 5
try:
    if len(prs.slides) > system_arch_idx:
        replace_image(prs.slides[system_arch_idx], '../results/figures/system_architecture.png')
        print(f"Updated system architecture slide at index {system_arch_idx}")
except Exception as e:
    print(f"Could not update system architecture slide: {str(e)}")

# Update the visual explanations slide (usually slide index 7 based on previous script)
visual_exp_idx = 7  # 0-indexed, corresponds to Slide 8
try:
    if len(prs.slides) > visual_exp_idx:
        replace_image(prs.slides[visual_exp_idx], '../results/figures/gradcam_examples.png')
        print(f"Updated visual explanations slide at index {visual_exp_idx}")
except Exception as e:
    print(f"Could not update visual explanations slide: {str(e)}")

# Update the uncertainty visualization slide (usually slide index 8 based on previous script)
uncertainty_idx = 8  # 0-indexed, corresponds to Slide 9
try:
    if len(prs.slides) > uncertainty_idx:
        replace_image(prs.slides[uncertainty_idx], '../results/figures/uncertainty_visualization.png')
        print(f"Updated uncertainty visualization slide at index {uncertainty_idx}")
except Exception as e:
    print(f"Could not update uncertainty visualization slide: {str(e)}")

# Update the recommendations slide (usually slide index 10 based on previous script)
recommendations_idx = 10  # 0-indexed, corresponds to Slide 11
try:
    if len(prs.slides) > recommendations_idx:
        replace_image(prs.slides[recommendations_idx], '../results/figures/uncertainty_recommendations.png')
        print(f"Updated recommendations slide at index {recommendations_idx}")
except Exception as e:
    print(f"Could not update recommendations slide: {str(e)}")

# Add new slides with the new visualizations
print("Adding new slides for additional visualizations...")
for viz in new_visualizations:
    slide = add_image_slide(viz['title'], viz['image_path'], viz['caption'])
    print(f"Added new slide: {viz['title']}")

# Save the updated presentation
prs.save(output_path)
print(f"Updated presentation saved as {output_path}")
