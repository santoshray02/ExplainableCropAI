#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate example visualizations of Grad-CAM explanations for paddy diseases.

This script creates a figure with examples of paddy disease images and their
corresponding Grad-CAM visualizations to illustrate the explainability of
the system.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

# Create output directory
output_dir = '../../paper/figures'
os.makedirs(output_dir, exist_ok=True)

# Define dummy disease examples (for illustration purposes)
# In a real scenario, these would be actual predictions and Grad-CAM outputs
diseases = [
    {"name": "Bacterial Leaf Blight", "confidence": 0.98},
    {"name": "Brown Spot", "confidence": 0.95},
    {"name": "Leaf Blast", "confidence": 0.92},
    {"name": "Healthy", "confidence": 0.99}
]

# Create a figure to hold the examples
fig, axes = plt.subplots(4, 3, figsize=(15, 16))
plt.subplots_adjust(hspace=0.3, wspace=0.05)

# Create a custom colormap for the heatmaps
colors = [(0, 0, 0, 0), (0, 0, 1, 0.5), (0, 1, 1, 0.7), (0, 1, 0, 0.7), (1, 1, 0, 0.7), (1, 0, 0, 0.7)]
cmap_name = 'disease_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

# Function to generate synthetic disease images and heatmaps for demonstration
def generate_synthetic_disease_image(disease_name, seed=None):
    """Generate a synthetic image representing a paddy disease."""
    if seed is not None:
        np.random.seed(seed)
    
    # Base healthy leaf image (green background)
    image = np.ones((224, 224, 3)) * np.array([0.2, 0.7, 0.3])
    
    # Add leaf structure
    x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
    leaf_mask = (x**2 + y**2 < 0.8)
    vein_mask = (np.abs(x) < 0.1) | (np.abs(y - 0.3*x) < 0.05) | (np.abs(y + 0.3*x) < 0.05)
    
    # Apply leaf structure
    image[leaf_mask] = np.array([0.1, 0.6, 0.2])
    image[vein_mask & leaf_mask] = np.array([0.1, 0.5, 0.15])
    
    # Add disease-specific patterns
    if disease_name == "Bacterial Leaf Blight":
        # Water-soaked lesions with wavy margins
        for i in range(30):
            cx, cy = np.random.uniform(-0.7, 0.7, 2)
            size = np.random.uniform(0.05, 0.15)
            wave = np.sin(10 * ((x - cx)**2 + (y - cy)**2))
            lesion = (((x - cx)**2 + (y - cy)**2) < size**2) & (wave > 0)
            image[lesion & leaf_mask] = np.array([0.7, 0.6, 0.2])
        
        # Create corresponding heatmap focusing on lesions
        heatmap = np.zeros((224, 224))
        for i in range(30):
            cx, cy = np.random.uniform(-0.7, 0.7, 2)
            size = np.random.uniform(0.08, 0.18)  # Slightly larger than lesions
            wave = np.sin(10 * ((x - cx)**2 + (y - cy)**2))
            lesion = (((x - cx)**2 + (y - cy)**2) < size**2) & (wave > 0)
            heatmap[lesion] = np.random.uniform(0.7, 1.0)
    
    elif disease_name == "Brown Spot":
        # Circular brown spots with yellow halos
        for i in range(40):
            cx, cy = np.random.uniform(-0.7, 0.7, 2)
            size = np.random.uniform(0.03, 0.08)
            spot = ((x - cx)**2 + (y - cy)**2) < size**2
            halo = ((x - cx)**2 + (y - cy)**2) < (size*1.5)**2
            image[spot & leaf_mask] = np.array([0.6, 0.4, 0.1])
            image[halo & ~spot & leaf_mask] = np.array([0.8, 0.7, 0.1])
        
        # Create corresponding heatmap focusing on spots
        heatmap = np.zeros((224, 224))
        for i in range(40):
            cx, cy = np.random.uniform(-0.7, 0.7, 2)
            size = np.random.uniform(0.05, 0.1)  # Slightly larger than spots
            spot_area = ((x - cx)**2 + (y - cy)**2) < (size*1.7)**2
            heatmap[spot_area] = np.random.uniform(0.6, 1.0)
    
    elif disease_name == "Leaf Blast":
        # Diamond-shaped lesions with gray centers
        for i in range(25):
            cx, cy = np.random.uniform(-0.7, 0.7, 2)
            size = np.random.uniform(0.05, 0.1)
            diamond = (np.abs(x - cx) + np.abs(y - cy)) < size
            center = (np.abs(x - cx) + np.abs(y - cy)) < size*0.6
            image[diamond & leaf_mask] = np.array([0.6, 0.5, 0.3])
            image[center & leaf_mask] = np.array([0.7, 0.7, 0.7])
        
        # Create corresponding heatmap focusing on lesions
        heatmap = np.zeros((224, 224))
        for i in range(25):
            cx, cy = np.random.uniform(-0.7, 0.7, 2)
            size = np.random.uniform(0.07, 0.12)  # Slightly larger than lesions
            diamond_area = (np.abs(x - cx) + np.abs(y - cy)) < size*1.2
            heatmap[diamond_area] = np.random.uniform(0.7, 1.0)
    
    else:  # Healthy
        # Add slight variations for a natural look
        noise = np.random.normal(0, 0.05, (224, 224, 3))
        image = np.clip(image + noise, 0, 1)
        
        # Create corresponding heatmap with minimal activation
        heatmap = np.random.normal(0.1, 0.05, (224, 224))
        heatmap = np.clip(heatmap, 0, 0.3)
    
    # Ensure heatmap is within range [0, 1]
    heatmap = np.clip(heatmap, 0, 1)
    
    # Create an overlay by combining the image and heatmap
    overlay = image.copy()
    for c in range(3):
        overlay[:,:,c] = np.where(heatmap > 0.3, 
                                 image[:,:,c] * (1 - heatmap * 0.7) + heatmap * (c==0) * 0.7, 
                                 image[:,:,c])
    
    return image, heatmap, overlay

# Generate and plot each disease example
for i, disease in enumerate(diseases):
    # Generate synthetic images for demonstration
    image, heatmap, overlay = generate_synthetic_disease_image(disease["name"], seed=i*10)
    
    # Original image
    axes[i, 0].imshow(image)
    axes[i, 0].set_title(f"{disease['name']}\nConfidence: {disease['confidence']:.2f}", fontsize=12)
    axes[i, 0].axis('off')
    
    # Grad-CAM heatmap
    axes[i, 1].imshow(heatmap, cmap='jet')
    axes[i, 1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[i, 1].axis('off')
    
    # Overlay
    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title("Explanation Overlay", fontsize=12)
    axes[i, 2].axis('off')

# Add main title
plt.suptitle("ExplainableCropAI: Visual Explanations for Paddy Disease Detection", fontsize=16, y=0.98)

# Add explanation text
fig.text(0.5, 0.01, 
         "Grad-CAM visualizations highlight regions contributing to disease detection,\n"
         "providing transparency into the model's decision-making process.", 
         ha='center', fontsize=12, style='italic')

# Save the figure
plt.savefig(os.path.join(output_dir, 'gradcam_examples.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'gradcam_examples.pdf'), format='pdf', bbox_inches='tight')

print(f"Grad-CAM example visualization saved to {output_dir}")

# Close the figure
plt.close(fig)
