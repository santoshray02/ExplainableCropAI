#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize uncertainty estimations for the ExplainableCropAI project.

This script provides functions to visualize Monte Carlo Dropout results,
showing prediction uncertainty in various ways to enhance explainability.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Define paths
DATA_DIR = '../../data/sample'
OUTPUT_DIR = '../../results/uncertainty_visualization'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mc_dropout_data():
    """Load Monte Carlo Dropout results."""
    mc_dropout_path = os.path.join(DATA_DIR, 'mc_dropout_results.csv')
    if os.path.exists(mc_dropout_path):
        data = pd.read_csv(mc_dropout_path)
        print(f"Loaded {len(data)} MC dropout samples")
        return data
    else:
        print("MC dropout data not found")
        return None

def calculate_uncertainty_metrics(mc_data):
    """Calculate uncertainty metrics from Monte Carlo Dropout results."""
    # Get disease names (all columns except image_id and forward_pass)
    disease_names = [col for col in mc_data.columns if col not in ['image_id', 'forward_pass']]
    
    # Group by image_id to process each image
    image_ids = mc_data['image_id'].unique()
    
    uncertainty_metrics = []
    
    for img_id in image_ids:
        img_data = mc_data[mc_data['image_id'] == img_id]
        
        # Calculate mean probability for each class
        mean_probs = img_data[disease_names].mean()
        
        # Find predicted class (highest mean probability)
        predicted_class = mean_probs.idxmax()
        prediction_prob = mean_probs.max()
        
        # Calculate predictive entropy
        # H(y|x,D) = -∑ p(y|x,D) * log(p(y|x,D))
        entropy = -np.sum(mean_probs * np.log2(mean_probs + 1e-10))
        
        # Calculate variance of predicted class probability across MC samples
        variance = img_data[predicted_class].var()
        
        # Calculate BALD (Bayesian Active Learning by Disagreement)
        # I[y,w|x,D] = H(y|x,D) - E_w[H(y|x,w,D)]
        
        # First, calculate entropy of each forward pass prediction
        sample_entropies = []
        for _, row in img_data.iterrows():
            probs = row[disease_names].values
            sample_entropy = -np.sum(probs * np.log2(probs + 1e-10))
            sample_entropies.append(sample_entropy)
        
        # BALD is the difference between predictive entropy and expected entropy
        expected_entropy = np.mean(sample_entropies)
        bald = entropy - expected_entropy
        
        uncertainty_metrics.append({
            'image_id': img_id,
            'predicted_class': predicted_class,
            'prediction_prob': prediction_prob,
            'entropy': entropy,
            'variance': variance,
            'bald': bald,
            'expected_entropy': expected_entropy
        })
    
    return pd.DataFrame(uncertainty_metrics)

def visualize_mc_distributions(mc_data, uncertainty_metrics, num_images=4):
    """
    Visualize probability distributions from Monte Carlo Dropout samples
    for selected images with different uncertainty levels.
    """
    # Sort images by uncertainty (entropy)
    sorted_images = uncertainty_metrics.sort_values('entropy', ascending=False)
    
    # Select images with different uncertainty levels
    if len(sorted_images) > num_images:
        # Get high, medium-high, medium-low, and low uncertainty images
        indices = [0, len(sorted_images)//3, 2*len(sorted_images)//3, len(sorted_images)-1]
        selected_images = sorted_images.iloc[indices]
    else:
        selected_images = sorted_images
    
    disease_names = [col for col in mc_data.columns if col not in ['image_id', 'forward_pass']]
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    for i, (_, row) in enumerate(selected_images.iterrows()):
        img_id = row['image_id']
        predicted_class = row['predicted_class']
        entropy = row['entropy']
        
        img_data = mc_data[mc_data['image_id'] == img_id]
        
        # Create subplot
        ax = fig.add_subplot(gs[i//2, i%2])
        
        # Plot distribution for predicted class
        sns.kdeplot(img_data[predicted_class], fill=True, color='blue', alpha=0.6, ax=ax)
        
        # Plot individual MC samples
        ax.scatter(img_data[predicted_class], np.zeros_like(img_data[predicted_class]), 
                  color='blue', alpha=0.7, s=30)
        
        # Add mean line
        mean_prob = img_data[predicted_class].mean()
        ax.axvline(mean_prob, color='red', linestyle='--', 
                  label=f'Mean: {mean_prob:.3f}')
        
        # Add confidence interval (±2 std)
        std_prob = img_data[predicted_class].std()
        ax.axvline(mean_prob - 2*std_prob, color='green', linestyle=':', 
                  label=f'95% CI')
        ax.axvline(mean_prob + 2*std_prob, color='green', linestyle=':')
        
        # Set titles and labels
        ax.set_title(f"Image {img_id}: {predicted_class}\nEntropy: {entropy:.3f}")
        ax.set_xlabel('Probability')
        ax.set_ylabel('Density')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mc_probability_distributions.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'mc_probability_distributions.pdf'))
    plt.close()
    
    print(f"MC probability distributions saved to {OUTPUT_DIR}")

def visualize_uncertainty_metrics(uncertainty_metrics):
    """Visualize various uncertainty metrics and their relationships."""
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Scatter plot: Prediction Probability vs Entropy
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(uncertainty_metrics['prediction_prob'], 
                         uncertainty_metrics['entropy'],
                         c=uncertainty_metrics['entropy'], 
                         cmap='viridis', 
                         s=100, alpha=0.7)
    
    # Add image IDs as labels
    for i, row in uncertainty_metrics.iterrows():
        ax1.annotate(row['image_id'], 
                    (row['prediction_prob'], row['entropy']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax1.set_title('Prediction Confidence vs. Uncertainty (Entropy)')
    ax1.set_xlabel('Prediction Probability')
    ax1.set_ylabel('Entropy')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Entropy')
    
    # 2. Bar plot: Entropy for each image
    ax2 = fig.add_subplot(gs[0, 1])
    sorted_entropy = uncertainty_metrics.sort_values('entropy', ascending=False)
    bars = ax2.bar(sorted_entropy['image_id'], sorted_entropy['entropy'], 
                  color=plt.cm.viridis(sorted_entropy['entropy']/sorted_entropy['entropy'].max()))
    
    ax2.set_title('Entropy by Image (Sorted)')
    ax2.set_xlabel('Image ID')
    ax2.set_ylabel('Entropy')
    ax2.set_xticklabels(sorted_entropy['image_id'], rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Scatter plot: BALD vs Variance
    ax3 = fig.add_subplot(gs[1, 0])
    scatter2 = ax3.scatter(uncertainty_metrics['variance'], 
                          uncertainty_metrics['bald'],
                          c=uncertainty_metrics['entropy'], 
                          cmap='viridis', 
                          s=100, alpha=0.7)
    
    # Add image IDs as labels
    for i, row in uncertainty_metrics.iterrows():
        ax3.annotate(row['image_id'], 
                    (row['variance'], row['bald']),
                    xytext=(5, 5), textcoords='offset points')
    
    ax3.set_title('Prediction Variance vs. BALD')
    ax3.set_xlabel('Variance of Predicted Class')
    ax3.set_ylabel('BALD (Mutual Information)')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax3, label='Entropy')
    
    # 4. Grouped bar chart of different uncertainty metrics
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Normalize metrics to 0-1 range for comparison
    metrics_to_plot = ['entropy', 'variance', 'bald']
    normalized_metrics = uncertainty_metrics[['image_id'] + metrics_to_plot].copy()
    
    for metric in metrics_to_plot:
        max_val = normalized_metrics[metric].max()
        if max_val > 0:  # Avoid division by zero
            normalized_metrics[metric] = normalized_metrics[metric] / max_val
    
    # Reshape for seaborn
    plot_data = pd.melt(normalized_metrics, 
                        id_vars=['image_id'], 
                        value_vars=metrics_to_plot,
                        var_name='Metric', value_name='Normalized Value')
    
    # Sort by entropy
    entropy_order = normalized_metrics.sort_values('entropy', ascending=False)['image_id'].tolist()
    plot_data['image_id'] = pd.Categorical(plot_data['image_id'], categories=entropy_order, ordered=True)
    
    # Create grouped bar chart
    sns.barplot(x='image_id', y='Normalized Value', hue='Metric', data=plot_data, ax=ax4)
    
    ax4.set_title('Comparison of Normalized Uncertainty Metrics')
    ax4.set_xlabel('Image ID')
    ax4.set_ylabel('Normalized Value')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(title='Metric')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'uncertainty_metrics_comparison.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'uncertainty_metrics_comparison.pdf'))
    plt.close()
    
    print(f"Uncertainty metrics comparison saved to {OUTPUT_DIR}")

def visualize_prediction_stability(mc_data):
    """
    Visualize the stability of predictions across Monte Carlo samples.
    Shows how predictions might change across forward passes.
    """
    # Get disease names
    disease_names = [col for col in mc_data.columns if col not in ['image_id', 'forward_pass']]
    
    # Select a few interesting images (those with highest disagreement in predictions)
    image_ids = mc_data['image_id'].unique()
    
    # Calculate prediction stability for each image
    stability_scores = []
    
    for img_id in image_ids:
        img_data = mc_data[mc_data['image_id'] == img_id]
        
        # Get top prediction for each forward pass
        top_preds = []
        for _, row in img_data.iterrows():
            probs = row[disease_names]
            top_pred = probs.idxmax()
            top_preds.append(top_pred)
        
        # Calculate stability score (percentage of agreement with mode)
        mode_pred = pd.Series(top_preds).mode()[0]
        stability = sum(pred == mode_pred for pred in top_preds) / len(top_preds)
        
        stability_scores.append({
            'image_id': img_id,
            'stability': stability,
            'mode_prediction': mode_pred,
            'unique_predictions': len(set(top_preds))
        })
    
    stability_df = pd.DataFrame(stability_scores)
    
    # Sort by stability (ascending) and select most interesting cases
    sorted_stability = stability_df.sort_values('stability')
    
    # If we have at least 2 interesting cases (stability < 1.0), select those
    # Otherwise select the first 2 images
    interesting_cases = sorted_stability[sorted_stability['stability'] < 1.0]
    if len(interesting_cases) >= 2:
        selected_images = interesting_cases.iloc[:2]
    else:
        selected_images = sorted_stability.iloc[:2]
    
    # Create visualization
    fig, axs = plt.subplots(len(selected_images), 1, figsize=(12, 8*len(selected_images)))
    if len(selected_images) == 1:
        axs = [axs]
    
    for i, (_, row) in enumerate(selected_images.iterrows()):
        img_id = row['image_id']
        stability = row['stability']
        mode_pred = row['mode_prediction']
        
        img_data = mc_data[mc_data['image_id'] == img_id]
        
        # Create a grid of subplots for each forward pass
        num_passes = len(img_data)
        
        # Calculate grid dimensions
        grid_cols = 5  # Fixed number of columns
        grid_rows = (num_passes + grid_cols - 1) // grid_cols  # Ceiling division
        
        gs_inner = GridSpec(grid_rows, grid_cols, figure=fig, 
                           left=0.1, right=0.9, top=0.9 - i*0.5, bottom=0.1 - i*0.5)
        
        # Create title for this set
        axs[i].set_title(f"Image {img_id} - Prediction Stability: {stability:.2f}\n"
                       f"Mode Prediction: {mode_pred}, Unique Predictions: {row['unique_predictions']}")
        axs[i].axis('off')
        
        # Create a small visualization for each forward pass
        for j, (_, fp_row) in enumerate(img_data.iterrows()):
            fp = fp_row['forward_pass']
            
            # Calculate grid position
            grid_row = j // grid_cols
            grid_col = j % grid_cols
            
            # Create subplot
            ax_inner = fig.add_subplot(gs_inner[grid_row, grid_col])
            
            # Get probabilities
            probs = fp_row[disease_names].sort_values(ascending=False)
            top_pred = probs.index[0]
            
            # Plot horizontal bar chart of top 3 probabilities
            top3_probs = probs.iloc[:3]
            colors = ['green' if pred == mode_pred else 'orange' for pred in top3_probs.index]
            
            ax_inner.barh(top3_probs.index, top3_probs.values, color=colors)
            ax_inner.set_title(f"Pass {int(fp)}: {top_pred}")
            ax_inner.set_xlim(0, 1)
            
            # Remove y labels after first column for cleaner look
            if grid_col > 0:
                ax_inner.set_yticklabels([])
    
    # Add a legend
    green_patch = mpatches.Patch(color='green', label='Mode Prediction')
    orange_patch = mpatches.Patch(color='orange', label='Different Prediction')
    fig.legend(handles=[green_patch, orange_patch], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_stability.png'), dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, 'prediction_stability.pdf'))
    plt.close()
    
    print(f"Prediction stability visualization saved to {OUTPUT_DIR}")

def create_uncertainty_dashboard(mc_data, uncertainty_metrics):
    """Create a comprehensive dashboard of uncertainty visualizations."""
    # Create a 2x2 dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
    
    # 1. Top-left: Entropy vs Confidence scatter plot
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(uncertainty_metrics['prediction_prob'], 
                         uncertainty_metrics['entropy'],
                         c=uncertainty_metrics['bald'], 
                         cmap='plasma', 
                         s=120, alpha=0.8)
    
    # Add image IDs as labels
    for i, row in uncertainty_metrics.iterrows():
        ax1.annotate(row['image_id'], 
                    (row['prediction_prob'], row['entropy']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=10)
    
    ax1.set_title('Prediction Confidence vs. Uncertainty', fontsize=14)
    ax1.set_xlabel('Prediction Probability', fontsize=12)
    ax1.set_ylabel('Entropy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('BALD (Mutual Information)', fontsize=12)
    
    # 2. Top-right: Prediction stability for the most uncertain image
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Find the most uncertain image
    most_uncertain_img = uncertainty_metrics.loc[uncertainty_metrics['entropy'].idxmax()]
    img_id = most_uncertain_img['image_id']
    img_data = mc_data[mc_data['image_id'] == img_id]
    
    disease_names = [col for col in mc_data.columns if col not in ['image_id', 'forward_pass']]
    
    # Calculate mean probabilities
    mean_probs = img_data[disease_names].mean().sort_values(ascending=False)
    std_probs = img_data[disease_names].std().reindex(mean_probs.index)
    
    # Plot top 5 classes
    top5_mean = mean_probs.head(5)
    top5_std = std_probs.head(5)
    
    # Horizontal bar chart with error bars
    ax2.barh(top5_mean.index, top5_mean.values, xerr=top5_std.values, 
            color='skyblue', alpha=0.7, capsize=5)
    
    ax2.set_title(f'Most Uncertain Image: {img_id}\nEntropy: {most_uncertain_img["entropy"]:.3f}', 
                 fontsize=14)
    ax2.set_xlabel('Mean Probability (with std dev)', fontsize=12)
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Bottom-left: Comparison of normalized uncertainty metrics
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Normalize metrics to 0-1 range for comparison
    metrics_to_plot = ['entropy', 'variance', 'bald']
    normalized_metrics = uncertainty_metrics[['image_id'] + metrics_to_plot].copy()
    
    for metric in metrics_to_plot:
        max_val = normalized_metrics[metric].max()
        if max_val > 0:  # Avoid division by zero
            normalized_metrics[metric] = normalized_metrics[metric] / max_val
    
    # Reshape for seaborn
    plot_data = pd.melt(normalized_metrics, 
                        id_vars=['image_id'], 
                        value_vars=metrics_to_plot,
                        var_name='Metric', value_name='Normalized Value')
    
    # Sort by entropy
    entropy_order = normalized_metrics.sort_values('entropy', ascending=False)['image_id'].tolist()
    plot_data['image_id'] = pd.Categorical(plot_data['image_id'], categories=entropy_order, ordered=True)
    
    # Create grouped bar chart
    sns.barplot(x='image_id', y='Normalized Value', hue='Metric', data=plot_data, ax=ax3)
    
    ax3.set_title('Comparison of Normalized Uncertainty Metrics', fontsize=14)
    ax3.set_xlabel('Image ID (sorted by entropy)', fontsize=12)
    ax3.set_ylabel('Normalized Value', fontsize=12)
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(title='Metric', fontsize=10)
    
    # 4. Bottom-right: MC samples distribution for most confident image
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Find the most confident image
    most_confident_img = uncertainty_metrics.loc[uncertainty_metrics['entropy'].idxmin()]
    conf_img_id = most_confident_img['image_id']
    conf_img_data = mc_data[mc_data['image_id'] == conf_img_id]
    
    # Get predicted class
    predicted_class = most_confident_img['predicted_class']
    
    # Plot distribution for predicted class
    sns.kdeplot(conf_img_data[predicted_class], fill=True, color='green', alpha=0.6, ax=ax4)
    
    # Plot individual MC samples
    ax4.scatter(conf_img_data[predicted_class], np.zeros_like(conf_img_data[predicted_class]), 
              color='green', alpha=0.7, s=30)
    
    # Add mean line
    mean_prob = conf_img_data[predicted_class].mean()
    ax4.axvline(mean_prob, color='red', linestyle='--', 
              label=f'Mean: {mean_prob:.3f}')
    
    # Add confidence interval (±2 std)
    std_prob = conf_img_data[predicted_class].std()
    ax4.axvline(mean_prob - 2*std_prob, color='blue', linestyle=':', 
              label=f'95% CI')
    ax4.axvline(mean_prob + 2*std_prob, color='blue', linestyle=':')
    
    ax4.set_title(f'Most Confident Image: {conf_img_id}\n{predicted_class}, Entropy: {most_confident_img["entropy"]:.3f}', 
                 fontsize=14)
    ax4.set_xlabel('Probability', fontsize=12)
    ax4.set_ylabel('Density', fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Add main title
    fig.suptitle('Uncertainty Visualization Dashboard for ExplainableCropAI', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(OUTPUT_DIR, 'uncertainty_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, 'uncertainty_dashboard.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Uncertainty dashboard saved to {OUTPUT_DIR}")

def main():
    """Main function to run all visualizations."""
    print("\n===== Uncertainty Visualization for ExplainableCropAI =====\n")
    
    # Load data
    mc_data = load_mc_dropout_data()
    if mc_data is None:
        return
    
    # Calculate uncertainty metrics
    print("\nCalculating uncertainty metrics...")
    uncertainty_metrics = calculate_uncertainty_metrics(mc_data)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_mc_distributions(mc_data, uncertainty_metrics)
    visualize_uncertainty_metrics(uncertainty_metrics)
    visualize_prediction_stability(mc_data)
    create_uncertainty_dashboard(mc_data, uncertainty_metrics)
    
    print("\nAll visualizations complete. Results saved in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
