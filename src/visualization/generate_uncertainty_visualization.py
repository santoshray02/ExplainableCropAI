#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate visualizations of uncertainty estimation for paddy disease predictions.

This script creates figures illustrating how Monte Carlo Dropout provides
uncertainty metrics and confidence intervals for disease predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.ticker import PercentFormatter

# Create output directory
output_dir = '../../paper/figures'
os.makedirs(output_dir, exist_ok=True)

# Set the aesthetic style of the plots
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']

# Define classes matching Paddy Doctor dataset (10 classes)
classes = [
    'Bacterial Leaf Blight', 'Brown Spot', 'Leaf Blast', 'Bacterial Leaf Streak',
    'Bacterial Panicle Blight', 'Healthy', 'Tungro', 'Hispa', 'Dead Heart', 'Downy Mildew'
]

# Generate synthetic MC dropout samples for demonstration
# In a real scenario, these would be actual predictions from multiple forward passes
def generate_mc_samples(true_class_idx, num_samples=50, seed=None):
    """Generate synthetic MC Dropout samples for demonstration."""
    if seed is not None:
        np.random.seed(seed)
    
    # Base probabilities - higher for the true class
    base_probs = np.ones(len(classes)) * 0.1
    base_probs[true_class_idx] = 0.7
    
    # Generate samples with noise
    samples = np.zeros((num_samples, len(classes)))
    for i in range(num_samples):
        # Add noise to probabilities
        noise = np.random.normal(0, 0.1, len(classes))
        probs = base_probs + noise
        # Ensure they sum to 1
        probs = np.maximum(probs, 0.001)  # Ensure no negative values
        probs = probs / probs.sum()
        samples[i] = probs
    
    return samples

# Create a figure for uncertainty visualization
fig = plt.figure(figsize=(18, 12))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

# Create subplots
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)  # MC Dropout samples for one image
ax2 = plt.subplot2grid((2, 3), (0, 2))             # Entropy histogram
ax3 = plt.subplot2grid((2, 3), (1, 0))             # Predictive entropy across classes
ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=2)  # Confidence intervals for all classes

# 1. Violin plot of MC Dropout samples for a correctly classified image
true_class_idx = 0  # Bacterial Leaf Blight
samples = generate_mc_samples(true_class_idx, num_samples=50, seed=42)

# Reshape data for seaborn
data = []
for i, class_name in enumerate(classes):
    for sample in samples:
        data.append({'Class': class_name, 'Probability': sample[i]})

# Convert list to DataFrame
data_df = pd.DataFrame(data)
# Create violin plot
sns.violinplot(x='Class', y='Probability', data=data_df, inner='quartile', ax=ax1)
ax1.set_title('MC Dropout Samples for Disease Prediction', fontsize=14)
ax1.set_xlabel('Disease Class', fontsize=12)
ax1.set_ylabel('Predicted Probability', fontsize=12)
ax1.set_ylim(0, 1)
ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
ax1.tick_params(axis='x', rotation=30)

# Highlight the correct class
ax1.axvspan(-0.5, 0.5, alpha=0.2, color='green')
ax1.text(0, 0.95, 'Correct Class', ha='center', va='top', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.2))

# 2. Histogram of predictive entropy
def entropy(probs):
    """Calculate entropy of a probability distribution."""
    return -np.sum(probs * np.log2(probs + 1e-12))

# Calculate entropy for a set of predictions (mixture of correct and incorrect)
entropy_values = []
for i in range(100):
    # Generate some examples with different levels of uncertainty
    class_idx = np.random.choice([0, 1, 2, 3])
    uncertainty_level = np.random.choice(['low', 'medium', 'high'])
    
    if uncertainty_level == 'low':
        # Low uncertainty - clear predictions
        base_prob = 0.85
    elif uncertainty_level == 'medium':
        # Medium uncertainty
        base_prob = 0.6
    else:
        # High uncertainty
        base_prob = 0.4
    
    # Base probabilities
    probs = np.ones(len(classes)) * ((1 - base_prob) / (len(classes) - 1))
    probs[class_idx] = base_prob
    
    # Add noise
    noise = np.random.normal(0, 0.05, len(classes))
    probs = probs + noise
    probs = np.maximum(probs, 0.001)  # Ensure no negative values
    probs = probs / probs.sum()
    
    entropy_values.append((entropy(probs), uncertainty_level))

# Plot entropy histogram
entropy_data = [e for e, _ in entropy_values]
sns.histplot(entropy_data, kde=True, ax=ax2)
ax2.set_title('Distribution of Predictive Entropy', fontsize=14)
ax2.set_xlabel('Entropy (bits)', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)

# Add markers for different uncertainty levels
low_entropy = np.mean([e for e, u in entropy_values if u == 'low'])
medium_entropy = np.mean([e for e, u in entropy_values if u == 'medium'])
high_entropy = np.mean([e for e, u in entropy_values if u == 'high'])

ax2.axvline(low_entropy, color='green', linestyle='--', alpha=0.7)
ax2.axvline(medium_entropy, color='orange', linestyle='--', alpha=0.7)
ax2.axvline(high_entropy, color='red', linestyle='--', alpha=0.7)

ax2.text(low_entropy, ax2.get_ylim()[1]*0.9, 'Low\nUncertainty', ha='center', 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.2))
ax2.text(high_entropy, ax2.get_ylim()[1]*0.6, 'High\nUncertainty', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.2))

# 3. Bar chart of predictive entropy across different classes
class_entropies = []
for i, class_name in enumerate(classes):
    # Generate samples for each class
    samples = generate_mc_samples(i, num_samples=30, seed=i*10)
    
    # Calculate mean prediction
    mean_pred = np.mean(samples, axis=0)
    
    # Calculate entropy
    ent = entropy(mean_pred)
    class_entropies.append((class_name, ent))

# Sort by entropy
class_entropies.sort(key=lambda x: x[1])

# Plot
class_names = [c for c, _ in class_entropies]
entropies = [e for _, e in class_entropies]
colors = ['green' if e < 0.7 else 'orange' if e < 1.2 else 'red' for e in entropies]

ax3.bar(class_names, entropies, color=colors, alpha=0.7)
ax3.set_title('Predictive Entropy by Disease Class', fontsize=14)
ax3.set_xlabel('Disease Class', fontsize=12)
ax3.set_ylabel('Entropy (bits)', fontsize=12)
ax3.tick_params(axis='x', rotation=30)

# Add threshold lines
ax3.axhline(0.7, color='green', linestyle='--', alpha=0.7)
ax3.axhline(1.2, color='red', linestyle='--', alpha=0.7)
ax3.text(0, 0.7, 'Low Uncertainty', ha='right', va='bottom', 
         bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.2))
ax3.text(0, 1.2, 'High Uncertainty', ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.2))

# 4. Confidence intervals for disease predictions
# Generate predictions with confidence intervals for multiple examples
num_examples = 5
example_names = [f"Example {i+1}" for i in range(num_examples)]
true_classes = np.random.choice(len(classes), size=num_examples)

# Generate mean and confidence intervals for each example
results = []
for i, true_class in enumerate(true_classes):
    samples = generate_mc_samples(true_class, num_samples=30, seed=i*100)
    mean_pred = np.mean(samples, axis=0)
    
    # Find class with highest mean probability
    predicted_class = np.argmax(mean_pred)
    
    # Calculate 95% confidence interval for the predicted class
    sorted_samples = np.sort(samples[:, predicted_class])
    lower_ci = sorted_samples[int(0.025 * len(sorted_samples))]
    upper_ci = sorted_samples[int(0.975 * len(sorted_samples))]
    
    results.append({
        'example': example_names[i],
        'true_class': classes[true_class],
        'predicted_class': classes[predicted_class],
        'mean_prob': mean_pred[predicted_class],
        'lower_ci': lower_ci,
        'upper_ci': upper_ci,
        'correct': true_class == predicted_class,
        'uncertainty': upper_ci - lower_ci
    })

# Sort by uncertainty
results.sort(key=lambda x: x['uncertainty'])

# Plot confidence intervals
example_positions = np.arange(len(results))
for i, res in enumerate(results):
    color = 'green' if res['correct'] else 'red'
    ax4.plot([example_positions[i], example_positions[i]], 
             [res['lower_ci'], res['upper_ci']], 
             color=color, linewidth=2)
    ax4.plot(example_positions[i], res['mean_prob'], 'o', 
             color=color, markersize=8)

# Add labels and customization
ax4.set_xticks(example_positions)
ax4.set_xticklabels([f"{r['example']}\n({r['predicted_class']})" for r in results], rotation=0)
ax4.set_yticks(np.arange(0, 1.1, 0.1))
ax4.yaxis.set_major_formatter(PercentFormatter(1.0))
ax4.set_ylim(0, 1)
ax4.set_title('Prediction Confidence Intervals with Uncertainty', fontsize=14)
ax4.set_xlabel('Example (Predicted Class)', fontsize=12)
ax4.set_ylabel('Prediction Probability with 95% CI', fontsize=12)

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', marker='o', linestyle='-', markersize=8, 
           label='Correct Prediction'),
    Line2D([0], [0], color='red', marker='o', linestyle='-', markersize=8, 
           label='Incorrect Prediction')
]
ax4.legend(handles=legend_elements, loc='lower right')

# Add annotations
for i, res in enumerate(results):
    # Add uncertainty value as text
    uncertainty_pct = res['uncertainty'] * 100
    ax4.text(example_positions[i], res['upper_ci'] + 0.05, 
             f"±{uncertainty_pct:.1f}%", ha='center', fontsize=9)
    
    # Indicate true class if prediction is wrong
    if not res['correct']:
        ax4.text(example_positions[i], res['lower_ci'] - 0.1, 
                 f"True: {res['true_class']}", ha='center', fontsize=9, 
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.1))

# Add main title
plt.suptitle("ExplainableCropAI: Uncertainty Quantification in Disease Prediction", 
             fontsize=16, y=0.98)

# Add explanation text
fig.text(0.5, 0.01, 
         "Uncertainty estimation provides confidence measures for disease predictions,\n"
         "helping farmers understand when to trust the model's output and when to seek expert advice.", 
         ha='center', fontsize=12, style='italic')

# Save the figure
plt.savefig(os.path.join(output_dir, 'uncertainty_visualization.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'uncertainty_visualization.pdf'), format='pdf', bbox_inches='tight')

print(f"Uncertainty visualization saved to {output_dir}")

# Close the figure
plt.close(fig)

# Create another figure to demonstrate the relationship between uncertainty and recommendation confidence
plt.figure(figsize=(14, 8))

# Set up the grid
ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 3), (0, 2))
ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=3)

# 1. Scatter plot of Uncertainty vs. Recommendation Confidence
np.random.seed(42)
num_points = 100
uncertainties = np.random.uniform(0, 2, num_points)  # Entropy values
confidence_factor = 1 - (uncertainties / 2) + np.random.normal(0, 0.1, num_points)
confidence_factor = np.clip(confidence_factor, 0, 1)

# Define regions
high_confidence = confidence_factor > 0.7
medium_confidence = (confidence_factor <= 0.7) & (confidence_factor > 0.4)
low_confidence = confidence_factor <= 0.4

# Plot
ax1.scatter(uncertainties[high_confidence], confidence_factor[high_confidence], 
           color='green', alpha=0.7, label='High Confidence')
ax1.scatter(uncertainties[medium_confidence], confidence_factor[medium_confidence], 
           color='orange', alpha=0.7, label='Medium Confidence')
ax1.scatter(uncertainties[low_confidence], confidence_factor[low_confidence], 
           color='red', alpha=0.7, label='Low Confidence')

# Add a trend line
z = np.polyfit(uncertainties, confidence_factor, 1)
p = np.poly1d(z)
ax1.plot(np.linspace(0, 2, 100), p(np.linspace(0, 2, 100)), 
         'k--', alpha=0.7, label='Trend')

ax1.set_xlabel('Prediction Uncertainty (Entropy)', fontsize=12)
ax1.set_ylabel('Recommendation Confidence Factor', fontsize=12)
ax1.set_title('Relationship between Prediction Uncertainty and Recommendation Confidence', 
              fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Add region markers
ax1.axhline(0.7, color='green', linestyle='--', alpha=0.5)
ax1.axhline(0.4, color='red', linestyle='--', alpha=0.5)
ax1.text(1.9, 0.7, 'High', ha='right', va='bottom', 
         bbox=dict(boxstyle='round,pad=0.2', facecolor='green', alpha=0.2))
ax1.text(1.9, 0.4, 'Low', ha='right', va='bottom', 
         bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.2))

# 2. Pie chart of recommendation types
labels = ['Specific Treatment', 'General Guidance', 'Expert Consultation']
sizes = [sum(high_confidence), sum(medium_confidence), sum(low_confidence)]
colors = ['green', 'orange', 'red']
explode = (0.1, 0, 0)  # explode the 1st slice

ax2.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax2.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
ax2.set_title('Distribution of Recommendation Types', fontsize=14)

# 3. Examples of recommendations with different confidence levels
examples = [
    {
        'disease': 'Bacterial Leaf Blight',
        'confidence': 0.94,
        'uncertainty': 0.2,
        'recommendation': 'Apply copper-based bactericide such as copper oxychloride (50% WP) @ 1250g/ha. '
                         'Ensure proper drainage in the field.',
        'confidence_level': 'High'
    },
    {
        'disease': 'Brown Spot',
        'confidence': 0.75,
        'uncertainty': 0.8,
        'recommendation': 'Consider applying fungicides containing propiconazole or trifloxystrobin. '
                         'Monitor field conditions closely.',
        'confidence_level': 'Medium'
    },
    {
        'disease': 'Leaf Blast',
        'confidence': 0.52,
        'uncertainty': 1.5,
        'recommendation': 'Multiple diseases possible. Consult local agricultural extension officer. '
                         'Send sample for laboratory testing.',
        'confidence_level': 'Low'
    }
]

# Create a table
cell_text = []
colors = []
for ex in examples:
    row = [
        ex['disease'],
        f"{ex['confidence']:.2f}",
        f"{ex['uncertainty']:.2f}",
        ex['recommendation']
    ]
    cell_text.append(row)
    
    if ex['confidence_level'] == 'High':
        colors.append(['white', 'white', 'white', 'palegreen'])
    elif ex['confidence_level'] == 'Medium':
        colors.append(['white', 'white', 'white', 'khaki'])
    else:
        colors.append(['white', 'white', 'white', 'lightcoral'])

ax3.axis('tight')
ax3.axis('off')
table = ax3.table(cellText=cell_text,
                  colLabels=['Disease', 'Prediction\nConfidence', 'Uncertainty\n(Entropy)', 'Recommendation'],
                  loc='center',
                  cellColours=colors,
                  cellLoc='left')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.5)
ax3.set_title('Examples of Context-Aware Recommendations Based on Uncertainty', fontsize=14, pad=20)

# Add main title
plt.suptitle("Uncertainty-Aware Recommendations in ExplainableCropAI", fontsize=16, y=0.98)

# Save the figure
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(os.path.join(output_dir, 'uncertainty_recommendations.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'uncertainty_recommendations.pdf'), format='pdf', bbox_inches='tight')

print(f"Uncertainty-based recommendations visualization saved to {output_dir}")

# Close the figure
plt.close()
