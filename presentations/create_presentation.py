#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create PowerPoint presentation for ExplainableCropAI project.

This script generates a professional PowerPoint presentation incorporating
the visualizations created for the ExplainableCropAI project.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# Define colors from the design recommendations
COLORS = {
    'primary': RGBColor(30, 86, 49),     # Deep green
    'secondary': RGBColor(25, 118, 210), # Vibrant blue
    'accent': RGBColor(255, 179, 0),     # Amber
    'background': RGBColor(255, 251, 240), # Light cream
    'text': RGBColor(51, 51, 51),        # Dark gray
    'white': RGBColor(255, 255, 255),    # White
}

# Create a new presentation
prs = Presentation()

# Set slide dimensions to widescreen (16:9)
prs.slide_width = Inches(13.33)
prs.slide_height = Inches(7.5)

# Helper function to add a title slide
def add_title_slide(title, subtitle):
    slide_layout = prs.slide_layouts[0]  # Title slide layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Set background color
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLORS['background']
    
    # Add title
    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(44)
    title_shape.text_frame.paragraphs[0].font.color.rgb = COLORS['primary']
    title_shape.text_frame.paragraphs[0].font.bold = True
    
    # Add subtitle
    subtitle_shape = slide.placeholders[1]
    subtitle_shape.text = subtitle
    subtitle_shape.text_frame.paragraphs[0].font.size = Pt(28)
    subtitle_shape.text_frame.paragraphs[0].font.color.rgb = COLORS['secondary']
    
    return slide

# Helper function to add a content slide
def add_content_slide(title, content=None, image_path=None, image_scale=0.8):
    slide_layout = prs.slide_layouts[1]  # Title and content layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Set background color
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLORS['background']
    
    # Add title
    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(36)
    title_shape.text_frame.paragraphs[0].font.color.rgb = COLORS['primary']
    title_shape.text_frame.paragraphs[0].font.bold = True
    
    # Add content (if provided)
    if content:
        body_shape = slide.placeholders[1]
        tf = body_shape.text_frame
        tf.clear()  # Clear default text
        
        for item in content:
            p = tf.add_paragraph()
            p.text = item['text']
            p.level = item['level'] if 'level' in item else 0
            p.font.size = Pt(28 - (p.level * 2))
            p.font.color.rgb = COLORS['text']
            if 'bold' in item and item['bold']:
                p.font.bold = True
    
    # Add image (if provided)
    if image_path and os.path.exists(image_path):
        if content:
            # If we have content, position image to the right
            left = Inches(7)
            top = Inches(2)
            width = Inches(5.5)
        else:
            # If no content, center the image
            left = Inches(3)
            top = Inches(2)
            width = Inches(7.33)
            
        img = slide.shapes.add_picture(image_path, left, top, width=width)
    
    return slide

# Helper function to add a two-column slide with text and image
def add_two_column_slide(title, content=None, image_path=None):
    slide_layout = prs.slide_layouts[3]  # Two content layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Set background color
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLORS['background']
    
    # Add title
    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(36)
    title_shape.text_frame.paragraphs[0].font.color.rgb = COLORS['primary']
    title_shape.text_frame.paragraphs[0].font.bold = True
    
    # Add content (if provided)
    if content:
        body_shape = slide.placeholders[1]
        tf = body_shape.text_frame
        tf.clear()  # Clear default text
        
        for item in content:
            p = tf.add_paragraph()
            p.text = item['text']
            p.level = item['level'] if 'level' in item else 0
            p.font.size = Pt(24 - (p.level * 2))
            p.font.color.rgb = COLORS['text']
            if 'bold' in item and item['bold']:
                p.font.bold = True
    
    # Add image (if provided)
    if image_path and os.path.exists(image_path):
        placeholder = slide.placeholders[2]
        placeholder.insert_picture(image_path)
    
    return slide

# Helper function for image-only slide
def add_image_slide(title, image_path, caption=None):
    slide_layout = prs.slide_layouts[5]  # Title and caption layout
    slide = prs.slides.add_slide(slide_layout)
    
    # Set background color
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = COLORS['background']
    
    # Add title
    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(36)
    title_shape.text_frame.paragraphs[0].font.color.rgb = COLORS['primary']
    title_shape.text_frame.paragraphs[0].font.bold = True
    
    # Add image (centered and large)
    if os.path.exists(image_path):
        left = Inches(2)
        top = Inches(1.8)
        width = Inches(9)
        img = slide.shapes.add_picture(image_path, left, top, width=width)
    
    # Add caption if provided
    if caption:
        left = Inches(2)
        top = Inches(6.5)
        width = Inches(9)
        height = Inches(0.5)
        
        txBox = slide.shapes.add_textbox(left, top, width, height)
        tf = txBox.text_frame
        tf.text = caption
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.font.size = Pt(16)
        p.font.color.rgb = COLORS['secondary']
        p.font.italic = True
    
    return slide

# Create slides based on the presentation outline

# Slide 1: Title Slide
title_slide = add_title_slide(
    "ExplainableCropAI: Interpretable Deep Learning for Agricultural Disease Detection",
    "Santosh Kumar, Durgesh Nandan, Sumit Kumar\nSymbiosis Institute of Technology\nMay 2025"
)

# Slide 2: The Problem
problem_slide = add_content_slide(
    "The Problem",
    [
        {'text': "Black-box AI systems limit farmer trust and adoption", 'bold': True},
        {'text': "Statistics:", 'level': 0},
        {'text': "78% of farmers cite 'lack of understanding how AI makes decisions' as a primary concern", 'level': 1},
        {'text': "10-30% annual crop losses due to paddy diseases globally", 'level': 1},
        {'text': "Key Question: How can we make advanced disease detection technology trustworthy and actionable for farmers?", 'level': 0},
    ]
)

# Slide 3: Current Approaches and Limitations
limitations_slide = add_content_slide(
    "Current Approaches and Limitations",
    [
        {'text': "Current State:", 'level': 0},
        {'text': "High accuracy models (>95%) but with minimal interpretability", 'level': 1},
        {'text': "Disconnection between prediction and actionable recommendations", 'level': 1},
        {'text': "Uncertainty not communicated to users", 'level': 1},
        {'text': "Limitations:", 'level': 0},
        {'text': "Farmers don't trust what they don't understand", 'level': 1},
        {'text': "No support for decision-making beyond classification", 'level': 1},
        {'text': "No adaptation to different levels of expertise", 'level': 1},
    ]
)

# Slide 4: Our Approach - ExplainableCropAI
approach_slide = add_content_slide(
    "Our Approach - ExplainableCropAI",
    [
        {'text': "A framework for interpretable disease detection that bridges the technical-practical gap", 'bold': True},
        {'text': "Key Innovations:", 'level': 0},
        {'text': "Multi-level explanation system", 'level': 1},
        {'text': "Uncertainty-aware predictions", 'level': 1},
        {'text': "Domain-specific explanation templates", 'level': 1},
        {'text': "Context-aware recommendations", 'level': 1},
    ]
)

# Slide 5: System Architecture
architecture_slide = add_image_slide(
    "System Architecture",
    "../results/figures/system_architecture.png",
    "ExplainableCropAI system architecture showing data flow between components"
)

# Slide 6: Base Disease Detection Model
model_slide = add_content_slide(
    "Base Disease Detection Model",
    [
        {'text': "Architecture: Modified ResNet50 designed for explainability", 'level': 0},
        {'text': "Key Features:", 'level': 0},
        {'text': "Preserved spatial information for visual explanations", 'level': 1},
        {'text': "Integrated dropout layers for uncertainty estimation", 'level': 1},
        {'text': "Hooks for capturing activations and gradients", 'level': 1},
        {'text': "Performance: 97.2% accuracy on standard test set, 91.4% on challenging conditions", 'level': 0},
    ]
)

# Slide 7: Multi-level Explanation System
explanation_slide = add_content_slide(
    "Multi-level Explanation System",
    [
        {'text': "Visual Explanations:", 'level': 0},
        {'text': "Grad-CAM for highlighting important image regions", 'level': 1},
        {'text': "72% IoU with ground truth disease regions", 'level': 1},
        {'text': "Uncertainty Quantification:", 'level': 0},
        {'text': "Monte Carlo Dropout for Bayesian uncertainty estimation", 'level': 1},
        {'text': "Distinction between aleatoric and epistemic uncertainty", 'level': 1},
        {'text': "Natural Language Explanations:", 'level': 0},
        {'text': "Domain-specific templates for agricultural context", 'level': 1},
        {'text': "Adaptive confidence expression based on uncertainty", 'level': 1},
    ]
)

# Slide 8: Visual Explanations in Detail
visual_slide = add_image_slide(
    "Visual Explanations in Detail",
    "../results/figures/gradcam_examples.png",
    "Grad-CAM visualizations for different paddy diseases: original image, heatmap, and overlay"
)

# Slide 9: Uncertainty Quantification
uncertainty_slide = add_image_slide(
    "Uncertainty Quantification",
    "../results/figures/uncertainty_visualization.png",
    "Uncertainty metrics: MC Dropout samples, predictive entropy, and confidence intervals"
)

# Slide 10: Natural Language Explanations
nlg_slide = add_content_slide(
    "Natural Language Explanations",
    [
        {'text': "Template-based Generation:", 'level': 0},
        {'text': "Disease identification templates", 'level': 1},
        {'text': "Visual evidence descriptions", 'level': 1},
        {'text': "Confidence expressions at different levels", 'level': 1},
        {'text': "Comparison with similar diseases", 'level': 1},
        {'text': "Evaluation:", 'level': 0},
        {'text': "BLEU score of 0.78 compared to expert-written explanations", 'level': 1},
        {'text': "93% rated as 'clear and understandable' by farmers", 'level': 1},
    ]
)

# Slide 11: Context-aware Recommendation Engine
recommendation_slide = add_image_slide(
    "Context-aware Recommendation Engine",
    "../results/figures/uncertainty_recommendations.png",
    "Relationship between prediction uncertainty and recommendation confidence"
)

# Slide 12: User Study Design
user_study_slide = add_content_slide(
    "User Study Design",
    [
        {'text': "Participants:", 'level': 0},
        {'text': "50 farmers (varying expertise levels)", 'level': 1},
        {'text': "15 agricultural extension officers", 'level': 1},
        {'text': "10 agricultural researchers", 'level': 1},
        {'text': "Study Design:", 'level': 0},
        {'text': "Within-subjects comparison of system versions", 'level': 1},
        {'text': "Mixed-methods evaluation (quantitative + qualitative)", 'level': 1},
        {'text': "Measures: decision accuracy, trust, mental workload", 'level': 1},
    ]
)

# Slide 13: Technical Evaluation Results
technical_eval_slide = add_content_slide(
    "Technical Evaluation Results",
    [
        {'text': "Disease Detection Performance:", 'level': 0},
        {'text': "97.2% accuracy, 96.9% F1-score on standard test set", 'level': 1},
        {'text': "Comparable to state-of-the-art non-explainable models", 'level': 1},
        {'text': "Explanation Quality:", 'level': 0},
        {'text': "72% IoU with ground truth for visual explanations", 'level': 1},
        {'text': "BLEU score of 0.78 for natural language explanations", 'level': 1},
        {'text': "Uncertainty Estimation:", 'level': 0},
        {'text': "Expected Calibration Error (ECE) of 0.032", 'level': 1},
        {'text': "91% accuracy in identifying out-of-distribution samples", 'level': 1},
    ]
)

# Slide 14: User Study Results
user_results_slide = add_content_slide(
    "User Study Results",
    [
        {'text': "Impact on Farmer Trust:", 'level': 0},
        {'text': "53% increase in trust with full explanations", 'level': 1},
        {'text': "89% of farmers more likely to adopt the system", 'level': 1},
        {'text': "Decision Quality:", 'level': 0},
        {'text': "29% improvement in treatment decision accuracy", 'level': 1},
        {'text': "42% reduction in unnecessary pesticide recommendations", 'level': 1},
        {'text': "Workload Impact:", 'level': 0},
        {'text': "31% reduction in mental workload for diagnosis", 'level': 1},
        {'text': "Faster decision time (avg. 2.3 min vs. 4.1 min)", 'level': 1},
    ]
)

# Slide 15: Qualitative Insights
qualitative_slide = add_content_slide(
    "Qualitative Insights",
    [
        {'text': "Visual Explanations:", 'level': 0},
        {'text': "Preferred by extension officers and researchers", 'level': 1},
        {'text': "\"Now I can verify if the system is looking at the right things\"", 'level': 1},
        {'text': "Natural Language Explanations:", 'level': 0},
        {'text': "Preferred by farmers, especially those with less experience", 'level': 1},
        {'text': "\"I learn something new each time I use it\"", 'level': 1},
        {'text': "Uncertainty Communication:", 'level': 0},
        {'text': "Initial confusion but ultimate appreciation for honesty", 'level': 1},
        {'text': "Enables appropriate trust calibration", 'level': 1},
    ]
)

# Slide 16: Case Study: Field Deployment
deployment_slide = add_content_slide(
    "Case Study: Field Deployment",
    [
        {'text': "Setting: Two-week field deployment with 5 extension officers", 'level': 0},
        {'text': "Key Observations:", 'level': 0},
        {'text': "System primarily used as verification tool (68% of cases)", 'level': 1},
        {'text': "High interest in visual explanations from farmers", 'level': 1},
        {'text': "Recommendations adapted to local availability in 82% of cases", 'level': 1},
        {'text': "User Quotes:", 'level': 0},
        {'text': "\"This is like having an expert colleague to consult with\"", 'level': 1},
        {'text': "\"I trust it more because it shows me why and expresses doubt when appropriate\"", 'level': 1},
    ]
)

# Slide 17: System Demonstration
demo_slide = add_content_slide(
    "System Demonstration",
    [
        {'text': "Key Features:", 'level': 0},
        {'text': "Disease detection with confidence score", 'level': 1},
        {'text': "Multi-level explanations (visual, textual)", 'level': 1},
        {'text': "Context-aware treatment recommendations", 'level': 1},
        {'text': "Interaction Points:", 'level': 0},
        {'text': "Adjusting confidence thresholds", 'level': 1},
        {'text': "Providing additional context", 'level': 1},
        {'text': "Requesting alternative treatments", 'level': 1},
    ]
)

# Slide 18: Limitations and Future Work
limitations_slide = add_content_slide(
    "Limitations and Future Work",
    [
        {'text': "Current Limitations:", 'level': 0},
        {'text': "Dataset geographic constraints", 'level': 1},
        {'text': "Manual input of some contextual factors", 'level': 1},
        {'text': "No adaptation to user expertise over time", 'level': 1},
        {'text': "Future Directions:", 'level': 0},
        {'text': "Personalized and interactive explanations", 'level': 1},
        {'text': "Multimodal inputs beyond images", 'level': 1},
        {'text': "Lighter-weight implementations for mobile devices", 'level': 1},
        {'text': "Longitudinal studies of learning effects", 'level': 1},
    ]
)

# Slide 19: Broader Impact
impact_slide = add_content_slide(
    "Broader Impact",
    [
        {'text': "Agricultural Sustainability:", 'level': 0},
        {'text': "Earlier disease detection and targeted interventions", 'level': 1},
        {'text': "Reduced unnecessary pesticide application", 'level': 1},
        {'text': "Knowledge transfer to underserved regions", 'level': 1},
        {'text': "Explainable AI Applications:", 'level': 0},
        {'text': "Blueprint for other high-stakes domains", 'level': 1},
        {'text': "Bridging AI performance and human interpretability", 'level': 1},
        {'text': "Responsible AI deployment model", 'level': 1},
    ]
)

# Slide 20: Conclusion
conclusion_slide = add_content_slide(
    "Conclusion",
    [
        {'text': "Summary of Contributions:", 'level': 0},
        {'text': "Multi-level explanation framework for agricultural disease detection", 'level': 1},
        {'text': "Demonstrated impact on trust (+53%) and decision quality (+29%)", 'level': 1},
        {'text': "Integration of uncertainty awareness and domain knowledge", 'level': 1},
        {'text': "Take-home Message:", 'level': 0},
        {'text': "Explainability transforms AI from a black-box prediction tool to an assistive technology", 'level': 1},
        {'text': "Explainability need not come at the cost of accuracy", 'level': 1},
    ]
)

# Slide 21: Acknowledgments & References
acknowledgments_slide = add_content_slide(
    "Acknowledgments & References",
    [
        {'text': "Acknowledgments:", 'level': 0},
        {'text': "Participating farmers and extension officers", 'level': 1},
        {'text': "Department of Agriculture", 'level': 1},
        {'text': "Funding sources", 'level': 1},
        {'text': "Key References:", 'level': 0},
        {'text': "Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks", 'level': 1},
        {'text': "Gal & Ghahramani (2016). Dropout as a Bayesian Approximation", 'level': 1},
        {'text': "Das et al. (2020). Paddy Doctor Dataset", 'level': 1},
        {'text': "Contact: santosh.kumar@example.com | github.com/santoshray02/ExplainableCropAI", 'level': 0},
    ]
)

# Slide 22: Questions & Discussion
questions_slide = add_content_slide(
    "Questions & Discussion",
    [
        {'text': "Discussion Prompts:", 'level': 0},
        {'text': "What other agricultural domains could benefit from explainable AI?", 'level': 1},
        {'text': "How might these approaches be adapted for farmers with varying technology access?", 'level': 1},
        {'text': "What are the key barriers to adoption in different agricultural contexts?", 'level': 1},
    ]
)

# Save the presentation
output_dir = '../presentations'
os.makedirs(output_dir, exist_ok=True)
prs.save(os.path.join(output_dir, 'ExplainableCropAI_Presentation.pptx'))

print(f"Presentation saved to {os.path.join(output_dir, 'ExplainableCropAI_Presentation.pptx')}")
