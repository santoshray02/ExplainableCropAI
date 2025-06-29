import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.dml import MSO_THEME_COLOR

def create_modern_presentation():
    # Create presentation
    prs = Presentation()
    
    # Define color scheme (Modern Agricultural Tech Theme)
    primary_color = RGBColor(30, 86, 49)      # Deep Green #1E5631
    secondary_color = RGBColor(25, 118, 210)   # Vibrant Blue #1976D2
    accent_color = RGBColor(255, 179, 0)       # Amber #FFB300
    text_color = RGBColor(33, 33, 33)          # Dark Gray #212121
    light_bg = RGBColor(255, 251, 240)        # Light Cream #FFFBF0
    
    # Helper function to add image if it exists
    def add_image_if_exists(slide, image_path, left, top, width, height):
        if os.path.exists(image_path):
            try:
                slide.shapes.add_picture(image_path, left, top, width, height)
                return True
            except Exception as e:
                print(f"Warning: Could not add image {image_path}: {e}")
                return False
        else:
            print(f"Warning: Image not found: {image_path}")
            return False
    
    # Slide 1: Title Slide
    slide_layout = prs.slide_layouts[0]  # Title slide layout
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    subtitle = slide.placeholders[1]
    
    title.text = "ExplainableCropAI"
    title_frame = title.text_frame
    title_para = title_frame.paragraphs[0]
    title_para.font.size = Pt(44)
    title_para.font.bold = True
    title_para.font.color.rgb = primary_color
    
    subtitle.text = "Interpretable Deep Learning for Agricultural Disease Detection and Decision Support\n\nSantosh Kumar, Durgesh Nandan, Sumit Kumar\nSymbiosis Institute of Technology\nSymbiosis International University\n\nMay 2025"
    subtitle_frame = subtitle.text_frame
    for para in subtitle_frame.paragraphs:
        para.font.size = Pt(18)
        para.font.color.rgb = text_color
        para.alignment = PP_ALIGN.CENTER
    
    # Add a decorative element or logo placeholder
    # Create a simple shape as visual element
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(8.5), Inches(1), Inches(1.5), Inches(0.5)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = accent_color
    shape.line.color.rgb = primary_color
    
    # Slide 2: Problem Statement with Visual
    slide_layout = prs.slide_layouts[5]  # Blank layout for custom arrangement
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.add_paragraph()
    title_para.text = "The Problem: Black-Box AI in Agriculture"
    title_para.font.color.rgb = primary_color
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    title_para.alignment = PP_ALIGN.LEFT
    
    # Add content in left column
    content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(4.5), Inches(5))
    content_frame = content_box.text_frame
    content_frame.text = """• Challenge: AI systems function as "black boxes"
    - Farmers don't trust what they can't understand
    - Limited adoption despite high accuracy

• Impact:
    - Significant crop losses due to paddy diseases
    - Underutilization of advanced AI technologies
    - Gap between AI capabilities and application

• Key Question:
    How can we make AI disease detection transparent and trustworthy for farmers?"""
    
    for para in content_frame.paragraphs:
        para.font.size = Pt(14)
        para.font.color.rgb = text_color
        para.space_after = Pt(6)
    
    # Add visual element (system architecture or explanation flowchart)
    add_image_if_exists(slide, "results/figures/explanation_flowchart.png", 
                       Inches(5.2), Inches(1.5), Inches(4.3), Inches(4.5))
    
    # Slide 3: Our Solution - ExplainableCropAI with Architecture Diagram
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.add_paragraph()
    title_para.text = "Our Solution: ExplainableCropAI Framework"
    title_para.font.color.rgb = primary_color
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    # Add system architecture diagram
    if add_image_if_exists(slide, "results/figures/system_architecture.png", 
                          Inches(0.5), Inches(1.5), Inches(9), Inches(3.5)):
        # Add brief description below the diagram
        desc_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.2), Inches(9), Inches(1.5))
        desc_frame = desc_box.text_frame
        desc_frame.text = """Key Innovations: Multi-level explanations • Domain-specific knowledge • User-centered design • Transparency without performance compromise"""
        for para in desc_frame.paragraphs:
            para.font.size = Pt(18)
            para.font.color.rgb = text_color
            para.alignment = PP_ALIGN.CENTER
    else:
        # Fallback to text-only if image not available
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
        content_frame = content_box.text_frame
        content_frame.text = """Key Innovations:

• Multi-level Explanation System
    - Visual heatmaps highlighting disease regions
    - Natural language descriptions
    - Uncertainty quantification

• Domain-Specific Knowledge Integration
    - Agricultural expertise embedded in explanations
    - Context-aware treatment recommendations

• User-Centered Design
    - Tailored for different expertise levels
    - Practical agricultural applications

• Transparency Without Compromise
    - 90% disease classification accuracy"""
        
        for para in content_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.space_after = Pt(8)
    
    # Slide 4: Visual Explanations - Grad-CAM Examples
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.add_paragraph()
    title_para.text = "Visual Explanations: Grad-CAM Disease Detection"
    title_para.font.color.rgb = primary_color
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    # Add Grad-CAM examples
    if add_image_if_exists(slide, "results/figures/gradcam_examples.png", 
                          Inches(0.5), Inches(1.5), Inches(9), Inches(4.5)):
        # Add description
        desc_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(9), Inches(1))
        desc_frame = desc_box.text_frame
        desc_frame.text = "Grad-CAM heatmaps highlight disease-affected regions, providing visual evidence for AI predictions"
        for para in desc_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.alignment = PP_ALIGN.CENTER
    else:
        # Fallback content
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
        content_frame = content_box.text_frame
        content_frame.text = """Visual Explanation Features:

• Grad-CAM Heatmaps
    - Highlight disease-affected regions
    - Provide visual evidence for predictions
    - No model retraining required

• Disease Pattern Recognition
    - Identify specific visual symptoms
    - Localize affected areas precisely

• Multi-scale Analysis
    - Global and local feature importance
    - Comprehensive visual understanding"""
        
        for para in content_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.space_after = Pt(8)
    
    # Slide 5: Uncertainty Visualization
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.add_paragraph()
    title_para.text = "Uncertainty Quantification & Recommendations"
    title_para.font.color.rgb = primary_color
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    # Add uncertainty visualization
    if add_image_if_exists(slide, "results/figures/uncertainty_visualization.png", 
                          Inches(0.5), Inches(1.5), Inches(4.5), Inches(3.5)):
        # Add recommendations visualization
        add_image_if_exists(slide, "results/figures/uncertainty_recommendations.png", 
                           Inches(5.2), Inches(1.5), Inches(4.3), Inches(3.5))
        
        # Add description
        desc_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.2), Inches(9), Inches(1.5))
        desc_frame = desc_box.text_frame
        desc_frame.text = """Monte Carlo Dropout provides confidence estimates • Uncertainty-aware treatment recommendations • Appropriate trust calibration"""
        for para in desc_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.alignment = PP_ALIGN.CENTER
    else:
        # Fallback content
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
        content_frame = content_box.text_frame
        content_frame.text = """Uncertainty Quantification Features:

• Monte Carlo Dropout
    - Provides confidence estimates
    - Quantifies prediction uncertainty
    - Supports appropriate trust calibration

• Treatment Recommendations
    - Uncertainty-aware suggestions
    - Severity-dependent protocols
    - Context-specific guidance

• Decision Support
    - When to seek expert consultation
    - Confidence thresholds for action
    - Risk assessment integration"""
        
        for para in content_frame.paragraphs:
            para.font.size = Pt(18)
            para.font.color.rgb = text_color
            para.space_after = Pt(10)
    
    # Slide 6: Technical Results with Performance Charts
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.add_paragraph()
    title_para.text = "Technical Performance Results"
    title_para.font.color.rgb = primary_color
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    # Add confusion matrix
    if add_image_if_exists(slide, "results/figures/confusion_matrix.png", 
                          Inches(0.5), Inches(1.5), Inches(4.5), Inches(3.5)):
        # Add performance metrics text
        metrics_box = slide.shapes.add_textbox(Inches(5.2), Inches(1.5), Inches(4.3), Inches(3.5))
        metrics_frame = metrics_box.text_frame
        metrics_frame.text = """Performance Metrics:

• Accuracy: 90.0%
• Precision: 89.5%
• Recall: 90.2%
• F1-Score: 89.8%

Explanation Quality:
• High localization accuracy
• Effective pattern highlighting
• No retraining required

Uncertainty Calibration:
• Well-calibrated confidence
• Effective OOD detection"""
        
        for para in metrics_frame.paragraphs:
            para.font.size = Pt(14)
            para.font.color.rgb = text_color
            para.space_after = Pt(8)
        
        # Add description
        desc_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.2), Inches(9), Inches(1.5))
        desc_frame = desc_box.text_frame
        desc_frame.text = "Confusion matrix shows balanced performance across all disease classes with minimal misclassification"
        for para in desc_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.alignment = PP_ALIGN.CENTER
    else:
        # Fallback to text-only metrics
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
        content_frame = content_box.text_frame
        content_frame.text = """Disease Detection Performance:
• Accuracy: 90.0%
• Precision: 89.5%
• Recall: 90.2%
• F1-Score: 89.8%

Visual Explanation Quality:
• Grad-CAM achieves highest localization accuracy
• Effective highlighting of disease-specific patterns
• No retraining required for explanation generation

Uncertainty Quantification:
• Well-calibrated confidence estimates
• Effective out-of-distribution detection
• Supports appropriate trust calibration

Natural Language Explanations:
• High content coverage compared to generic templates
• Accessible reading level (8.2 grade level)
• Positive expert evaluation ratings"""
        
        for para in content_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.space_after = Pt(10)
    
    # Slide 7: User Study Results
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.add_paragraph()
    title_para.text = "User Evaluation Results"
    title_para.font.color.rgb = primary_color
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    # Add user study results chart
    if add_image_if_exists(slide, "results/figures/user_study_results.png", 
                          Inches(0.5), Inches(1.5), Inches(5), Inches(4)):
        # Add explanation effectiveness chart
        add_image_if_exists(slide, "results/figures/explanation_effectiveness.png", 
                           Inches(5.7), Inches(1.5), Inches(3.8), Inches(4))
        
        # Add description
        desc_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.7), Inches(9), Inches(1))
        desc_frame = desc_box.text_frame
        desc_frame.text = "User studies demonstrate improved decision-making accuracy and trust with explanation systems"
        for para in desc_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.alignment = PP_ALIGN.CENTER
    else:
        # Fallback content
        content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
        content_frame = content_box.text_frame
        content_frame.text = """Key Findings:

• Decision Support
    - Explanations improved decision-making accuracy
    - Particularly beneficial for users with less expertise
    - Visual explanations enhance understanding

• Trust Calibration
    - Increased trust with explanation systems
    - Visual explanations valued for verification
    - Appropriate trust formation

• Cognitive Load Reduction
    - Reduced mental effort for complex cases
    - Explanations simplify AI interpretation
    - Improved task completion efficiency

• Learning Value
    - Educational benefits beyond decision support
    - Understanding of disease characteristics
    - Treatment rationale comprehension"""
        
        for para in content_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.space_after = Pt(10)
    
    # Slide 8: User Interface Demo
    slide = prs.slides.add_slide(slide_layout)
    
    # Add title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(1))
    title_frame = title_box.text_frame
    title_para = title_frame.add_paragraph()
    title_para.text = "System Interface & User Experience"
    title_para.font.color.rgb = primary_color
    title_para.font.size = Pt(32)
    title_para.font.bold = True
    
    # Add user interface screenshot
    if add_image_if_exists(slide, "results/figures/user_interface.png", 
                          Inches(0.5), Inches(1.5), Inches(9), Inches(4.5)):
        # Add description
        desc_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(9), Inches(1))
        desc_frame = desc_box.text_frame
        desc_frame.text = "Intuitive interface integrating visual explanations, uncertainty estimates, and treatment recommendations"
        for para in desc_frame.paragraphs:
            para.font.size = Pt(16)
            para.font.color.rgb = text_color
            para.alignment = PP_ALIGN.CENTER
    else:
        # Check alternative location
        if add_image_if_exists(slide, "src/visualization/results/figures/user_interface.png", 
                              Inches(0.5), Inches(1.5), Inches(9), Inches(4.5)):
            desc_box = slide.shapes.add_textbox(Inches(0.5), Inches(6.2), Inches(9), Inches(1))
            desc_frame = desc_box.text_frame
            desc_frame.text = "Intuitive interface integrating visual explanations, uncertainty estimates, and treatment recommendations"
            for para in desc_frame.paragraphs:
                para.font.size = Pt(16)
                para.font.color.rgb = text_color
                para.alignment = PP_ALIGN.CENTER
        else:
            # Fallback content
            content_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(9), Inches(5))
            content_frame = content_box.text_frame
            content_frame.text = """User Interface Features:

• Integrated Dashboard
    - Disease detection results
    - Visual explanation heatmaps
    - Confidence indicators

• Treatment Recommendations
    - Context-aware suggestions
    - Severity-based protocols
    - Expert consultation guidance

• Educational Components
    - Disease information
    - Prevention strategies
    - Learning resources

• Accessibility Features
    - Multi-language support
    - Intuitive navigation
    - Mobile-friendly design"""
            
            for para in content_frame.paragraphs:
                para.font.size = Pt(18)
                para.font.color.rgb = text_color
                para.space_after = Pt(10)
    
    # Slide 9: Key Contributions
    slide_layout = prs.slide_layouts[1]  # Title and content layout
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "Key Contributions & Impact"
    title.text_frame.paragraphs[0].font.color.rgb = primary_color
    title.text_frame.paragraphs[0].font.size = Pt(32)
    title.text_frame.paragraphs[0].font.bold = True
    
    content = slide.placeholders[1]
    content.text = """Research Contributions:

• Novel Multi-level Explanation Framework
    - First comprehensive explainable AI system for paddy diseases
    - Integration of visual, textual, and uncertainty explanations

• Domain-Specific Knowledge Integration
    - Agricultural expertise embedded in AI explanations
    - Context-aware recommendation generation

• Empirical Validation
    - Demonstrated impact on user trust and decision-making
    - Competitive accuracy without sacrificing explainability

Broader Impact:

• Agricultural Sustainability
    - Earlier disease detection and targeted interventions
    - Reduced unnecessary pesticide application

• Responsible AI Deployment
    - Blueprint for explainable AI in agriculture
    - Bridging AI performance and human interpretability"""
    
    content_frame = content.text_frame
    for para in content_frame.paragraphs:
        para.font.size = Pt(16)
        para.font.color.rgb = text_color
        para.space_after = Pt(10)
    
    # Slide 10: Future Work & Limitations
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "Limitations & Future Directions"
    title.text_frame.paragraphs[0].font.color.rgb = primary_color
    title.text_frame.paragraphs[0].font.size = Pt(32)
    title.text_frame.paragraphs[0].font.bold = True
    
    content = slide.placeholders[1]
    content.text = """Current Limitations:

• Dataset geographic constraints
• Controlled evaluation environment
• Manual input of contextual factors
• Computational requirements for resource-constrained devices

Future Directions:

• Personalized and Interactive Explanations
    - Adaptive explanation generation
    - User feedback integration

• Extended Applications
    - Multiple crop types and diseases
    - Broader agricultural decision contexts

• Enhanced Deployment
    - Lighter-weight mobile implementations
    - Real-world field testing

• Long-term Studies
    - Learning effects over time
    - Adoption patterns analysis"""
    
    content_frame = content.text_frame
    for para in content_frame.paragraphs:
        para.font.size = Pt(16)
        para.font.color.rgb = text_color
        para.space_after = Pt(10)
    
    # Slide 11: Conclusion
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "Conclusion"
    title.text_frame.paragraphs[0].font.color.rgb = primary_color
    title.text_frame.paragraphs[0].font.size = Pt(32)
    title.text_frame.paragraphs[0].font.bold = True
    
    content = slide.placeholders[1]
    content.text = """Key Takeaways:

• ExplainableCropAI successfully bridges the gap between AI accuracy and interpretability

• Multi-level explanations provide comprehensive decision support for agricultural stakeholders

• 90% disease classification accuracy demonstrates that explainability doesn't compromise performance

• User evaluation confirms improved trust and decision-making with explanations

• Framework provides a foundation for responsible AI deployment in agriculture

Impact Statement:
"Transforming AI from a black-box prediction tool into an assistive technology that augments farmer decision-making"

This research contributes to the critical advancement of trustworthy AI systems in agriculture, supporting sustainable farming practices and food security."""
    
    content_frame = content.text_frame
    for para in content_frame.paragraphs:
        para.font.size = Pt(16)
        para.font.color.rgb = text_color
        para.space_after = Pt(10)
    
    # Slide 12: Acknowledgments & Contact
    slide = prs.slides.add_slide(slide_layout)
    
    title = slide.shapes.title
    title.text = "Acknowledgments & Contact"
    title.text_frame.paragraphs[0].font.color.rgb = primary_color
    title.text_frame.paragraphs[0].font.size = Pt(32)
    title.text_frame.paragraphs[0].font.bold = True
    
    content = slide.placeholders[1]
    content.text = """Acknowledgments:
• Symbiosis Institute of Technology for research support
• Faculty guidance and mentorship
• Academic research community at Symbiosis International University

Contact Information:
• Author: Santosh Kumar
• Email: santosh.kumar.mtech2022@sitpune.edu.in
• Institution: Symbiosis Institute of Technology
• GitHub: github.com/santoshray02/ExplainableCropAI

Research Context:
This research was conducted as part of an M.Tech program, representing an academic exploration of explainable AI techniques applied to agricultural disease detection.

Thank you for your attention!
Questions and Discussion Welcome"""
    
    content_frame = content.text_frame
    for para in content_frame.paragraphs:
        para.font.size = Pt(18)
        para.font.color.rgb = text_color
        para.space_after = Pt(12)
    
    # Save presentation
    output_path = "presentations/ExplainableCropAI_Modern_Presentation.pptx"
    prs.save(output_path)
    print(f"Modern presentation with visuals saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_modern_presentation()
