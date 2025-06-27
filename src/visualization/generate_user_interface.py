import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import os
import sys

def create_user_interface():
    """Create a realistic user interface mockup for ExplainableCropAI system"""
    
    # Create figure with specific size for paper
    fig = plt.figure(figsize=(14, 10))
    
    # Remove axes
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Define colors
    primary_color = '#2E7D32'  # Green for agricultural theme
    secondary_color = '#1976D2'  # Blue
    accent_color = '#FF6F00'  # Orange
    bg_color = '#F5F5F5'
    white = '#FFFFFF'
    text_dark = '#212121'
    text_light = '#757575'
    
    # Background
    bg = Rectangle((0, 0), 100, 100, facecolor=bg_color, edgecolor='none')
    ax.add_patch(bg)
    
    # Header
    header = Rectangle((0, 90), 100, 10, facecolor=primary_color, edgecolor='none')
    ax.add_patch(header)
    ax.text(5, 95, 'ExplainableCropAI', fontsize=20, color=white, weight='bold', va='center')
    ax.text(5, 92, 'Paddy Disease Detection & Decision Support System', fontsize=12, color=white, va='center')
    
    # Navigation tabs
    tab_y = 87
    tabs = ['Upload Image', 'Disease Detection', 'Explanations', 'Recommendations', 'History']
    tab_width = 15
    for i, tab in enumerate(tabs):
        x = 5 + i * (tab_width + 1)
        if i == 2:  # Active tab
            tab_rect = FancyBboxPatch((x, tab_y), tab_width, 3, 
                                     boxstyle="round,pad=0.1",
                                     facecolor=white, edgecolor='none')
            ax.add_patch(tab_rect)
            ax.text(x + tab_width/2, tab_y + 1.5, tab, fontsize=10, 
                   color=primary_color, weight='bold', ha='center', va='center')
        else:
            ax.text(x + tab_width/2, tab_y + 1.5, tab, fontsize=10, 
                   color=white, ha='center', va='center')
    
    # Main content area
    main_area = FancyBboxPatch((2, 2), 96, 83, 
                              boxstyle="round,pad=0.5",
                              facecolor=white, edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(main_area)
    
    # Left panel - Original Image and Prediction
    left_panel = FancyBboxPatch((5, 45), 28, 35, 
                               boxstyle="round,pad=0.3",
                               facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(left_panel)
    
    # Image placeholder
    img_rect = Rectangle((7, 55), 24, 18, facecolor='#E8F5E9', edgecolor=primary_color, linewidth=2)
    ax.add_patch(img_rect)
    ax.text(19, 64, 'Paddy Leaf Image', fontsize=10, color=text_light, ha='center', va='center')
    ax.text(19, 61, '(Bacterial Leaf Blight)', fontsize=9, color=text_dark, ha='center', va='center', style='italic')
    
    # Prediction result
    ax.text(7, 52, 'Disease Detected:', fontsize=11, color=text_dark, weight='bold')
    pred_box = FancyBboxPatch((7, 48), 24, 3, 
                             boxstyle="round,pad=0.2",
                             facecolor='#FFEBEE', edgecolor='#D32F2F', linewidth=1)
    ax.add_patch(pred_box)
    ax.text(19, 49.5, 'Bacterial Leaf Blight (97.2%)', fontsize=11, 
           color='#D32F2F', weight='bold', ha='center', va='center')
    
    # Middle panel - Visual Explanation
    middle_panel = FancyBboxPatch((36, 45), 28, 35, 
                                 boxstyle="round,pad=0.3",
                                 facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(middle_panel)
    
    ax.text(50, 77, 'Visual Explanation (Grad-CAM)', fontsize=11, 
           color=text_dark, weight='bold', ha='center')
    
    # Heatmap placeholder
    heatmap_rect = Rectangle((38, 55), 24, 18, facecolor='#FFF3E0', edgecolor=accent_color, linewidth=2)
    ax.add_patch(heatmap_rect)
    
    # Simulate heatmap overlay
    for i in range(3):
        for j in range(3):
            intensity = np.random.uniform(0.3, 0.9)
            heat_patch = Rectangle((40 + j*7, 58 + i*5), 6, 4, 
                                 facecolor=accent_color, alpha=intensity, edgecolor='none')
            ax.add_patch(heat_patch)
    
    ax.text(50, 64, 'Disease Regions', fontsize=10, color=text_dark, ha='center', va='center', weight='bold')
    
    # Confidence indicator
    ax.text(38, 52, 'Model Confidence:', fontsize=10, color=text_dark)
    conf_bar = Rectangle((38, 49), 20, 1.5, facecolor='#E0E0E0', edgecolor='none')
    ax.add_patch(conf_bar)
    conf_fill = Rectangle((38, 49), 19.4, 1.5, facecolor=primary_color, edgecolor='none')
    ax.add_patch(conf_fill)
    ax.text(59, 49.75, '97.2%', fontsize=9, color=text_dark, weight='bold', va='center')
    
    # Right panel - Natural Language Explanation
    right_panel = FancyBboxPatch((67, 45), 28, 35, 
                                boxstyle="round,pad=0.3",
                                facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(right_panel)
    
    ax.text(81, 77, 'Natural Language Explanation', fontsize=11, 
           color=text_dark, weight='bold', ha='center')
    
    # Explanation text box
    exp_box = FancyBboxPatch((69, 55), 24, 18, 
                            boxstyle="round,pad=0.3",
                            facecolor='#F3F4F6', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(exp_box)
    
    # Explanation text
    explanation_lines = [
        "This image shows a paddy plant",
        "affected by Bacterial Leaf Blight,",
        "as indicated by the water-soaked",
        "lesions with wavy margins visible",
        "on the leaf surface.",
        "",
        "The system is highly confident in",
        "this diagnosis (97.2% certainty).",
        "",
        "Key visual indicators:",
        "• Water-soaked lesions",
        "• Wavy, irregular margins",
        "• Yellowish-brown coloration"
    ]
    
    y_pos = 71
    for line in explanation_lines:
        if line.startswith("•"):
            ax.text(71, y_pos, line, fontsize=8.5, color=text_dark, va='top')
        elif line == "":
            y_pos -= 0.8
            continue
        elif "Key visual indicators:" in line:
            ax.text(70, y_pos, line, fontsize=9, color=text_dark, weight='bold', va='top')
        else:
            ax.text(70, y_pos, line, fontsize=9, color=text_dark, va='top')
        y_pos -= 1.3
    
    # Uncertainty indicator
    ax.text(69, 52, 'Uncertainty Level:', fontsize=10, color=text_dark)
    unc_box = FancyBboxPatch((69, 48.5), 10, 2, 
                            boxstyle="round,pad=0.2",
                            facecolor='#E8F5E9', edgecolor=primary_color, linewidth=1)
    ax.add_patch(unc_box)
    ax.text(74, 49.5, 'Low', fontsize=10, color=primary_color, weight='bold', ha='center', va='center')
    
    # Bottom panel - Treatment Recommendations
    bottom_panel = FancyBboxPatch((5, 5), 90, 37, 
                                 boxstyle="round,pad=0.3",
                                 facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1)
    ax.add_patch(bottom_panel)
    
    ax.text(50, 39, 'Context-Aware Treatment Recommendations', fontsize=12, 
           color=text_dark, weight='bold', ha='center')
    
    # Recommendation sections
    sections = [
        {
            'title': 'Immediate Treatment',
            'icon': '⚡',
            'content': [
                '• Apply Streptomycin sulfate',
                '  (500 ppm) as foliar spray',
                '• Remove infected leaves',
                '• Ensure proper drainage'
            ],
            'x': 8, 'color': '#D32F2F'
        },
        {
            'title': 'Prevention Measures',
            'icon': '🛡️',
            'content': [
                '• Use resistant varieties',
                '• Maintain field sanitation',
                '• Avoid excess nitrogen',
                '• Monitor weather conditions'
            ],
            'x': 38, 'color': secondary_color
        },
        {
            'title': 'Long-term Management',
            'icon': '📅',
            'content': [
                '• Crop rotation with non-hosts',
                '• Balanced fertilization',
                '• Regular field monitoring',
                '• Maintain plant spacing'
            ],
            'x': 68, 'color': primary_color
        }
    ]
    
    for section in sections:
        # Section box
        sec_box = FancyBboxPatch((section['x'], 8), 26, 28, 
                                boxstyle="round,pad=0.3",
                                facecolor=white, edgecolor=section['color'], linewidth=1.5)
        ax.add_patch(sec_box)
        
        # Section header
        header_box = FancyBboxPatch((section['x'], 32), 26, 4, 
                                   boxstyle="round,pad=0.2",
                                   facecolor=section['color'], alpha=0.1, edgecolor='none')
        ax.add_patch(header_box)
        
        ax.text(section['x'] + 13, 34, section['title'], fontsize=11, 
               color=section['color'], weight='bold', ha='center', va='center')
        
        # Content
        y_pos = 29
        for line in section['content']:
            ax.text(section['x'] + 2, y_pos, line, fontsize=9, color=text_dark, va='top')
            y_pos -= 2.5
    
    # Action buttons
    buttons = [
        {'text': 'Save Report', 'x': 70, 'color': primary_color},
        {'text': 'Get Expert Opinion', 'x': 82, 'color': secondary_color}
    ]
    
    for btn in buttons:
        btn_box = FancyBboxPatch((btn['x'], 2), 12, 3, 
                                boxstyle="round,pad=0.2",
                                facecolor=btn['color'], edgecolor='none')
        ax.add_patch(btn_box)
        ax.text(btn['x'] + 6, 3.5, btn['text'], fontsize=10, 
               color=white, weight='bold', ha='center', va='center')
    
    # Status bar
    status_bar = Rectangle((0, 0), 100, 2, facecolor='#37474F', edgecolor='none')
    ax.add_patch(status_bar)
    ax.text(2, 1, 'Ready', fontsize=9, color=white, va='center')
    ax.text(98, 1, 'Connected to Server | Location: Maharashtra, India', 
           fontsize=9, color=white, ha='right', va='center')
    
    # Save the figure
    output_dir = 'results/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as both PDF and PNG
    plt.savefig(os.path.join(output_dir, 'user_interface.pdf'), 
                bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    plt.savefig(os.path.join(output_dir, 'user_interface.png'), 
                bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
    
    plt.close()
    
    print("User interface mockup generated successfully!")
    print("Files saved:")
    print("- results/figures/user_interface.pdf")
    print("- results/figures/user_interface.png")

if __name__ == "__main__":
    try:
        create_user_interface()
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
