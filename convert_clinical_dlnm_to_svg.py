#!/usr/bin/env python3
"""
Convert clinical-only DLNM PNG plots to SVG format
"""

import base64
import os
from PIL import Image

def png_to_svg(png_file, svg_file):
    """Convert PNG to SVG by embedding"""
    try:
        with open(png_file, 'rb') as f:
            png_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Get image dimensions
        with Image.open(png_file) as img:
            width, height = img.size
        
        # Create SVG with embedded PNG
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
     width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <image x="0" y="0" width="{width}" height="{height}" 
         xlink:href="data:image/png;base64,{png_data}"/>
</svg>'''
        
        with open(svg_file, 'w') as f:
            f.write(svg_content)
        
        print(f"Converted {png_file} to {svg_file}")
        return True
        
    except Exception as e:
        print(f"Failed to convert {png_file}: {e}")
        return False

def main():
    """Convert all clinical DLNM PNG files to SVG"""
    png_files = [
        'enbel_dlnm_bp_3d_clinical.png',
        'enbel_dlnm_bp_overall_clinical.png', 
        'enbel_dlnm_bp_slices_clinical.png',
        'enbel_dlnm_glucose_3d_clinical.png',
        'enbel_dlnm_glucose_overall_clinical.png',
        'enbel_dlnm_glucose_slices_clinical.png'
    ]
    
    converted = 0
    for png_file in png_files:
        if os.path.exists(png_file):
            svg_file = png_file.replace('.png', '.svg')
            if png_to_svg(png_file, svg_file):
                converted += 1
        else:
            print(f"PNG file not found: {png_file}")
    
    print(f"\nConverted {converted}/{len(png_files)} clinical DLNM files to SVG")

if __name__ == "__main__":
    main()