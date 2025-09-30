#!/usr/bin/env python3
"""
Convert PNG DLNM plots to SVG format
"""

import base64
import subprocess
import os

def png_to_svg(png_file, svg_file):
    """Convert PNG to SVG using ImageMagick if available, otherwise embed PNG in SVG"""
    try:
        # Try ImageMagick convert
        result = subprocess.run(['convert', png_file, svg_file], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Converted {png_file} to {svg_file} using ImageMagick")
            return True
    except FileNotFoundError:
        pass
    
    # Fallback: embed PNG in SVG
    try:
        with open(png_file, 'rb') as f:
            png_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Get image dimensions (basic estimate)
        from PIL import Image
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
        
        print(f"Converted {png_file} to {svg_file} by embedding")
        return True
        
    except Exception as e:
        print(f"Failed to convert {png_file}: {e}")
        return False

def main():
    """Convert all DLNM PNG files to SVG"""
    png_files = [
        'enbel_dlnm_bp_3d.png',
        'enbel_dlnm_bp_overall.png', 
        'enbel_dlnm_bp_slices.png',
        'enbel_dlnm_glucose_3d.png',
        'enbel_dlnm_glucose_overall.png',
        'enbel_dlnm_glucose_slices.png'
    ]
    
    converted = 0
    for png_file in png_files:
        if os.path.exists(png_file):
            svg_file = png_file.replace('.png', '.svg')
            if png_to_svg(png_file, svg_file):
                converted += 1
        else:
            print(f"PNG file not found: {png_file}")
    
    print(f"\nConverted {converted}/{len(png_files)} files to SVG")

if __name__ == "__main__":
    main()