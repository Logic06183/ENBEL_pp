#!/usr/bin/env python3
"""
Convert the perfected population health impact slide to PNG format.
"""

import cairosvg
import os

def convert_svg_to_png():
    """Convert the SVG slide to high-quality PNG."""
    
    # File paths
    svg_file = "enbel_population_health_impact_perfected.svg"
    png_file = "enbel_population_health_impact_perfected.png"
    
    # Convert SVG to PNG with high DPI for presentation quality
    cairosvg.svg2png(
        url=svg_file,
        write_to=png_file,
        output_width=1920,  # High resolution for presentation
        output_height=1080,
        dpi=150  # High DPI for crisp text and graphics
    )
    
    print(f"Successfully converted {svg_file} to {png_file}")
    print(f"PNG dimensions: 1920x1080 pixels at 150 DPI")
    
    # Check file sizes
    svg_size = os.path.getsize(svg_file) / 1024  # KB
    png_size = os.path.getsize(png_file) / 1024  # KB
    
    print(f"SVG file size: {svg_size:.1f} KB")
    print(f"PNG file size: {png_size:.1f} KB")

if __name__ == "__main__":
    convert_svg_to_png()