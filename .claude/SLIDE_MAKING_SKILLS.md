# Scientific Slide Making Skills (SVG)

This worktree is dedicated to creating publication-quality SVG slides for scientific presentations and posters.

## Core Expertise

### 1. SVG Technical Mastery
- **Scalable Vector Graphics**: Resolution-independent, publication-quality output
- **Figma Compatibility**: Structure SVGs for import/export with Figma
- **Text as Paths vs Editable**: Balance between consistency and editability
- **Layer Organization**: Logical grouping for easy editing
- **Color Schemes**: Scientific palettes (colorblind-safe, print-ready)

### 2. Scientific Visualization Principles
- **Visual Hierarchy**: Guide viewer attention to key findings
- **Data-Ink Ratio**: Maximize information, minimize clutter
- **Typography**: Sans-serif fonts (Arial, Helvetica) at readable sizes (≥10pt)
- **Color Theory**: 3-5 color maximum, consistent meaning across slides
- **White Space**: Breathing room, not cluttered

### 3. Domain-Specific Elements
- **Climate Data**: Temperature maps, time series, anomaly plots
- **Geospatial Maps**: Ward boundaries, study locations, heatmaps
- **Statistical Plots**: SHAP, feature importance, correlation matrices
- **Model Results**: Performance metrics, comparison tables
- **Methodology Diagrams**: Pipeline flowcharts, framework schematics

## Standard Slide Specifications

### Dimensions & Layout
```python
# Standard slide dimensions
SLIDE_WIDTH = 1920   # 16:9 aspect ratio
SLIDE_HEIGHT = 1080
DPI = 150            # High-resolution for print

# Margins
MARGIN_TOP = 100
MARGIN_BOTTOM = 80
MARGIN_LEFT = 80
MARGIN_RIGHT = 80

# Grid system
COLS = 12            # 12-column grid
GUTTER = 20          # Space between columns
```

### Color Palette
```python
# Primary colors (climate-health theme)
COLOR_PRIMARY = '#2E7D32'      # Dark green (health)
COLOR_SECONDARY = '#D32F2F'    # Red (climate/heat)
COLOR_ACCENT = '#1976D2'       # Blue (water/cooling)
COLOR_NEUTRAL = '#424242'      # Dark gray (text)
COLOR_LIGHT = '#F5F5F5'        # Light gray (background)

# Data visualization (colorblind-safe)
COLOR_SCALE = [
    '#1b9e77',  # Teal
    '#d95f02',  # Orange
    '#7570b3',  # Purple
    '#e7298a',  # Pink
    '#66a61e',  # Green
    '#e6ab02',  # Yellow
    '#a6761d',  # Brown
]

# Heatmaps (temperature)
TEMP_CMAP = 'YlOrRd'  # Yellow-Orange-Red
```

### Typography
```python
# Font sizes
TITLE_SIZE = 48
SUBTITLE_SIZE = 32
HEADING_SIZE = 24
BODY_SIZE = 18
CAPTION_SIZE = 14

# Font families
FONT_TITLE = 'Arial Black'
FONT_BODY = 'Arial'
FONT_MONO = 'Courier New'  # For code/data
```

## Slide Creation Workflow

### 1. Planning
```python
# Define slide structure
slide_spec = {
    'title': 'Climate-Health Biomarker Sensitivity',
    'layout': 'two_column',  # or 'full_width', 'three_panel'
    'elements': [
        {'type': 'map', 'data': 'johannesburg_wards'},
        {'type': 'chart', 'data': 'shap_values'},
        {'type': 'table', 'data': 'performance_metrics'}
    ],
    'annotations': ['n=11,398', 'R²=0.93', 'p<0.001']
}
```

### 2. Data Preparation
```python
import pandas as pd
import geopandas as gpd

# Load and validate data
clinical_df = pd.read_csv('data/CLINICAL_DATASET_COMPLETE_CLIMATE.csv')
gcro_df = pd.read_csv('data/GCRO_SOCIOECONOMIC_CLIMATE_ENHANCED_LABELED.csv')
shapefile = gpd.read_file('data/shapefiles/wards.shp')

# Verify data integrity
assert clinical_df['CD4 cell count (cells/µL)'].notna().sum() > 1000
assert len(shapefile) == 135  # Johannesburg wards
```

### 3. SVG Generation
```python
import svgwrite
from svgwrite import cm, mm

# Create drawing canvas
dwg = svgwrite.Drawing(
    'slide_01.svg',
    size=(f'{SLIDE_WIDTH}px', f'{SLIDE_HEIGHT}px'),
    viewBox=f'0 0 {SLIDE_WIDTH} {SLIDE_HEIGHT}'
)

# Add background
dwg.add(dwg.rect(
    insert=(0, 0),
    size=('100%', '100%'),
    fill=COLOR_LIGHT
))

# Add title
title = dwg.add(dwg.g(id='title'))
title.add(dwg.text(
    'Climate-Health Biomarker Analysis',
    insert=(SLIDE_WIDTH/2, MARGIN_TOP),
    text_anchor='middle',
    font_size=TITLE_SIZE,
    font_family=FONT_TITLE,
    fill=COLOR_NEUTRAL
))

# Save
dwg.save()
```

### 4. Quality Checks
- ✅ All text is readable (min 10pt at final size)
- ✅ Color contrast meets WCAG AA standards (4.5:1 for body text)
- ✅ Data labels are accurate and match source
- ✅ SVG validates and opens in Figma/Inkscape
- ✅ File size <2MB (optimize paths, remove unused defs)

## Common Slide Types

### 1. Title Slide
```python
def create_title_slide(title, subtitle, authors, affiliation):
    dwg = svgwrite.Drawing('slide_title.svg', size=SLIDE_SIZE)

    # Centered layout
    y_pos = SLIDE_HEIGHT / 2 - 100

    # Title
    dwg.add(dwg.text(
        title,
        insert=(SLIDE_WIDTH/2, y_pos),
        text_anchor='middle',
        font_size=60,
        font_weight='bold'
    ))

    # Subtitle
    y_pos += 80
    dwg.add(dwg.text(
        subtitle,
        insert=(SLIDE_WIDTH/2, y_pos),
        text_anchor='middle',
        font_size=36
    ))

    # Authors
    y_pos += 120
    dwg.add(dwg.text(
        authors,
        insert=(SLIDE_WIDTH/2, y_pos),
        text_anchor='middle',
        font_size=24
    ))

    return dwg
```

### 2. Data Visualization Slide
```python
def create_viz_slide(title, viz_data, annotations):
    """
    Layout: Title at top, large visualization in center, annotations at bottom
    """
    dwg = svgwrite.Drawing('slide_viz.svg', size=SLIDE_SIZE)

    # Title bar
    dwg.add(dwg.rect(
        insert=(0, 0),
        size=(SLIDE_WIDTH, 120),
        fill=COLOR_PRIMARY
    ))
    dwg.add(dwg.text(
        title,
        insert=(MARGIN_LEFT, 70),
        font_size=TITLE_SIZE,
        fill='white'
    ))

    # Visualization area (leave margins)
    viz_group = dwg.add(dwg.g(
        id='visualization',
        transform=f'translate({MARGIN_LEFT}, 150)'
    ))

    # Embed matplotlib/seaborn figure or create native SVG
    # ... add visualization elements ...

    # Annotations footer
    annotation_y = SLIDE_HEIGHT - MARGIN_BOTTOM
    for i, annotation in enumerate(annotations):
        dwg.add(dwg.text(
            annotation,
            insert=(MARGIN_LEFT + i*300, annotation_y),
            font_size=CAPTION_SIZE,
            fill=COLOR_NEUTRAL
        ))

    return dwg
```

### 3. Map Slide
```python
import matplotlib.pyplot as plt
import geopandas as gpd
from matplotlib.patches import Rectangle

def create_map_slide(title, shapefile, data_col, cmap='YlOrRd'):
    """
    Geospatial map with data overlay
    """
    # Create matplotlib figure (convert to SVG)
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.patch.set_facecolor(COLOR_LIGHT)

    # Plot choropleth
    shapefile.plot(
        column=data_col,
        cmap=cmap,
        ax=ax,
        legend=True,
        edgecolor='black',
        linewidth=0.5
    )

    # Add title
    ax.set_title(title, fontsize=TITLE_SIZE, pad=20)
    ax.axis('off')

    # Add annotations
    ax.text(
        0.02, 0.02,
        f'n = {len(shapefile):,} wards',
        transform=ax.transAxes,
        fontsize=CAPTION_SIZE,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    # Save as SVG
    plt.tight_layout()
    plt.savefig('slide_map.svg', format='svg', bbox_inches='tight')
    plt.close()
```

### 4. Table/Metrics Slide
```python
def create_table_slide(title, df, highlight_rows=None):
    """
    Clean table with optional row highlighting
    """
    dwg = svgwrite.Drawing('slide_table.svg', size=SLIDE_SIZE)

    # Title
    dwg.add(dwg.text(
        title,
        insert=(MARGIN_LEFT, MARGIN_TOP),
        font_size=TITLE_SIZE
    ))

    # Table dimensions
    table_top = MARGIN_TOP + 100
    row_height = 50
    col_widths = [300, 150, 150, 150, 150]

    # Header row
    y = table_top
    for i, col in enumerate(df.columns):
        x = MARGIN_LEFT + sum(col_widths[:i])
        dwg.add(dwg.rect(
            insert=(x, y),
            size=(col_widths[i], row_height),
            fill=COLOR_PRIMARY,
            stroke='white',
            stroke_width=2
        ))
        dwg.add(dwg.text(
            col,
            insert=(x + col_widths[i]/2, y + row_height/2 + 5),
            text_anchor='middle',
            font_size=BODY_SIZE,
            font_weight='bold',
            fill='white'
        ))

    # Data rows
    y += row_height
    for idx, row in df.iterrows():
        # Highlight if specified
        fill_color = '#FFEB3B' if idx in (highlight_rows or []) else 'white'

        for i, val in enumerate(row):
            x = MARGIN_LEFT + sum(col_widths[:i])
            dwg.add(dwg.rect(
                insert=(x, y),
                size=(col_widths[i], row_height),
                fill=fill_color,
                stroke=COLOR_NEUTRAL,
                stroke_width=1
            ))
            dwg.add(dwg.text(
                str(val),
                insert=(x + col_widths[i]/2, y + row_height/2 + 5),
                text_anchor='middle',
                font_size=BODY_SIZE
            ))

        y += row_height

    return dwg
```

### 5. Flowchart/Diagram Slide
```python
def create_flowchart_slide(title, steps):
    """
    Pipeline or methodology flowchart
    """
    dwg = svgwrite.Drawing('slide_flowchart.svg', size=SLIDE_SIZE)

    # Title
    dwg.add(dwg.text(title, insert=(MARGIN_LEFT, MARGIN_TOP), font_size=TITLE_SIZE))

    # Flowchart layout
    step_width = 300
    step_height = 100
    spacing = 80

    x = MARGIN_LEFT
    y = MARGIN_TOP + 150

    for i, step in enumerate(steps):
        # Step box
        dwg.add(dwg.rect(
            insert=(x, y),
            size=(step_width, step_height),
            fill=COLOR_ACCENT,
            rx=10, ry=10  # Rounded corners
        ))

        # Step text
        dwg.add(dwg.text(
            step['label'],
            insert=(x + step_width/2, y + step_height/2),
            text_anchor='middle',
            font_size=BODY_SIZE,
            fill='white',
            font_weight='bold'
        ))

        # Arrow to next step (if not last)
        if i < len(steps) - 1:
            arrow_start_x = x + step_width
            arrow_end_x = x + step_width + spacing

            # Horizontal arrow
            dwg.add(dwg.line(
                start=(arrow_start_x, y + step_height/2),
                end=(arrow_end_x, y + step_height/2),
                stroke=COLOR_NEUTRAL,
                stroke_width=3
            ))

            # Arrowhead
            dwg.add(dwg.polygon(
                points=[
                    (arrow_end_x, y + step_height/2),
                    (arrow_end_x - 15, y + step_height/2 - 10),
                    (arrow_end_x - 15, y + step_height/2 + 10)
                ],
                fill=COLOR_NEUTRAL
            ))

        x += step_width + spacing

    return dwg
```

## Advanced Techniques

### 1. Embedding Matplotlib Figures
```python
import io
import base64
from matplotlib import pyplot as plt

def embed_matplotlib_in_svg(fig, dwg, x, y, width, height):
    """
    Convert matplotlib figure to base64 PNG and embed in SVG
    """
    # Save figure to bytes
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', transparent=True)
    buf.seek(0)

    # Encode as base64
    img_data = base64.b64encode(buf.read()).decode()

    # Embed in SVG
    image = dwg.image(
        href=f'data:image/png;base64,{img_data}',
        insert=(x, y),
        size=(width, height)
    )
    dwg.add(image)
```

### 2. SHAP Waterfall Plot (Native SVG)
```python
def create_shap_waterfall_svg(shap_values, feature_names, base_value):
    """
    Create SHAP waterfall plot entirely in SVG (no matplotlib)
    """
    dwg = svgwrite.Drawing('shap_waterfall.svg', size=(800, 600))

    # Sort by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_values))[::-1][:10]

    # Bar chart parameters
    bar_height = 40
    bar_spacing = 10
    y_start = 100

    # Base value line
    dwg.add(dwg.line(
        start=(400, y_start),
        end=(400, y_start + (bar_height + bar_spacing) * len(sorted_idx)),
        stroke='black',
        stroke_width=2,
        stroke_dasharray='5,5'
    ))

    # Draw bars
    y = y_start
    for idx in sorted_idx:
        shap_val = shap_values[idx]
        feature = feature_names[idx]

        # Bar color (positive = red, negative = blue)
        color = COLOR_SECONDARY if shap_val > 0 else COLOR_ACCENT

        # Bar width proportional to SHAP value
        bar_width = abs(shap_val) * 200

        # Draw bar
        dwg.add(dwg.rect(
            insert=(400, y),
            size=(bar_width, bar_height),
            fill=color,
            opacity=0.8
        ))

        # Feature label
        dwg.add(dwg.text(
            feature,
            insert=(390, y + bar_height/2),
            text_anchor='end',
            font_size=14
        ))

        # SHAP value label
        dwg.add(dwg.text(
            f'{shap_val:.3f}',
            insert=(410 + bar_width, y + bar_height/2),
            font_size=14
        ))

        y += bar_height + bar_spacing

    return dwg
```

### 3. Interactive Elements (for HTML export)
```python
def add_tooltip(dwg, element, tooltip_text):
    """
    Add SVG tooltip (works in browsers)
    """
    title = dwg.title()
    title.text = tooltip_text
    element.add(title)
```

### 4. Animation (for digital presentations)
```python
def add_fade_in_animation(element, delay=0, duration=1):
    """
    Add fade-in animation to SVG element
    """
    element.add(svgwrite.animate.Animate(
        attributeName='opacity',
        from_='0',
        to='1',
        begin=f'{delay}s',
        dur=f'{duration}s',
        fill='freeze'
    ))
```

## Integration with Python Data Analysis

### From SHAP to Slide
```python
import shap
import matplotlib.pyplot as plt

# Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# Create SHAP plot
fig, ax = plt.subplots(figsize=(12, 8))
shap.plots.beeswarm(shap_values, show=False)

# Convert to SVG slide
plt.savefig('temp_shap.svg', format='svg', bbox_inches='tight')

# Wrap in slide template
dwg = svgwrite.Drawing('slide_shap.svg', size=SLIDE_SIZE)
dwg.add(dwg.rect(insert=(0,0), size=('100%','100%'), fill=COLOR_LIGHT))
dwg.add(dwg.text(
    'SHAP Feature Importance',
    insert=(SLIDE_WIDTH/2, 80),
    text_anchor='middle',
    font_size=TITLE_SIZE
))

# Embed SHAP figure
with open('temp_shap.svg', 'r') as f:
    shap_svg = f.read()
    # ... embed in dwg ...

dwg.save()
```

### From GeoPandas to Map Slide
```python
import geopandas as gpd
import matplotlib.pyplot as plt

# Load shapefile
wards = gpd.read_file('data/shapefiles/wards_with_data.shp')

# Create map
fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
wards.plot(
    column='study_count',
    cmap='YlOrRd',
    legend=True,
    ax=ax,
    edgecolor='black',
    linewidth=0.5
)

# Style
ax.set_title('Johannesburg Study Distribution', fontsize=48, pad=20)
ax.axis('off')

# Add scale bar, north arrow, etc.
# ...

# Save as SVG slide
plt.savefig(
    'slide_05_study_distribution.svg',
    format='svg',
    bbox_inches='tight',
    dpi=150
)
```

## File Organization

```
presentation_slides_final/
├── main_presentation/           # Production slides
│   ├── slide_01_title.svg
│   ├── slide_02_climate_patterns.svg
│   ├── ...
│   └── slide_12_references.svg
├── archive_versions/            # Previous iterations
│   └── *.svg
├── scripts/                     # Generation scripts
│   ├── create_slide_02.py
│   ├── create_slide_03.py
│   └── ...
├── data/                        # Data for visualizations
│   ├── shapefiles/
│   ├── shap_values/
│   └── summary_statistics/
└── README.md                    # Slide deck overview
```

## Quality Assurance Checklist

### Before Finalizing Slides
- [ ] All data labels match source data exactly
- [ ] Statistical annotations are correct (n, R², p-values)
- [ ] Color palette is colorblind-safe
- [ ] Text is readable at projected size (≥10pt)
- [ ] SVG validates (no errors in Inkscape/Figma)
- [ ] File size optimized (<2MB per slide)
- [ ] Consistent styling across all slides
- [ ] Citations/references included where needed
- [ ] Version control (commit with descriptive message)

### Accessibility
- [ ] Color contrast ≥4.5:1 for text
- [ ] Text alternatives for data visualizations
- [ ] No color-only information encoding
- [ ] Patterns/textures for colorblind accessibility

## Common Pitfalls & Solutions

### Problem: Text not editable in Figma
**Solution**: Don't convert text to paths unless necessary. Use standard fonts.

### Problem: SVG file too large (>5MB)
**Solution**:
- Simplify paths (reduce points)
- Remove unused definitions
- Compress embedded images
- Use references for repeated elements

### Problem: Colors look different when printed
**Solution**: Use CMYK-safe colors, test print, avoid pure RGB colors

### Problem: Map projections distorted
**Solution**: Use appropriate CRS for South Africa (EPSG:32735 for UTM Zone 35S)

### Problem: Data doesn't align with visualization
**Solution**: Always verify data source, use assertions in code, manual QA

## Resources

### Tools
- **Python**: matplotlib, seaborn, geopandas, svgwrite
- **R**: ggplot2, sf, svglite
- **Editors**: Inkscape (free), Adobe Illustrator, Figma (web-based)
- **Validation**: W3C SVG Validator

### Color Palettes
- ColorBrewer (cartography): https://colorbrewer2.org/
- Viridis (perceptually uniform): matplotlib default
- Scientific color maps: https://www.fabiocrameri.ch/colourmaps/

### Best Practices
- Tufte, E. (2001). *The Visual Display of Quantitative Information*
- Few, S. (2012). *Show Me the Numbers*
- Wilke, C. (2019). *Fundamentals of Data Visualization*

## Git Workflow

```bash
# Start new slide work
git checkout -b feat/add-slide-13-future-work

# Create slide
python scripts/create_slide_13.py

# Review output
open presentation_slides_final/main_presentation/slide_13_future_work.svg

# Commit
git add .
git commit -m "feat: add slide 13 with future research directions"

# Merge when ready
git checkout main
git merge feat/add-slide-13-future-work
```

## Success Metrics

### Visual Quality
- Publication-ready (suitable for Nature, Science, Lancet)
- Figma-compatible (can be imported and edited)
- Print-ready (300 DPI equivalent sharpness)

### Scientific Accuracy
- All data traceable to source
- Statistical annotations correct
- Citations complete

### Usability
- Editable in standard tools (Figma, Inkscape)
- File sizes manageable (<2MB per slide)
- Consistent styling for professional appearance
