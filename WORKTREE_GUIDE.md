# Git Worktree Organization Guide

This repository uses Git worktrees to organize different aspects of the ENBEL climate-health project into focused workspaces. Each worktree has specialized skills documentation to guide development in that area.

## Active Worktrees

### 1. Main Repository (Current)
- **Path**: `/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp`
- **Branch**: `main`
- **Purpose**: Central integration point for all work
- **Contains**:
  - Complete codebase
  - Data processing pipelines
  - Analysis results
  - Documentation

### 2. Model Refinement & Optimization
- **Path**: `ENBEL_pp_model_refinement/`
- **Branch**: `feat/model-optimization`
- **Purpose**: ML model development and performance improvement
- **Skills**: `.claude/MODEL_OPTIMIZATION_SKILLS.md`
- **Focus Areas**:
  - Hyperparameter optimization (Optuna, GridSearch)
  - Feature engineering and selection
  - SHAP/XAI analysis and visualization
  - Model evaluation and comparison
  - Ensemble methods and stacking
  - Performance benchmarking

**When to use**: When working on improving model R², expanding feature sets, or generating SHAP visualizations.

### 3. Methodology Development
- **Path**: `ENBEL_pp_methodology/`
- **Branch**: `feat/methodology-development`
- **Purpose**: Statistical methods and causal inference
- **Skills**: `.claude/METHODOLOGY_SKILLS.md`
- **Focus Areas**:
  - Distributed Lag Non-Linear Models (DLNM)
  - Case-crossover design
  - Mixed effects models (hierarchical data)
  - Mediation analysis
  - Sensitivity analyses
  - Multiple testing correction
  - Power analysis

**When to use**: When implementing rigorous statistical frameworks, validating ML findings causally, or developing new methodological approaches.

### 4. Slide Making & Visualization
- **Path**: `ENBEL_pp_slide_maker/`
- **Branch**: `feat/presentation-slides`
- **Purpose**: Creating publication-quality SVG slides
- **Skills**: `.claude/SLIDE_MAKING_SKILLS.md`
- **Focus Areas**:
  - SVG technical mastery (Figma-compatible)
  - Scientific visualization principles
  - Geospatial maps (Johannesburg wards)
  - SHAP plots and statistical diagrams
  - Color theory and typography
  - 16:9 slide specifications

**When to use**: When creating presentation slides, conference posters, or publication figures.

### 5. Publication Manuscript
- **Path**: `../ENBEL_pp_manuscript/`
- **Branch**: `publication-manuscript`
- **Purpose**: LaTeX manuscript for journal submission
- **Skills**: `.claude/LATEX_SCIENTIFIC_WRITING_SKILLS.md`
- **Focus Areas**:
  - LaTeX technical mastery (document classes, BibTeX)
  - Scientific writing (IMRAD structure, active voice)
  - Journal-specific formatting (Lancet, Nature, EHP)
  - Mathematical typesetting
  - Figure and table management
  - Bibliography management

**When to use**: When drafting or revising the manuscript for publication.

## Workflow Guide

### Switching Between Worktrees

```bash
# List all worktrees
git worktree list

# Navigate to specific worktree
cd ENBEL_pp_model_refinement   # For model work
cd ENBEL_pp_methodology         # For statistical methods
cd ENBEL_pp_slide_maker         # For slides
cd ../ENBEL_pp_manuscript       # For manuscript

# Return to main
cd /Users/craig/Library/Mobile\ Documents/com~apple~CloudDocs/ENBEL_pp
```

### Typical Development Workflow

1. **Start in Main**: Review current state, understand what needs to be done
2. **Switch to Specialized Worktree**: Navigate to appropriate workspace
3. **Develop**: Make changes using the specialized skills documentation as reference
4. **Commit Locally**: Commit work in the specialized branch
5. **Merge to Main**: When work is complete and tested, merge back to main

### Merging Changes Back to Main

```bash
# From main branch
git merge feat/model-optimization      # Merge model work
git merge feat/methodology-development # Merge methodology work
git merge feat/presentation-slides     # Merge slide work
git merge publication-manuscript       # Merge manuscript updates
```

### When to Merge

- **Model Refinement**: After achieving performance improvements or completing SHAP analysis
- **Methodology**: After validating new statistical methods or completing DLNM analysis
- **Slides**: After completing a slide deck or major visualization updates
- **Manuscript**: After completing a manuscript section or major revision

## Skills Documentation

Each worktree contains a `.claude/` directory with specialized skills documentation:

### MODEL_OPTIMIZATION_SKILLS.md
Comprehensive guide to:
- Hyperparameter optimization frameworks
- Feature engineering best practices
- SHAP analysis workflows
- Model evaluation metrics
- Performance benchmarking
- Code organization for ML experiments

### METHODOLOGY_SKILLS.md
Complete reference for:
- DLNM implementation in R
- Case-crossover design
- Mixed effects models
- Mediation analysis
- Sensitivity analyses
- Multiple testing corrections
- Statistical best practices

### SLIDE_MAKING_SKILLS.md
Expert guide to:
- SVG creation with Python/matplotlib
- Scientific color palettes
- Typography and layout
- Geospatial visualizations
- SHAP plot styling
- Publication-quality standards

### LATEX_SCIENTIFIC_WRITING_SKILLS.md
Professional manual for:
- LaTeX document structure
- IMRAD section templates
- Mathematical typesetting
- Bibliography management (BibTeX)
- Figure and table formatting
- Journal-specific requirements
- Compilation workflow

## Best Practices

### 1. Stay Focused
Each worktree is designed for a specific type of work. When you switch contexts, switch worktrees.

### 2. Commit Often
Commit your work frequently within each specialized branch. This allows you to experiment without fear of losing work.

### 3. Test Before Merging
Always test your work within the specialized worktree before merging to main.

### 4. Document Changes
Use descriptive commit messages that explain what was accomplished and why.

### 5. Leverage Skills Documentation
The `.claude/` skills documents are comprehensive guides. Refer to them when you need:
- Code examples
- Best practices
- Workflow patterns
- Troubleshooting help

### 6. Keep Worktrees in Sync
Periodically merge `main` into specialized branches to keep them up-to-date:

```bash
# From specialized worktree
git merge main
```

## Project Structure Summary

```
ENBEL_pp/                                    # Main repository
├── ENBEL_pp_model_refinement/               # Model optimization workspace
│   └── .claude/MODEL_OPTIMIZATION_SKILLS.md
├── ENBEL_pp_methodology/                    # Statistical methods workspace
│   └── .claude/METHODOLOGY_SKILLS.md
├── ENBEL_pp_slide_maker/                    # Visualization workspace
│   └── .claude/SLIDE_MAKING_SKILLS.md
├── data/                                    # Shared data (symlinked)
├── scripts/                                 # Analysis scripts
├── results/                                 # Analysis outputs
└── CLAUDE.md                                # Main project guide

ENBEL_pp_manuscript/                         # Publication workspace
├── .claude/LATEX_SCIENTIFIC_WRITING_SKILLS.md
├── manuscript/
│   ├── manuscript.tex
│   ├── sections/
│   ├── figures/
│   └── references.bib
└── compiled/
    └── manuscript.pdf
```

## Quick Reference

### Need to...
- **Improve model performance?** → `cd ENBEL_pp_model_refinement`
- **Run DLNM validation?** → `cd ENBEL_pp_methodology`
- **Create presentation slides?** → `cd ENBEL_pp_slide_maker`
- **Write/revise manuscript?** → `cd ../ENBEL_pp_manuscript`
- **Integrate everything?** → Stay in main repo

### Git Commands Cheatsheet
```bash
# View all worktrees
git worktree list

# Create new worktree
git worktree add <path> -b <branch-name>

# Remove worktree
git worktree remove <path>

# Merge branch to main
git merge <branch-name>

# Keep specialized branch updated
cd <worktree-path>
git merge main
```

## Maintenance

### Regular Tasks
1. **Weekly**: Merge main into specialized branches to keep them current
2. **After major changes**: Merge specialized work back to main
3. **Before presentations**: Update slides worktree with latest results
4. **Before submission**: Update manuscript worktree with final results

### Cleanup
If a worktree is no longer needed:
```bash
git worktree remove ENBEL_pp_old_branch
git branch -d feat/old-branch
```

## Getting Help

Each skills document includes:
- Complete code examples
- Best practices
- Common pitfalls and solutions
- Resource links

Start by reading the relevant skills document in the `.claude/` directory of each worktree.

---

**Last Updated**: 2025-10-30
**Maintained By**: Claude Code + Craig Saunders
