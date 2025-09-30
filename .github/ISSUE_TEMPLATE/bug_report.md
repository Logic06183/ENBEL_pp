---
name: Bug Report
about: Create a report to help us improve the ENBEL analysis pipeline
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

## 🐛 Bug Description

A clear and concise description of what the bug is.

## 🔄 Steps to Reproduce

1. Set up environment with: `...`
2. Run command: `...`
3. With configuration: `...`
4. See error: `...`

## 🎯 Expected Behavior

A clear and concise description of what you expected to happen.

## 📊 Actual Behavior

A clear and concise description of what actually happened.

## 📸 Screenshots/Output

If applicable, add screenshots, error messages, or command output to help explain your problem.

```
Paste any error messages or relevant output here
```

## 🖥️ Environment Information

**System Information:**
- OS: [e.g. Ubuntu 20.04, macOS 12.0, Windows 10]
- Python version: [e.g. 3.10.5]
- ENBEL version: [e.g. 1.0.0]

**Package Versions:**
```bash
# Run: pip list | grep -E "(pandas|numpy|scikit-learn|xgboost|shap)"
# Paste output here
```

**Configuration:**
- Analysis mode: [simple/improved/state-of-the-art]
- Configuration file: [default.yaml/custom.yaml/etc.]
- Docker/Local installation: [Docker/Local]

## 📁 Data Information

**Dataset characteristics:**
- Number of samples: [e.g. 10,000]
- Number of features: [e.g. 150]
- Biomarkers analyzed: [e.g. systolic blood pressure, glucose]
- Missing data percentage: [e.g. ~5%]

**Privacy note:** Please do NOT include actual data files or sensitive information.

## 🔬 Scientific Context

**Analysis context:**
- Research question: [Brief description]
- Expected findings: [What were you trying to discover?]
- Analysis stage: [Data loading/preprocessing/modeling/interpretation]

## 🧪 Reproducibility

**Reproducibility checklist:**
- [ ] I can reproduce this bug consistently
- [ ] I have tried with a fresh environment
- [ ] I have checked that random seeds are set
- [ ] I have tried with different data subsets
- [ ] The issue occurs with provided example data

**Minimal reproducible example:**
```python
# Please provide a minimal code example that reproduces the issue
# Use synthetic data if possible
```

## 🔍 Additional Context

**Related issues:**
- Links to any related issues or discussions

**Potential solutions:**
- Any ideas about what might be causing this?
- Have you tried any workarounds?

**Impact assessment:**
- [ ] Blocks critical analysis
- [ ] Affects result accuracy
- [ ] Performance issue
- [ ] Documentation issue
- [ ] Minor inconvenience

## 📋 Checklist

Before submitting, please ensure:

- [ ] I have searched for similar issues
- [ ] I have checked the documentation
- [ ] I have provided all requested information
- [ ] I have tested with the latest version
- [ ] I have removed any sensitive data from examples

## 🏷️ Labels

Please add relevant labels:
- **Priority:** `high`, `medium`, `low`
- **Component:** `data-processing`, `modeling`, `visualization`, `documentation`
- **Environment:** `docker`, `local`, `cloud`