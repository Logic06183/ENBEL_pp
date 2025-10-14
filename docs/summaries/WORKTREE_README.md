# Repository Cleanup and Optimization Worktree

**Branch**: `feat/repo-cleanup-optimization`
**Purpose**: Clean up and optimize the ENBEL repository structure

## Focus Areas

### 1. Repository Structure
- [ ] Organize analysis scripts into logical directories
- [ ] Move archived/old scripts to proper archive structure
- [ ] Create consistent naming conventions
- [ ] Clean up root directory clutter

### 2. Code Organization
- [ ] Consolidate duplicate scripts
- [ ] Remove deprecated code
- [ ] Organize presentation slides and outputs
- [ ] Structure data processing pipelines

### 3. Documentation
- [ ] Update README.md with current project state
- [ ] Document directory structure
- [ ] Create/update CONTRIBUTING.md
- [ ] Improve inline documentation

### 4. CI/CD & DevOps
- [ ] Review and optimize GitHub Actions workflows
- [ ] Update pre-commit hooks
- [ ] Improve test coverage and organization
- [ ] Streamline dependency management

### 5. Data Management
- [ ] Document data sources and versions
- [ ] Create data dictionary updates
- [ ] Organize output files
- [ ] Archive old analysis results

## Git Commands for This Worktree

```bash
# Work in this directory
cd "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp/ENBEL_pp_cleanup"

# Check status
git status

# Commit changes
git add .
git commit -m "refactor: reorganize repository structure"

# Push to remote
git push origin feat/repo-cleanup-optimization

# When done, return to main and remove worktree
cd "/Users/craig/Library/Mobile Documents/com~apple~CloudDocs/ENBEL_pp"
git worktree remove ENBEL_pp_cleanup
```

## Tips

- Make atomic commits for each cleanup task
- Test that scripts still run after reorganization
- Update import paths if moving Python modules
- Keep documentation in sync with changes
- Use the devops-organizer agent for complex reorganization tasks
