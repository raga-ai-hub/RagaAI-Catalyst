# Pull Request Process

## Before Creating a PR

1. **Update Your Fork**
```bash
git remote add upstream https://github.com/raga-ai-hub/agentneo.git
git fetch upstream
git rebase upstream/main
```

2. **Run Tests**
```bash
pytest tests/
pytest --cov=agentneo tests/
```

3. **Check Code Style**
```bash
black .
isort .
flake8
```

## Creating a Pull Request

### PR Checklist
- [ ] Updated documentation
- [ ] Added/updated tests
- [ ] Ran code formatting tools
- [ ] Updated changelog
- [ ] Resolved merge conflicts
- [ ] Added type hints
- [ ] Included docstrings

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code cleanup

## Test Plan
Describe how you tested your changes

## Breaking Changes
List any breaking changes

## Additional Notes
Any extra information
```

## Review Process

### What We Look For
1. Code Quality
   - Clean and readable code
   - Proper error handling
   - Efficient implementations

2. Testing
   - Adequate test coverage
   - Edge case handling
   - Performance considerations

3. Documentation
   - Clear docstrings
   - Updated README/docs
   - Code comments where needed

### Review Timeline
- Initial review: 1-2 business days
- Follow-up reviews: 1 business day
- Final approval: 1-2 business days

## After Approval

1. **Squash and Merge**
```bash
git rebase -i main
git push -f origin feature-branch
```

2. **Update Local Repository**
```bash
git checkout main
git pull upstream main
```

3. **Clean Up**
```bash
git branch -d feature-branch
git push origin --delete feature-branch
```