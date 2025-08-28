# GeminiSDR Documentation

This directory contains the documentation for GeminiSDR, built using Sphinx.

## Structure

```
docs/
├── index.rst                 # Main documentation index
├── conf.py                   # Sphinx configuration
├── requirements.txt          # Documentation dependencies
├── Makefile                  # Build commands
├── _templates/               # Custom templates
│   ├── module.rst           # Module documentation template
│   ├── class.rst            # Class documentation template
│   ├── function.rst         # Function documentation template
│   ├── guide_template.rst   # User guide template
│   └── tutorial_template.rst # Tutorial template
├── guides/                   # User guides
│   ├── installation/        # Installation guides
│   ├── quickstart/          # Quick start guides
│   └── advanced/            # Advanced usage guides
├── api/                      # Auto-generated API docs
├── architecture/             # System architecture docs
├── development/              # Developer guides
└── examples/                 # Examples and tutorials
```

## Building Documentation

### Prerequisites

Install documentation dependencies:

```bash
pip install -r docs/requirements.txt
```

### Quick Build

Use the provided build script:

```bash
python scripts/build_docs.py
```

### Manual Build

1. Generate API documentation:
   ```bash
   cd docs
   make api
   ```

2. Build HTML documentation:
   ```bash
   make html
   ```

3. Check links:
   ```bash
   make linkcheck
   ```

### Build Options

The build script supports various options:

```bash
# Clean and rebuild everything
python scripts/build_docs.py --clean

# Generate API docs only
python scripts/build_docs.py --api

# Build HTML only
python scripts/build_docs.py --html

# Check links only
python scripts/build_docs.py --check-links
```

## Writing Documentation

### API Documentation

API documentation is automatically generated from docstrings. Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 0) -> bool:
    """Brief description of the function.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1.
        param2: Description of param2. Defaults to 0.
        
    Returns:
        Description of return value.
        
    Raises:
        ValueError: Description of when this is raised.
        
    Example:
        >>> result = example_function("test", 5)
        >>> print(result)
        True
    """
    return True
```

### User Guides

Use the guide template in `_templates/guide_template.rst` as a starting point for new guides.

### Tutorials

Use the tutorial template in `_templates/tutorial_template.rst` for step-by-step tutorials.

## Continuous Integration

Documentation is automatically built and deployed via GitHub Actions:

- **Pull Requests**: Documentation is built and tested
- **Main Branch**: Documentation is built and deployed to GitHub Pages

## Local Development

For live reloading during development:

```bash
pip install sphinx-autobuild
cd docs
make livehtml
```

This will start a local server with automatic rebuilding when files change.

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root directory and have installed the package in development mode:
   ```bash
   pip install -e .
   ```

2. **Missing Dependencies**: Install all documentation requirements:
   ```bash
   pip install -r docs/requirements.txt
   ```

3. **Build Failures**: Check the build output for specific errors. Common issues include:
   - Missing docstrings
   - Incorrect reStructuredText syntax
   - Broken cross-references

### Getting Help

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Review existing documentation files for examples
- Open an issue if you encounter problems