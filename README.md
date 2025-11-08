# estimator

Time series GAM (Generalized Additive Model) estimators for load forecasting.

## Installation

```bash
uv sync
```

## Documentation

### Building Documentation Locally

To build the documentation locally:

1. Install documentation dependencies:
   ```bash
   uv sync --group docs
   ```

2. Generate documentation:
   ```bash
   python generate_docs.py
   ```

   Or to open in browser after building:
   ```bash
   python generate_docs.py --open
   ```

3. View the documentation:
   Open `docs/_build/html/index.html` in your browser.

### Alternative: Using Make

You can also use the Makefile in the `docs` directory:

```bash
cd docs
make html
```

## Development

### Running Tests

```bash
uv run pytest
```

### Running Tests with Coverage

```bash
uv run pytest --cov=tsgam_estimator --cov=load_model_estimator --cov-report=html
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.

## Contributors

See [CONTRIBUTORS](CONTRIBUTORS) for a list of contributors.

