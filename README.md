# Time-series generalized additive model (tsgam)

Time series generalized additive model (tsgam) is a package for fitting "generalized addive models" (GAMs) augmented with time-dependent features. The idea is to fit a statistical model that estimates a target time-series based on linear or nonlinear responses to exogenous variables, features encoding one or more natural periodicies, and a features encoding long-term trends. For nonlinear exogenous variables, we model the response with natural cubic basis splines. The (multi-)periodic components are modeled with trucated Fourier series, plus cross terms when multiple periods are present (e.g., daily and year periodicities). We currently have long-term trend models for linear trends and monotonic nonlinear trends. Nonlinear trends can now be configured explicitly as `nonlinear_decreasing` or `nonlinear_increasing`, while legacy `nonlinear` remains a backward-compatible alias for the decreasing form.

## Installation

From PyPI:

```bash
uv add tsgam-estimator
```

For local development:

```bash
uv sync --group dev
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
uv sync --group test
uv run pytest
```

### Running Type Checks

```bash
uv sync --group typecheck
uv run ty check
```

### Running Tests with Coverage

```bash
uv run pytest --cov=tsgam_estimator --cov-report=html
```

### Working with Examples and Notebooks

```bash
uv sync --group examples
uv sync --group notebooks
```

## License

BSD 3-Clause License - see [LICENSE](LICENSE) for details.

## Contributors

See [CONTRIBUTORS](CONTRIBUTORS) for a list of contributors.
