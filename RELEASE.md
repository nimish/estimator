# Release process

This project publishes Python distributions from GitHub Actions using PyPI
trusted publishing. The package version is declared in `pyproject.toml`; the
runtime `tsgam_estimator.__version__` is read from installed package metadata.

## One-time PyPI setup

Configure a trusted publisher for this repository in PyPI:

- Repository: `nimish/estimator`
- Workflow: `.github/workflows/release.yml`
- Environment: leave blank unless the workflow is later updated to use one

No long-lived PyPI API token is required.

## Prepare a release

1. Update `version` in `pyproject.toml`.
2. Install and test with the locked environment:

   ```bash
   uv sync --frozen --group dev
   uv run ty check
   uv run pytest
   ```

3. Build and check the distributions:

   ```bash
   uv build
   uv run python scripts/check_package_contents.py
   uv run --with twine twine check dist/*
   ```

4. Commit the version change.
5. Create and push a matching tag:

   ```bash
   git tag v0.1.0
   git push origin main v0.1.0
   ```

The release workflow publishes to PyPI only for tags matching `v*.*.*`.
