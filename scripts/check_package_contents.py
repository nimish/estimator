"""Validate that built distributions contain only releasable package files."""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
from pathlib import Path


ALLOWED_SDIST_ROOT_FILES = {
    "LICENSE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
}
ALLOWED_SDIST_PACKAGE_FILES = {
    "src/tsgam_estimator/__init__.py",
    "src/tsgam_estimator/_design.py",
    "src/tsgam_estimator/_estimator.py",
    "src/tsgam_estimator/_forecast.py",
    "src/tsgam_estimator/_forecast_plotting.py",
    "src/tsgam_estimator/_problem.py",
    "src/tsgam_estimator/_sklearn.py",
    "src/tsgam_estimator/py.typed",
    "src/tsgam_estimator/tsgam_estimator.py",
}
ALLOWED_WHEEL_PACKAGE_FILES = {
    "tsgam_estimator/__init__.py",
    "tsgam_estimator/_design.py",
    "tsgam_estimator/_estimator.py",
    "tsgam_estimator/_forecast.py",
    "tsgam_estimator/_forecast_plotting.py",
    "tsgam_estimator/_problem.py",
    "tsgam_estimator/_sklearn.py",
    "tsgam_estimator/py.typed",
    "tsgam_estimator/tsgam_estimator.py",
}


def _check_sdist(path: Path) -> list[str]:
    errors: list[str] = []
    with tarfile.open(path) as archive:
        names = [member.name for member in archive.getmembers() if member.isfile()]

    for name in names:
        parts = Path(name).parts
        if len(parts) < 2:
            continue
        relative = "/".join(parts[1:])
        if (
            relative in ALLOWED_SDIST_ROOT_FILES
            or relative in ALLOWED_SDIST_PACKAGE_FILES
        ):
            continue
        errors.append(f"{path.name}: unexpected sdist member {relative}")
    return errors


def _check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist() if not info.is_dir()]

    for name in names:
        if name in ALLOWED_WHEEL_PACKAGE_FILES:
            continue
        if name.startswith("tsgam_estimator-") and ".dist-info/" in name:
            continue
        errors.append(f"{path.name}: unexpected wheel member {name}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dist", nargs="?", default="dist", type=Path)
    args = parser.parse_args()

    sdists = sorted(args.dist.glob("*.tar.gz"))
    wheels = sorted(args.dist.glob("*.whl"))
    if not sdists or not wheels:
        print(
            f"expected at least one sdist and one wheel in {args.dist}", file=sys.stderr
        )
        return 1

    errors: list[str] = []
    for path in sdists:
        errors.extend(_check_sdist(path))
    for path in wheels:
        errors.extend(_check_wheel(path))

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("package contents are release-scoped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
