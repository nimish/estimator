#!/usr/bin/env python3
# Copyright (c) 2025 Alliance for Sustainable Energy, LLC and Nimish Telang
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to generate documentation using Sphinx.

Usage:
    python generate_docs.py
    python generate_docs.py --open  # Open docs in browser after building
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Generate documentation")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open documentation in browser after building",
    )
    parser.add_argument(
        "--format",
        choices=["html", "pdf", "epub"],
        default="html",
        help="Documentation format to generate (default: html)",
    )
    args = parser.parse_args()

    # Get the project root directory
    project_root = Path(__file__).parent
    docs_dir = project_root / "docs"

    if not docs_dir.exists():
        print(f"Error: docs directory not found at {docs_dir}")
        sys.exit(1)

    # Change to docs directory
    import os
    os.chdir(docs_dir)

    # Build documentation
    print("Building documentation...")
    try:
        if args.format == "html":
            subprocess.run(
                ["sphinx-build", "-b", "html", ".", "_build/html"],
                check=True,
            )
            print("\nDocumentation built successfully!")
            print(f"Open {docs_dir / '_build' / 'html' / 'index.html'} in your browser to view.")

            if args.open:
                import webbrowser
                html_path = docs_dir / "_build" / "html" / "index.html"
                webbrowser.open(f"file://{html_path.absolute()}")
        else:
            subprocess.run(
                ["sphinx-build", "-b", args.format, ".", f"_build/{args.format}"],
                check=True,
            )
            print(f"\nDocumentation built successfully in {args.format} format!")
            print(f"Output: {docs_dir / '_build' / args.format}")
    except subprocess.CalledProcessError as e:
        print(f"Error building documentation: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: sphinx-build not found. Make sure Sphinx is installed:")
        print("  uv sync --group docs")
        sys.exit(1)


if __name__ == "__main__":
    main()

