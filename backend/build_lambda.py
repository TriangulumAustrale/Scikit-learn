"""Build the Lambda deployment zip.

Downloads Linux wheels (so this works from Windows without Docker), strips
what Lambda does not need, and zips the result. Run from backend/:

    python build_lambda.py
"""

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
BUILD_DIR = BACKEND / "build" / "lambda"
DIST = BACKEND / "dist" / "lambda.zip"

# Lambda's Python 3.13 runtime is Amazon Linux 2023. manylinux_2_28 matters:
# the older manylinux2014 tag has no wheel for scikit-learn 1.9.0, and pip
# would silently resolve to 1.7.2, which cannot load this model artifact.
# Two tags, because pip matches them literally: scikit-learn 1.9.0 ships only
# manylinux_2_28 wheels, while pydantic-core ships only manylinux_2_17 ones.
# Amazon Linux 2023 runs both.
PLATFORMS = ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64"]
PYTHON_VERSION = "3.13"

LIMIT_MB = 250


def megabytes(path):
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def install_dependencies():
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--target", str(BUILD_DIR),
            *[arg for tag in PLATFORMS for arg in ("--platform", tag)],
            "--python-version", PYTHON_VERSION,
            "--only-binary=:all:",
            "--no-compile",
            "-r", str(BACKEND / "requirements-lambda.txt"),
        ],
        check=True,
    )


def strip_unneeded():
    """Remove test suites, bytecode and packaging metadata."""
    removed = 0.0
    for directory in sorted(BUILD_DIR.rglob("*"), reverse=True):
        if not directory.is_dir():
            continue
        if directory.name in {"tests", "test", "__pycache__"} or directory.name.endswith(
            (".dist-info", ".egg-info")
        ):
            removed += megabytes(directory)
            shutil.rmtree(directory, ignore_errors=True)
    for pyc in BUILD_DIR.rglob("*.pyc"):
        removed += pyc.stat().st_size / 1e6
        pyc.unlink(missing_ok=True)
    return removed


def main():
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True)

    print(f"Installing {'/'.join(PLATFORMS)} wheels for Python {PYTHON_VERSION}...")
    install_dependencies()
    print(f"  dependencies:  {megabytes(BUILD_DIR):6.1f} MB")

    print("Stripping tests, bytecode and metadata...")
    print(f"  removed:       {strip_unneeded():6.1f} MB")

    shutil.copytree(BACKEND / "app", BUILD_DIR / "app")
    shutil.copytree(BACKEND / "model", BUILD_DIR / "model")
    shutil.rmtree(BUILD_DIR / "app" / "__pycache__", ignore_errors=True)

    unzipped = megabytes(BUILD_DIR)
    print(f"  with app+model:{unzipped:6.1f} MB unzipped  (Lambda limit {LIMIT_MB} MB)")

    DIST.parent.mkdir(parents=True, exist_ok=True)
    DIST.unlink(missing_ok=True)
    print("Zipping...")
    with zipfile.ZipFile(DIST, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in BUILD_DIR.rglob("*"):
            if item.is_file():
                archive.write(item, item.relative_to(BUILD_DIR))

    print(f"\n  {DIST.relative_to(BACKEND)}: {megabytes(DIST):.1f} MB zipped, "
          f"{unzipped:.1f} MB unzipped")
    if unzipped > LIMIT_MB:
        print(f"  OVER the {LIMIT_MB} MB limit by {unzipped - LIMIT_MB:.1f} MB")
        return 1
    print(f"  {LIMIT_MB - unzipped:.1f} MB under the limit. Handler: app.main.handler")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
