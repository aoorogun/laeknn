"""Setup script for laeknn.

This script falls back to the classic setuptools build system for
environments that do not yet support PEP 517/pyproject builds.  It
mirrors the metadata declared in ``pyproject.toml``.
"""

from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="laeknn",
    version="0.1.0",
    author="Okunola Orogun",
    author_email="info@endowtech.com",
    description="Locally Adaptive Evidential K‑Nearest Neighbours (LAE‑KNN) algorithm",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://endowtech.example.com/laeknn",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    include_package_data=True,
)
