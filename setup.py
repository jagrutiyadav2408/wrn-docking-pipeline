"""Packaging for the generalized virtual-screening pipeline."""
from pathlib import Path

from setuptools import setup

_here = Path(__file__).parent
_readme = (_here / "README.md")
long_description = _readme.read_text(encoding="utf-8") if _readme.is_file() else ""

setup(
    name="docksuite",
    version="1.0.0",
    description="Configuration-driven, target-agnostic virtual-screening & benchmark pipeline "
                "(AutoDock-GPU backend, Gnina-ready).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Mack / alotsolutions",
    license="MIT",
    packages=["src"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<3",
        "pandas>=2.0,<3",
        "scikit-learn>=1.3",
        "openpyxl>=3.1",
        "matplotlib>=3.7",
        "tqdm>=4.65",
    ],
    extras_require={
        "chem": ["rdkit>=2023.9"],          # conda-forge or rdkit-pypi wheels
        "dev": ["pytest>=7.4", "pytest-mock>=3.11"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
