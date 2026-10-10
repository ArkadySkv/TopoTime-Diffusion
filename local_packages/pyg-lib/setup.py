# local_packages/pyg-lib/setup.py
"""Setup script for building the local pyg-lib package extension."""

from setuptools import find_packages, setup

setup(
    name="pyg-lib",
    version="0.9.0",
    packages=find_packages(),
    zip_safe=False,
)
