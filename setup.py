from setuptools import setup, find_packages
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")


setup(
    name="serpens",
    version="0.3.1",
    author="Moritz Meyer zu Westram",
    description=(
        "SERPENS - Simulating the Evolution of Ring Particles "
        "Emergent from Natural Satellites"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/momzw/SERPENS",
    license="GPL-3.0",
    packages=find_packages(exclude=["legacy", "legacy.*", "testing", "testing.*"]),
    package_data={
        "src.cerpens": ["serpens_hotloop.c", "Makefile"],
        "resources": ["*"],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26",
        "numba>=0.59",
        "scipy>=1.13",
        "rebound>=4.6",
        "reboundx>=4.6",
        "matplotlib>=3.8",
        "pandas>=2.2",
        "plotly>=5.22",
        "tqdm>=4.66",
        "h5py>=3.13",
        "jupyter>=1.1",
        "notebook>=7.5",
        "ipykernel>=7.2"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
