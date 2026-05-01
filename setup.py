import subprocess
import sys

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
from pathlib import Path

long_description = Path("README.md").read_text(encoding="utf-8")


class BuildWithMake(build_py):
    """Run the root Makefile to compile the C hotloop before packaging."""

    def run(self):
        try:
            subprocess.check_call(
                ["make", "-C", str(Path(__file__).parent.resolve())],
            )
        except subprocess.CalledProcessError as exc:
            print(
                "WARNING: C hotloop compilation failed (exit code "
                f"{exc.returncode}). The pure-Python fallback will be used.",
                file=sys.stderr,
            )
        except FileNotFoundError:
            print(
                "WARNING: 'make' not found. Skipping C hotloop compilation. "
                "The pure-Python fallback will be used.",
                file=sys.stderr,
            )
        super().run()


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
    cmdclass={"build_py": BuildWithMake},
    package_data={
        "src.cerpens": ["serpens_hotloop.c", "serpens_hotloop.so",
                        "serpens_hotloop.dll", "Makefile"],
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
