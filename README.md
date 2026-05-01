[![DOI](https://zenodo.org/badge/522167778.svg)](https://zenodo.org/badge/latestdoi/522167778)

# Welcome to SERPENS

![SERPENS Examples](docs/img/W69.png?raw=true)

_Simulating Ring Particles Emergent from Natural Satellites_ <br>
This project was developed as a 3D weighted test-particle Monte-Carlo simulation of evaporating exomoons and their signatures in exoplanetary spectra. 

## Installation
1. Clone this directory to your system using `git clone https://github.com/momzw/SERPENS.git` .<br>
   It's recommended that you first create a directory into which you wish to install SERPENS and navigate into it.
3. Create and activate a Python environment. If `venv` is available, you can run 
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   Otherwise, use your preferred method to create and activate a Python environment.
4. Navigate to the SERPENS project root and run `pip install -e .` . This will install all necessary dependencies.

## Repository Layout

- `src/`: main simulation, analysis, scheduling, and visualization code.
- `src/cerpens/`: C-accelerated simulation components.
- `resources/`: default simulation objects and input parameter templates.
- `notebooks/`: exploratory and workflow notebooks.
- `testing/`: ad-hoc test scripts and examples.
- `legacy/`: legacy code kept for reference.
- `simdata/`: generated simulation outputs (runtime artifacts).

## Notes on Generated Files

- Runtime output files are produced during simulations (for example in `simdata/` and `schedule_archive/`).
- Local IDE, cache, and temporary files are intentionally ignored via `.gitignore`.

## Wiki
Check out the [wiki](https://github.com/momzw/SERPENS/wiki) for more information on how to work with SERPENS. 
