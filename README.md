[![DOI](https://zenodo.org/badge/522167778.svg)](https://zenodo.org/badge/latestdoi/522167778)

# Welcome to SERPENS

![SERPENS Examples](docs/img/W69.png?raw=true)

_Simulating Ring Particles Emergent from Natural Satellites_ <br>
This project was developed as a 3D weighted test-particle Monte-Carlo simulation of evaporating exomoons and their signatures in exoplanetary spectra. 

## Installation

### 1. First Step
Navigate to a location where you want to install SERPENS.
Clone this directory from the official [GitHub repository](https://github.com/momzw/SERPENS):
```bash
git clone https://github.com/momzw/SERPENS.git
```
Afterwards it's advisable to create a virtual environment with `venv`, 
e.g., `python -m venv serpens_env && source serpens_env/bin/activate`.
The second command (after `&&`) activates the virtual environment. For Windows this command might differ.

After you cloned the repository, enter it and install via `pip install -e .` . This installs SERPENS 
into the folder you just navigated into. 


### 2. REBOUND and REBOUNDx C libraries for CERPENS
The C-version of SERPENS, called CERPENS, requires the C libraries of [rebound](https://github.com/hannorein/rebound) and [reboundx](https://github.com/dtamayo/reboundx) at specific versions. 
These must be built from source. The code expects them to be in the parent directory. 

For REBOUND:
```bash
git clone --depth 1 --branch 4.6.0 https://github.com/hannorein/rebound.git ../rebound && \
    cd ../rebound && make
```

For REBOUNDx:
```bash
git clone --depth 1 --branch 4.6.1 https://github.com/dtamayo/reboundx.git ../reboundx && \
    cd ../reboundx && make
```

### 3. Make CERPENS
After the installation of REBOUND and REBOUNDx, we can make the C-version of SERPENS. 
For that, make sure to be in the SERPENS directory (where the `Makefile` is located) and execute `make` in the shell. 


## Alternative Installation with Docker 
The installation steps are all included in the Dockerfile. 
To build the Docker image, navigate to the SERPENS directory and execute `docker build -t serpens .`. 
To run the Docker container, execute `docker run -it serpens bash`. 

If we aim to run Jupyter notebooks, we can use the Docker container as a Jupyter server. 
To do so, execute `docker run -it -p 8888:8888 serpens bash` and start the server inside the container with 
`jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root` . Then navigate to `localhost:8888` in a web browser (or click the link
in the console that was created after the server startup). Instead of 8888, you can also use other port numbers.
This is especially important if you already have other notebooks or other things running on port 8888. If asked for a token, 
you can find it in the link provided by the server in the console. Alternatively, pass token-related flags when 
starting up the Jupyter server. 


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
