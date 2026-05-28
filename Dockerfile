FROM python:3.12-slim

# Build tools + git + LaTeX packages required for matplotlib usetex rendering
# (amssymb is in texlive-latex-extra; wasysym is in texlive-fonts-extra)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-latex-extra \
    texlive-fonts-extra \
    cm-super \
    dvipng \
    && rm -rf /var/lib/apt/lists/*

# Clone and build rebound and reboundx C libraries (headers needed to compile
# serpens_hotloop.c; shared libs needed at runtime via ctypes).
# Both must share a common parent (/deps) because reboundx's Makefile locates
# rebound via the relative path ../../rebound from within reboundx/src/.
RUN git clone --depth 1 --branch 4.6.0 https://github.com/hannorein/rebound.git /deps/rebound && \
    cd /deps/rebound && make

ENV REB_DIR=/deps/rebound

RUN git clone --depth 1 --branch 4.6.1 https://github.com/dtamayo/reboundx.git /deps/reboundx && \
    cd /deps/reboundx && make

WORKDIR /app

# Install Python dependencies from the cloned C libraries (builds from source)
# and remaining packages from requirements.txt

# 1. Build tools for '--no-build-isolation'
# 2. rebound
# 3. reboundx
RUN pip install --no-cache-dir setuptools wheel && \
    pip install --no-cache-dir --no-build-isolation -e /deps/rebound && \
    pip install --no-cache-dir --no-build-isolation -e /deps/reboundx

ENV LD_LIBRARY_PATH=/deps/rebound:/deps/reboundx:$LD_LIBRARY_PATH

# 4. Remaining requirements (isolated for better checkpointing)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Following for matplotlib backends, but needs additional tk fixing.
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-tk

# Copy project source
COPY . .

# Build the C hot-loop extension, pointing to the cloned source trees.
RUN cd src/cerpens && make REBOUND_DIR=/deps/rebound/src REBOUNDX_DIR=/deps/reboundx/src

# Make the src package importable from project root
ENV PYTHONPATH=/app

# Jupyter usage:
# Host machine: docker run -it -p 8888:8888 image:version bash
# Inside the Container : jupyter notebook --ip 0.0.0.0 --port 8888 --no-browser --allow-root
# Host machine access: localhost:8888