# cycle-diffusion-for-anime-face: full environment.yml (conda) environment, so
# main.py and everything under model/ (including the Stable Diffusion / Latent
# Diffusion code paths that depend on transformers/datasets/pytorch-lightning)
# stays runnable, PLUS a modern PyTorch build.
#
# torch/torchvision are intentionally installed via pip AFTER the conda env
# (environment.yml itself pins neither), and at version 2.1.2+cu121 rather than
# the CUDA-11.3-era 1.11.0 build this project historically used: that older
# wheel's libtorch_cpu.so requests an executable stack (GNU_STACK segment
# flagged RWE), and on this project's WSL2 host the kernel refuses the
# mprotect() needed to satisfy that at dynamic-link time --
#   ImportError: libtorch_cpu.so: cannot enable executable stack as shared
#   object requires: Invalid argument
# -- which prevents `import torch` from working at all. torch==2.1.2 (a much
# newer build) does not have this requirement (verified directly on this host).
# If you target a GPU whose compute capability isn't covered by the cu121
# wheels, or don't hit the executable-stack issue on your own host, adjust the
# --index-url/version here freely -- nothing else in this Dockerfile depends on
# this specific torch version.

FROM continuumio/miniconda3

WORKDIR /workspace/cycle-diffusion-for-anime-face

RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget gcc g++ libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy environment.yml first to leverage Docker's layer cache.
COPY environment.yml .
RUN conda env create -f environment.yml

# Make subsequent RUN commands use the new environment.
SHELL ["conda", "run", "-n", "generative_prompt", "/bin/bash", "-c"]

RUN pip install --no-cache-dir \
        torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

# Per this repo's own README.md ("Dependencies" section).
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git
RUN git clone https://github.com/CompVis/taming-transformers.git /opt/taming-transformers \
    && cd /opt/taming-transformers && pip install --no-cache-dir -e .

COPY . /workspace/cycle-diffusion-for-anime-face

# docker-compose.yml overrides `entrypoint` per-service (ffhq2anime/diagA/diagB);
# this default lets `docker compose run --rm <service> bash` or a raw
# `docker run` still drop into an interactive shell in the right conda env.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "generative_prompt"]
CMD ["/bin/bash"]
