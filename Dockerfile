FROM continuumio/miniconda3

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    gcc \
    g++ \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy environment.yml first to leverage Docker cache
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "generative_prompt", "/bin/bash", "-c"]

# Install CLIP
RUN pip install git+https://github.com/openai/CLIP.git

# Install PyTorch (using version 1.11.0 with CUDA 11.3 which is a stable combination for this era of code)
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install taming-transformers
RUN git clone https://github.com/CompVis/taming-transformers.git /workspace/taming-transformers && \
    cd /workspace/taming-transformers && \
    pip install -e .

# Copy the rest of the code
COPY . /workspace/cycle-diffusion-for-anime-face
WORKDIR /workspace/cycle-diffusion-for-anime-face

# Entrypoint to always run in the conda environment
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "generative_prompt"]
CMD ["/bin/bash"]
