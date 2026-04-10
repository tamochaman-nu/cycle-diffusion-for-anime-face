# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Fix for NVIDIA GPG key error (common in older base images)
RUN rm -f /etc/apt/sources.list.d/cuda.list && \
    rm -f /etc/apt/sources.list.d/nvidia-ml.list

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only the dependency file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies via pip
RUN pip install --no-cache-dir -r requirements.txt

# Install CLIP from OpenAI
RUN pip install git+https://github.com/openai/CLIP.git

# Copy and install taming-transformers from the submodule
COPY taming-transformers /opt/taming-transformers
RUN pip install /opt/taming-transformers

# Note: The project code is mounted via docker-compose for development.
# If you want to build a standalone image, uncomment the following line:
# COPY . .

# Set user-friendly environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default command to bash
CMD ["bash"]
