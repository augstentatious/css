FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    texlive-latex-extra \
    texlive-fonts-recommended \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy src/data/scripts
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY data/README.md /app/data/

# Default cmd: eval loop
CMD ["python", "scripts/eval_advbench.sh"]

# Build: docker build -t css-safety .
# Run: docker run --gpus all -v $(pwd)/results:/app/results css-safety --seed 42
