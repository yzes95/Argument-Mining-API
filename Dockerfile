FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Avoid prompts during installations
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 and pip
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as the default python
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Set the working directory
WORKDIR /app

# We pre-create all necessary cache directories and set open permissions.
# This now includes the '/app/.cache/cupy' directory.
RUN mkdir -p /app/.cache/huggingface /app/.cache/cupy && \
    chmod -R 777 /app/.cache


# --- Tell Hugging Face libraries to use this new directory ---
ENV HF_HOME="/app/.cache/huggingface"
ENV CUPY_CACHE_DIR="/app/.cache/cupy"

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip3 install --no-cache-dir -r requirements.txt


# --- Pre-download and cache the models during the build ---
# 1. Download BERT
RUN python3 -c "from transformers import BertModel, BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')"

# 2. Download RoBERTa
RUN python3 -c "from transformers import RobertaModel, RobertaTokenizer; RobertaTokenizer.from_pretrained('roberta-base'); RobertaModel.from_pretrained('roberta-base')"


# Copy the application file
COPY . .

# Expose the port
EXPOSE 7860

# The command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]