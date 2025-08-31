#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

alias python=python3.12
alias pip=pip3.12

# Create virtual environment using python3.12
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip inside the venv
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Create necessary folders
mkdir -p vectorstore/faiss_index

# --- Ollama installation ---
brew install ollama

# --- Pull LLaMA 3.1 8B instruct model ---
# This will download the model for local usage
# ollama pull llama3.1:8b-instruct-q8_0

# --- Run the model locally ---
# ollama run llama3.1:8b-instruct-q8_0

echo "Setup complete. Activate venv using: source venv/bin/activate"
