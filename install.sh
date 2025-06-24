#!/bin/bash

# Exit on error
set -e

# Note: To make this script executable, run:
# chmod +x install.sh

echo "Creating a Python 3.12 virtual environment..."

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 is not installed. Please install Python 3.12 and try again."
    echo ""
    echo "Installation hints:"
    echo "- Ubuntu/Debian: sudo apt-get update && sudo apt-get install python3.12 python3.12-venv python3.12-dev"
    echo "- macOS: brew install python@3.12"
    echo "- Windows: Download from https://www.python.org/downloads/"
    echo "- Using pyenv: pyenv install 3.12"
    echo ""
    exit 1
fi

# Create a virtual environment
python3.12 -m venv venv

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# First install the cpu or cuda version of torch
echo "Installing PyTorch..."
# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA is available. Installing PyTorch with CUDA support..."
    pip install torch~=2.7.1 --index-url https://download.pytorch.org/whl/cu118
else
    echo "CUDA is not available. Installing CPU-only version of PyTorch..."
    pip install torch~=2.7.1 --index-url https://download.pytorch.org/whl/cpu
fi

# Install dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installation complete! To activate the virtual environment, run:"
echo "source venv/bin/activate"
