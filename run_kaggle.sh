#!/bin/bash
# Kaggle setup script
# Run this inside your Kaggle notebook via: !bash run_kaggle.sh

echo "Setting up Route Resilience Kaggle Environment..."

# Install requirements
pip install -r requirements.txt

# Create output directories required by M1 -> M4 contracts
mkdir -p /kaggle/working/checkpoints
mkdir -p /kaggle/working/predictions
mkdir -p /kaggle/working/graphs
mkdir -p /kaggle/working/analysis

echo "Environment ready. Refer to README.md for CLI commands."
