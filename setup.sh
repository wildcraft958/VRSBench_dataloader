#!/bin/bash
# Setup script for VRSBench DataLoader

set -e

echo "======================================"
echo "VRSBench DataLoader Setup"
echo "======================================"

# Create directories
echo "Creating directories..."
mkdir -p data/images
mkdir -p data/annotations
mkdir -p logs
mkdir -p hf_cache

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Set HuggingFace token (optional)
echo ""
echo "Setting up HuggingFace authentication (optional)..."
echo "To avoid rate limits, get a token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Enter your HuggingFace token (or press Enter to skip): " hf_token

if [ ! -z "$hf_token" ]; then
    echo "export HUGGINGFACE_HUB_TOKEN=$hf_token" >> ~/.bashrc
    export HUGGINGFACE_HUB_TOKEN=$hf_token
    echo "✓ Token saved to ~/.bashrc"
else
    echo "Skipped. You can set it later with: export HUGGINGFACE_HUB_TOKEN=your_token"
fi

# Download sample data (optional)
echo ""
read -p "Download VRSBench validation data? (y/n): " download_data

if [ "$download_data" = "y" ]; then
    echo "Downloading validation annotations..."
    python3 << EOF
from vrsbench_dataloader_production import DownloadManager, VRSBenchConfig, StructuredLogger, MetricsCollector
import os

config = VRSBenchConfig()
logger = StructuredLogger("Setup", config)
metrics = MetricsCollector()
dm = DownloadManager(config, logger, metrics)

# Download annotations
dm.download_with_retries(
    "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/Annotations_val.zip",
    "./data/annotations/Annotations_val.zip"
)

print("✓ Validation annotations downloaded")

# Optionally download images (large file ~5GB)
# dm.download_with_retries(
#     "https://huggingface.co/datasets/xiang709/VRSBench/resolve/main/images.zip",
#     "./data/images/images.zip"
# )
EOF
fi

echo ""
echo "======================================"
echo "✓ Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Review README.md for usage examples"
echo "2. Run example scripts:"
echo "   python example_classification.py"
echo "   python example_vqa.py"
echo "   python example_grounding.py"
echo ""
echo "3. Or run tests:"
echo "   python vrsbench_dataloader_production.py \\"
echo "       --images-dir ./data/images \\"
echo "       --annotations-jsonl ./data/annotations/annotations_val.jsonl \\"
echo "       --task classification"
echo ""
