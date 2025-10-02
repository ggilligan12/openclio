#!/bin/bash

# OpenClio Local Development Setup Script
# Run from repo root: ./dev/setup_dev.sh

set -e  # Exit on error

echo "================================================"
echo "OpenClio Local Development Setup"
echo "================================================"

# Check we're in repo root
if [ ! -f "setup.py" ]; then
    echo "Error: Please run from repository root: ./dev/setup_dev.sh"
    exit 1
fi

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Install package in development mode
echo ""
echo "Installing OpenClio in development mode..."
pip3 install -e .

echo ""
echo "✓ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Authenticate with GCP:"
echo "   gcloud auth application-default login"
echo ""
echo "2. Set your project ID in the notebook or code:"
echo "   PROJECT_ID = 'your-gcp-project-id'"
echo ""
echo "3. Run tests:"
echo "   python dev/test_local.py"
echo ""
echo "4. Or start Jupyter:"
echo "   jupyter notebook notebook.ipynb"
echo ""
echo "See dev/QUICKSTART.md and dev/LOCAL_SETUP.md for more details."
