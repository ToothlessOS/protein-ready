#!/bin/bash

# Setup script for Protein Representation Learning with Fine-tuning
# This script helps set up the environment and provides usage examples

set -e

echo "=== Protein Representation Learning Setup ==="
echo ""

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    print_status "Python version $python_version is compatible"
else
    print_error "Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating project directories..."
mkdir -p dataset/protein_g
mkdir -p fine_tuning_results/checkpoints
mkdir -p fine_tuning_results/logs
mkdir -p lightning_logs

# Download example data (if needed)
if [ ! "$(ls -A dataset/protein_g 2>/dev/null)" ]; then
    print_warning "No protein graph data found in dataset/protein_g/"
    print_warning "Please add your protein graph files (.pt format) to this directory"
fi

# Check for CUDA availability
print_status "Checking CUDA availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
print_status "Setup completed successfully!"
echo ""

# Print usage examples
echo "=== Usage Examples ==="
echo ""

echo "1. Pretraining (original functionality):"
echo "   python main.py --batch_size 32 --lr 1e-4 --max_epochs 100"
echo ""

echo "2. Fine-tuning for classification:"
echo "   python main.py --downstream_task --task_type classification --num_classes 3 --pretrained_path path/to/model.ckpt"
echo ""

echo "3. Fine-tuning for regression:"
echo "   python main.py --downstream_task --task_type regression --output_dim 1 --pretrained_path path/to/model.ckpt"
echo ""

echo "4. Run examples:"
echo "   cd examples && python classification_example.py"
echo "   cd examples && python regression_example.py"
echo ""

echo "5. Quick feature extraction:"
cat << 'EOF'
   python -c "
   from model.protein_encoder import ProteinEncoder
   encoder = ProteinEncoder.from_pretrained('path/to/model.ckpt')
   # Use encoder for feature extraction
   "
EOF
echo ""

# Print helpful tips
echo "=== Tips ==="
echo ""
echo "• For GPU training, ensure CUDA is properly installed"
echo "• Start with smaller batch sizes if you encounter memory issues"
echo "• Use --fast_dev_run for quick testing"
echo "• Monitor training with TensorBoard: tensorboard --logdir=fine_tuning_results/logs"
echo "• Check examples/ directory for detailed usage patterns"
echo ""

print_status "Ready to go! Activate the environment with: source venv/bin/activate"
