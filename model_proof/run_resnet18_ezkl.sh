#!/bin/bash
#SBATCH --account=aroyc0
#SBATCH --job-name=ezkl-resnet18
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --partition=standard
#SBATCH --output=ezkl_resnet18.out
#SBATCH --error=ezkl_resnet18.err

module purge
module load gcc/11.2.0
module load python/3.10.4
module load rust/1.72.1

# Clone ezkl repo if not done already
if [ ! -d "ezkl" ]; then
  git clone https://github.com/zkonduit/ezkl.git
fi

cd ezkl

# Build CLI
cargo build --release

cd ..

# Create venv if not present
if [ ! -d "ezkl-env" ]; then
  python -m venv ezkl-env
fi

source ezkl-env/bin/activate
pip install --upgrade pip
pip install torch torchvision onnx onnxruntime

# Run your Python script (which uses ezkl CLI, not Python API)
python generate_resnet18_certificate.py
