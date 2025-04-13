#!/bin/bash
#SBATCH --comment=af-goldilocks
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ahafletcher1@sheffield.ac.uk
#SBATCH --output=log/%j.%x.out
#SBATCH --error=log/%j.%x.err

# --- Environment Setup ---
echo "Loading Anaconda module..."
module load Anaconda3/2024.02-1
echo "Activating Conda environment 'goldilocks'..."
eval "$(conda shell.bash hook)"
conda activate goldilocks
echo "Conda environment activated."

# --- Print Debug Information ---
echo "--- Debug Information ---"
echo "Which Python:"
which python
echo "Python Version:"
python --version
echo "Conda Info:"
conda info
echo "GPU Status:"
nvidia-smi
echo "--- End Debug Information ---"

# --- Check Python Availability ---
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi
echo "Python 3 found."

# --- Check Python Script Existence ---
SCRIPT_PATH="./src/sft_model.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi
echo "Python script found at $SCRIPT_PATH"

# --- Hugging Face Credentials Check ---
echo "--- Hugging Face Credentials Check ---"

export HUGGING_FACE_HUB_TOKEN="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"

if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "HUGGING_FACE_HUB_TOKEN environment variable is SET."
else
    echo "HUGGING_FACE_HUB_TOKEN environment variable is NOT SET."
fi

# Attempt to check login status using the CLI tool
echo "Running 'huggingface-cli whoami' to check authentication status:"
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli whoami
else
    echo "huggingface-cli command not found. Cannot run 'whoami'."
fi
echo "--- End Hugging Face Credentials Check ---"


# --- Run Experiments ---
echo "Starting model SFT..."

python3 "$SCRIPT_PATH"

echo "SFT completed."