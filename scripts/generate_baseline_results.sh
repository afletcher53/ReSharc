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

SCRIPT_PATH="./src/baseline_model.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi
echo "Python script found at $SCRIPT_PATH"

echo "--- Hugging Face Credentials Check ---"

export HUGGING_FACE_HUB_TOKEN="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"

# Check for the environment variable
if [ -n "$HUGGING_FACE_HUB_TOKEN" ]; then
    echo "HUGGING_FACE_HUB_TOKEN environment variable is SET."
else
    echo "HUGGING_FACE_HUB_TOKEN environment variable is NOT SET."
fi

echo "Running 'huggingface-cli whoami' to check authentication status:"
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli whoami
else
    echo "huggingface-cli command not found. Cannot run 'whoami'."
fi
echo "--- End Hugging Face Credentials Check ---"


# --- Define Models ---
models=("meta-llama/Llama-3.2-3B-Instruct" "meta-llama/Llama-3.1-8B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct" "Qwen/Qwen2.5-Coder-14B-Instruct")
# models=(""Qwen/Qwen2.5-Coder-0.5B-Instruct" "meta-llama/Llama-3.3-70B-Instruct" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct" "Qwen/Qwen2.5-Coder-14B-Instruct")


# --- Run Experiments ---
echo "Starting model runs..."
# Iterate through all combinations
for model in "${models[@]}"; do
 echo "----------------------------------------"
 echo "Running with model: $model"
 echo "----------------------------------------"
 python3 "$SCRIPT_PATH" --model_name "$model"
 echo "Finished model: $model"
done

echo "All models completed."