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

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate dolphin

# Print Python and conda information for debugging
which python
python --version
conda info

nvidia-smi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Path to the Python script
SCRIPT_PATH="./src/baseline_model.py"

# Check if the Python script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Python script not found at $SCRIPT_PATH"
    exit 1
fi



models=("Qwen/Qwen2.5-Coder-0.5B-Instruct")


# Iterate through all combinations
for model in "${models[@]}"; do
 echo "Running with parameters: $ds_dir $model"
 python3 "$SCRIPT_PATH" --model_name "$model"
done

echo "All completed."