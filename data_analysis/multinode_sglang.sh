#!/bin/bash -l
#SBATCH --job-name=sgl-multinode
#SBATCH --partition=mbzuai
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --output=slurm/sgl-multinode-%j.out
#SBATCH --error=slurm/sgl-multinode-%j.err
#SBATCH --time=24:00:00

source /mbz/users/shibo.hao/miniforge3/etc/profile.d/conda.sh
conda activate sglang


# Define parameters
model=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

echo "[INFO] Running inference"
echo "[INFO] Model: $model"
echo "[INFO] TP Size: $tp_size"

# Set NCCL initialization address using the hostname of the head node
HEAD_NODE=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
NCCL_INIT_ADDR="${HEAD_NODE}:8000"
echo "[INFO] NCCL_INIT_ADDR: $NCCL_INIT_ADDR"

# Launch the model server on each node using SLURM
srun --ntasks=4 --nodes=4 --output="SLURM_Logs/%x_%j_node$SLURM_NODEID.out" \
    --error="SLURM_Logs/%x_%j_node$SLURM_NODEID.err" \
    python3 -m sglang.launch_server \
    --model-path "$model" \
    --grammar-backend "xgrammar" \
    --data-parallel-size 32  \
    --dist-init-addr "$NCCL_INIT_ADDR" \
    --nnodes 4 \
    --node-rank "$SLURM_NODEID" &

# Wait for the NCCL server to be ready on port 30000
while ! nc -z "$HEAD_NODE" 30000; do
    sleep 1
    echo "[INFO] Waiting for $HEAD_NODE:30000 to accept connections"
done

echo "[INFO] $HEAD_NODE:30000 is ready to accept connections"

# Keep the script running until the SLURM job times out
wait