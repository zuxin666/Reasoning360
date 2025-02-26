#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --partition=mbzuai
#SBATCH --exclude="g42-h100-instance-[119]"
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1500G
#SBATCH --output=slurm_log/slurm_ppo_v1_%j.out
#SBATCH --error=slurm_log/slurm_ppo_v1_%j.err
#SBATCH --exclusive
#SBATCH --time=12:00:00

# source ~/.bashrc

# # Activate the Conda environment
# conda activate verl

# Determine the master address from the first allocated node
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
master_addr=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# master_addr=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
echo "Master address: $master_addr"

# Set the number of processes per node and total number of nodes
nproc_per_node=8
# nnodes=4 # set by SBATCH params

# Set the path to save checkpoints
save_path=/mbz/users/yuheng.zha/LongReasoning/sft_ckpts/multi_qwen_2_5_32b_limo_sys_prompt_full_len

# Launch the distributed training job using torchrun.
srun bash ./slurm-sub_qwen_32_sp2_lig.sh $nproc_per_node $master_addr $save_path

