# SLURM_NODEID is an integer starting at 0 that serves as node rank
nnodes=${SLURM_JOB_NUM_NODES}
node_rank=${SLURM_NODEID}
nproc_per_node=$1
master_addr=$2
save_path=$3
echo "Node rank: $node_rank"
if [ "$node_rank" -eq 0 ]; then
    echo "Number of nodes: ${nnodes}"
    echo "Number of processes per node: ${nproc_per_node}"
    echo "Master address: ${master_addr}"
fi
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate verl

torchrun \
    --nnodes=${nnodes} \
    --nproc_per_node=${nproc_per_node} \
    --node_rank=${node_rank} \
    --master_addr=${master_addr} \
    --master_port=30001 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/mbz/users/yuheng.zha/LongReasoning/datasets/limo/train.parquet \
    data.val_files=/mbz/users/yuheng.zha/LongReasoning/datasets/limo/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-5 \
    optim.warmup_steps_ratio=0.05 \
    optim.weight_decay=0.0001 \
    +data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=16 \
    model.partial_pretrain=Qwen/Qwen2.5-32B-Instruct \
    model.use_liger=True \
    trainer.default_local_dir=${save_path} \
    trainer.project_name=limo-sft \
    trainer.experiment_name=limo-sft-qwen-2.5-32b-instruct-sp2-liger-sys-prompt \
    trainer.logger="['console', 'wandb']" \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=5 \
    ulysses_sequence_parallel_size=8 \
    use_remove_padding=true