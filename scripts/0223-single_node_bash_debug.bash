export head_node=${nodes[0]}

head_node_ip=$(hostname --ip-address)
port=6379
address_head=$head_node_ip:$port

# Experiment config
WORKING_DIR=${HOME}/Reasoning360
DATA_DIR=${WORKING_DIR}/data
deepscaler_train_path=${DATA_DIR}/deepscaler_preview/train.parquet
# math_test_path=${DATA_DIR}/deepscaler_preview/math.parquet
aime_test_path=${DATA_DIR}/deepscaler_preview/aime.parquet

train_files="['$deepscaler_train_path']"
test_files="['$aime_test_path']"
BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

WANDB_PROJECT=Reasoning360
WANDB_EXPERIMENT_NAME=deepscaler-${BASE_MODEL##*/}-${SLURM_JOB_ID}

export worker_num=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=ens10f0np0
export HYDRA_FULL_ERROR=1

# Remove existing Ray cluster
ray stop
rm -rf /tmp/ray/ray_current_cluster

# Start Ray head node
ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus 96 --num-gpus 8

# Start training
"${CONDA_BIN_PATH}python" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=64 \
    data.val_batch_size=2048 \
    data.max_prompt_length=2048 \
    data.max_response_length=8096 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$worker_num \
    trainer.save_freq=3 \
    trainer.test_freq=3 \
    trainer.total_epochs=5