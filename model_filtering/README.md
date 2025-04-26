# Difficulty Filter Pipeline

A pipeline for difficulty filtering using reward functions, supporting data-parallel and tensor-parallel processing.

## Resource Requirements

- 8x H200 (140GB) GPUs

## Example Usage

### Qwen2.5-7B-Instruct (~0.09s per data point on leetcode2k)
```bash
python model_filtering/diff_filter.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_parquet_path "data/train/codegen_mbpp_374.parquet" \
  --output_dir "./diff_filter_output" \
  --max_prompt_length 2048 \
  --truncation "left" \
  --checkpoint_freq 5 \
  --dp_size 1 \
  --tp_size 1 \
  --node_size 1 \
  --node_rank 0 \
  --master_addr "127.0.0.1" \
  --master_port 0 \
  --batch_size 128 \
  --reward_workers 64 \
  --max_new_tokens 4096
```

### Qwen2.5-32B-Instruct (~0.23s per data point on leetcode2k)
```bash
python model_filtering/diff_filter.py \
  --model_path "Qwen/Qwen2.5-32B-Instruct" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --max_prompt_length 2048 \
  --truncation "left" \
  --checkpoint_freq 5 \
  --dp_size 2 \
  --tp_size 4 \
  --node_size 1 \
  --node_rank 0 \
  --master_addr "127.0.0.1" \
  --master_port 0 \
  --batch_size 128 \
  --reward_workers 64 \
  --max_new_tokens 4096
```

### DeepSeek-R1-Distill-Qwen-32B (~6s per data point on leetcode2k)
```bash
python model_filtering/diff_filter.py \
  --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --max_prompt_length 2048 \
  --truncation "left" \
  --checkpoint_freq 5 \
  --dp_size 2 \
  --tp_size 4 \
  --node_size 1 \
  --node_rank 0 \
  --master_addr "127.0.0.1" \
  --master_port 0 \
  --batch_size 128 \
  --reward_workers 64 \
  --max_new_tokens 32768
```