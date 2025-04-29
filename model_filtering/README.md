# Difficulty Filter Pipeline

A pipeline for difficulty filtering using reward functions, supporting data-parallel and tensor-parallel processing. The pipeline now operates in two stages:
1. **Inference stage**: Runs the model to generate responses (GPU-intensive)
2. **Reward stage**: Evaluates responses with reward functions (CPU-intensive)

## Resource Requirements

- 8x H200 (140GB) GPUs

## Example Usage

### Stage 1: Inference

#### Qwen2.5-7B-Instruct (~0.09s per data point on leetcode2k)
```bash
python model_filtering/run_inference.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --max_prompt_length 2048 \
  --truncation "left" \
  --dp_size 8 \
  --tp_size 1 \
  --node_size 1 \
  --node_rank 0 \
  --master_addr "127.0.0.1" \
  --master_port 0 \
  --batch_size 128 \
  --n 16 \
  --max_new_tokens 4096
```

#### DeepSeek-R1-Distill-Qwen-32B (~17.5s per data point on leetcode2k)
```bash
python model_filtering/run_inference.py \
  --model_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --max_prompt_length 2048 \
  --truncation "left" \
  --dp_size 2 \
  --tp_size 4 \
  --node_size 1 \
  --node_rank 0 \
  --master_addr "127.0.0.1" \
  --master_port 0 \
  --batch_size 128 \
  --n 16 \
  --max_new_tokens 32768
```

### Stage 2: Reward Calculation

After the inference stage is complete, run the reward calculation:

```bash
python model_filtering/run_reward.py \
  --model_path "Qwen/Qwen2.5-7B-Instruct" \
  --dataset_parquet_path "data/train/codegen__leetcode2k_2.4k.parquet" \
  --output_dir "./diff_filter_output" \
  --reward_workers 64 \
  --correct_reward_threshold 1.0
```

## Checkpoint & Resumption

The pipeline automatically saves batch results to JSON files in the output directory as it processes data. If a run is interrupted, the pipeline will automatically resume from where it left off by:

1. Scanning the output directory for existing batch files
2. Loading results from these files
3. Continuing from the next unprocessed batch

### Advanced Checkpoint Options

The pipeline provides flags to control checkpoint behavior:

- **Inference stage**:
  - `--force_regenerate`: Ignores all existing checkpoint files and starts processing from the beginning, overwriting previous results

- **Reward stage**:
  - `--recalculate_rewards`: Recalculates all reward scores using the previously generated model responses (useful when implementing new evaluation metrics without re-running the expensive model inference)

## Utility Functions

After running models with data-parallel processing, you can use the utility functions to concatenate and analyze results.

### Concatenating Results

For multi-DP runs, results are distributed across multiple directories. The `concat` command gathers all results into a single file:

```bash
python -m model_filtering.utils concat \
  --output_dir "./diff_filter_output" \
  --dataset "codegen__leetcode2k_2.4k" \
  --model "Qwen2.5-7B-Instruct"
```

Some advanced options:
- `--force`: Force regeneration of concatenated files even if they already exist

### Analyzing Results

Analyze difficulty distributions of one or more models:

```bash
python -m model_filtering.utils analyze \
  --output_dir "./diff_filter_output" \
  --dataset "codegen__leetcode2k_2.4k" 
```

Analyze specific models:

```bash
python -m model_filtering.utils analyze \
  --output_dir "./diff_filter_output" \
  --dataset "codegen__leetcode2k_2.4k" \
  --models "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct"
```

Some advanced options:
- `--force_reconcat`: Force regeneration of concatenated files even if they already exist
- `--save_concat`: Don't save concatenated results after analysis

### Output Files

The utility functions produce the following output files for each model:

- `concatenated_results.json`: Combined results from all DP workers
- `analysis_results.json`: Statistics and difficulty distribution analysis

The analysis output categorizes problems into seven difficulty levels:
- Impossible (pass rate exactly 0.0)
- Very Hard (pass rate 0.0-0.2, exclusive)
- Hard (pass rate 0.2-0.4)
- Medium (pass rate 0.4-0.6)
- Easy (pass rate 0.6-0.8)
- Very Easy (pass rate 0.8-1.0, exclusive)
- Perfect (pass rate exactly 1.0)