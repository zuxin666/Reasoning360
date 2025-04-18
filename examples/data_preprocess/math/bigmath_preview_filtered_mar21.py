"""Downloads, processes, and saves math datasets."""

import os
import datasets
from typing import Dict, List, Optional, Any, Union
import enum
import argparse
import pandas as pd
import json
import random

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from verl.utils.data_process.prompt import build_zero_style_prompt
from verl.utils.data_process.utils import add_suffix, set_seed

InstructionFollow = "Please output the final answer within \\boxed{}."

def extract_solution(solution_str: str) -> str:
    """Extracts the final answer assuming it's in the last \\boxed{}."""
    return remove_boxed(last_boxed_only_string(solution_str))


def make_map_fn(split: str, data_source: str, prompt_style: str) -> callable:
    def process_fn(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        question = example.pop("problem")
        answer = example.pop("answer")
        
        if prompt_style == "zero_style":
            prompt = build_zero_style_prompt(extra_instruction=InstructionFollow)
        else:
            raise ValueError(f"Invalid prompt style: {prompt_style}")

        if prompt_style == "zero_style":
            data = {
                "data_source": data_source,
                "prompt": [],  # no messages-like prompt. instead, use from-scratch raw_prompt
                "raw_prompt": prompt.replace("{{prompt}}", question),
                "ability": "math",
                "apply_chat_template": False,
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {"split": split, "index": idx},
            }
        else:
            raise ValueError(f"Invalid prompt style: {args.prompt_style}")

        if idx == 0:
            print(f"data_source: {data_source}, split: {split}, idx: {idx}")
            print("\n" + "=" * 100 + f"{data_source} {split} {idx}" + "=" * 10)
            print(data)
        return data

    return process_fn
    
    
if __name__ == "__main__":
    """Main script execution: parse args, load, process, and save datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to save the processed data files. Will be modified based on other parameters.",
    )
    parser.add_argument(
        "--output-train-filename",
        default="bigmath_filtered_mar21",
        help="Directory to save the processed data files. Will be modified based on other parameters.",
    )
    parser.add_argument(
        "--train-sample-size",
        type=int,
        default=None,
        help="Number of samples to use from the training dataset. If None, use all samples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility when sampling data.",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        choices=["zero_style", "instruction"],
        default="zero_style",
        help="Prompt style to use: 'zero_style' for think-then-answer format or 'instruction' for direct instruction format.",
    )
    args = parser.parse_args()
    set_seed(args.seed)

    train_data_source = "SDSB/big_math_partial_mar21_filtered_basic"
    test_data_sources = [
        "nanoverl/minerva",
        "SDSB/aime_repeated_8x",
        "SDSB/amc_repeated_4x",
        "nanoverl/olympiad_bench",
        "nanoverl/math",
    ]

    # Download the datasets from Hugging Face Hub
    cache_dir = datasets.config.HF_DATASETS_CACHE
    print(f"Loading the {train_data_source} dataset...")
    train_dataset = datasets.load_dataset(
        train_data_source, trust_remote_code=True, split="train"
    )
    print(f"Loading the test datasets...")
    test_datasets = [
        datasets.load_dataset(test_data_source, trust_remote_code=True, split="test")
        for test_data_source in test_data_sources
    ]

    # Process the train dataset using the map function
    train_data = train_dataset.map(function=make_map_fn("train", train_data_source, args.prompt_style), with_indices=True)
    if args.train_sample_size is not None:
        # Optionally sample the training data
        train_indices = list(range(len(train_data)))
        random.shuffle(train_indices)
        train_indices = train_indices[: min(args.train_sample_size, len(train_data))]
        train_data = train_data.select(train_indices)

    # Save the processed train data to a Parquet file
    args.output_train_filename = add_suffix(args.output_train_filename, args.train_sample_size)
    train_data.to_parquet(os.path.join(args.data_dir, f"{args.output_train_filename}.parquet"))
    print(f"train data size:", len(train_data))
    print(train_data[0])

    # Process and save each test dataset
    for test_data_source, test_data in zip(test_data_sources, test_datasets):
        process_fn = make_map_fn("test", test_data_source, args.prompt_style)
        test_data = test_data.map(process_fn, with_indices=True)
        dataset_name = os.path.basename(test_data_source.lower())
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(args.data_dir, "test", f"{dataset_name}.parquet")) # Save test data
        print(f"test data size: ({dataset_name})", len(test_df))

    print(f"Done! \n"
          f"Train data saved to {args.data_dir}/train/{args.output_train_filename}.parquet\n"
          f"Test data saved to {args.data_dir}/test/")
