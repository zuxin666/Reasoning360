# Copyright 2024
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess a CSV dataset to parquet format
"""
import os
import datasets
import argparse
import re
from sklearn.model_selection import train_test_split

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
)
    
def make_prefix(dp, model_type = 'instruct'):
    constraints = dp['input']
    result = dp['ground_truth']
    instruction = dp['instruction']
    if model_type == 'instruct':
        prefix = f"{instruction}. The constraints are: {constraints}. Think step by step to find the answer. Show your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <answer> ['pigeon', 'sparrow', 'quail'] </answer>."
    else:
        prefix = f"{instruction}. The constraints are: {constraints}."
    return prefix


def extract_from_ground_truth(text):
    if isinstance(text, list):
        return text
    else:
        return eval(text)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', default='../data/puzzles_dataset/puzzles_dataset.json', help='Path to json file')
    parser.add_argument('--local_dir', default='../data/puzzles_dataset', help='Local directory to save parquet files')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory (optional)')
    parser.add_argument('--test_size', type=float, default=0.1, help='Proportion of data for test set')
    parser.add_argument('--data_source', default='ordering_puzzle_dataset', help='Name of data source')
    parser.add_argument('--model_type', default='instruct', choices = ['base', 'instruct'], help='Model type base or instruct')
    args = parser.parse_args()
    
    # Load dataset from CSV
    dataset = datasets.load_dataset('json', data_files=args.json_path)['train']
   
    prompt ="""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the response. The reasoning process is enclosed within <think> </think> i.e., <think> reasoning process here </think> respond to the user's question here.

User: {{prompt}} Please put your answer within <answer> and </answer> tags, for example <answer> ['pigeon', 'sparrow', 'quail'] </answer>.
Assistant: <think>
"""
    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example, args.model_type)
            num_objects = example['num_objects']
            final_arrangement = extract_from_ground_truth(example['ground_truth'])
            if args.model_type == 'instruct':
                return {
                    "data_source": args.data_source,
                    "prompt": [{
                        "role": "user",
                        "content": question,
                    }],
                    "ability": "logical_reasoning",
                    "reward_model": {
                            "style": "rule",
                            "ground_truth": final_arrangement
                        },
                    "apply_chat_template": True,
                    "extra_info": {
                        'id': example['id'] if 'id' in example else str(idx),
                        'raw_instruction': example['instruction'],
                        'raw_input': example['input'],
                        'num_objects': num_objects,
                    }
                }
            elif args.model_type == 'base':
                return {
                    "data_source": args.data_source,
                    "prompt": [],
                    "raw_prompt": prompt.replace("{{prompt}}", question),
                    "ability": "logical_reasoning",
                    "reward_model": {
                            "style": "rule",
                            "ground_truth": final_arrangement
                        },
                    "apply_chat_template": False,
                    "extra_info": {
                        'id': example['id'] if 'id' in example else str(idx),
                        'raw_instruction': example['instruction'],
                        'raw_input': example['input'],
                        'num_objects': num_objects,
                    }
                }
        return process_fn
    
    # Transform dataset
    processed_dataset = dataset.map(function=make_map_fn('train'), with_indices=True)
    
    # Create train/test split
    train_indices, test_indices = train_test_split(
        range(len(processed_dataset)), 
        test_size=args.test_size, 
        random_state=42
    )
    
    # Create train and test datasets
    train_dataset = processed_dataset.select(train_indices)
    test_dataset = processed_dataset.select(test_indices)
    
    # Add split information
    def add_split_info(example, split_name):
        example['extra_info']['split'] = split_name
        return example
    
    train_dataset = train_dataset.map(lambda x: add_split_info(x, 'train'))
    test_dataset = test_dataset.map(lambda x: add_split_info(x, 'test'))
    
    # Create directory if it doesn't exist
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Save to parquet
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
            print(f"Data copied to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("HDFS utilities not available. Install verl package for HDFS support.")
            
    print(f"Conversion complete. Files saved to {local_dir}")
    print(f"Train set: {len(train_dataset)} examples")
    print(f"Test set: {len(test_dataset)} examples")