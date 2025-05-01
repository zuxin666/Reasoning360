#!/usr/bin/env python3
import os
import json
import glob
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict
import torch
from datetime import datetime
import argparse
from rich.console import Console
from rich.theme import Theme
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from torch.utils.data.dataloader import default_collate

# --------------------------------------------------------------------------- #
# Rich console setup                                                          #
# --------------------------------------------------------------------------- #
custom_theme = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "bold red",
    "highlight": "bold magenta",
    "metric": "bold cyan",
    "time": "bold blue",
})
console = Console(theme=custom_theme)

# --------------------------------------------------------------------------- #
# Helper: make anything JSON serialisable                                     #
# --------------------------------------------------------------------------- #
def json_default(obj):
    """Fallback encoder for json.dump."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.dim() == 0 else obj.tolist()
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.item() if np.ndim(obj) == 0 else obj.tolist()
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")

# --------------------------------------------------------------------------- #
# Preserves dictionary and list formats without converting to tensors         #
# --------------------------------------------------------------------------- #
def custom_collate_fn(batch):
    """
    Custom collate function that preserves dictionary and list formats without converting to tensors
    """
    elem = batch[0]
    if isinstance(elem, dict):
        # For dictionaries, process each key separately
        result = {}
        for key in elem:
            values = [d[key] for d in batch]
            # Recursively process values for each key
            result[key] = custom_collate_fn(values)
        return result
    elif isinstance(elem, list):
        # For lists, return original list directly
        return batch
    elif isinstance(elem, tuple):
        # For tuples, process each element separately
        transposed = zip(*batch)
        result = []
        for samples in transposed:
            result.append(custom_collate_fn(samples))
        return tuple(result)
    else:
        # For other types, use default collate
        try:
            return default_collate(batch)
        except:
            # If default collate fails, return original data
            return batch

# --------------------------------------------------------------------------- #
# Data loading, concatenation, and analysis                                   #
# --------------------------------------------------------------------------- #
def read_json(file_path: str) -> Dict:
    """Read a single JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        console.print(f"[error]Error reading {file_path}: {e}[/error]")
        return {}

def load_concatenated_results(
    path: Optional[str] = None,
    base_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    force_reconcat: bool = False
) -> Dict:
    """
    Load concatenated results either from a specific path or by constructing the path.
    
    Args:
        path: Full path to the model directory (e.g., "diff_filter_output/codegen__leetcode2k_2.4k/DeepSeek-R1-Distill-Qwen-32B")
        base_dir: Base output directory (used with dataset_name and model_name)
        dataset_name: Name of the dataset (used with base_dir and model_name)
        model_name: Name of the model (used with base_dir and dataset_name)
        force_reconcat: Whether to force re-concatenation if concatenated results already exist
        
    Returns:
        Dictionary containing the concatenated results
    """
    # Determine the model directory path
    if path is not None:
        model_dir = path
    elif all([base_dir, dataset_name, model_name]):
        model_dir = os.path.join(base_dir, dataset_name, model_name.split("/")[0])
    else:
        console.print("[error]Either path or all of base_dir, dataset_name, and model_name must be provided[/error]")
        return {}
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        console.print(f"[error]Directory not found: {model_dir}[/error]")
        return {}
    
    # Path to concatenated results
    concat_results_path = os.path.join(model_dir, "concatenated_results.json")
    
    # Check if concatenated results already exist and can be loaded
    if os.path.exists(concat_results_path) and not force_reconcat:
        console.print(f"[info]Loading existing concatenated results from {concat_results_path}[/info]")
        return read_json(concat_results_path)
    
    # Concatenate results without saving
    if path is not None:
        # Extract dataset_name and model_name from specific_path
        parts = path.split(os.path.sep)
        if len(parts) >= 2:
            extracted_dataset = parts[-2]
            extracted_model = parts[-1]
            return concat_model_results(os.path.dirname(os.path.dirname(path)), 
                                      extracted_dataset, extracted_model, 
                                      force_reconcat=True, save_to_file=False)
        else:
            console.print(f"[error]Could not extract dataset and model from path: {path}[/error]")
            return {}
    else:
        return concat_model_results(base_dir, dataset_name, model_name, 
                                  force_reconcat=True, save_to_file=False)

def concat_model_results(output_dir: str, dataset_name: str, model_name: str, force_reconcat: bool = False, save_to_file: bool = True) -> Dict:
    """Concatenate final_results.json from all dp* directories for a model.
    
    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        model_name: Name of the model
        force_reconcat: Whether to force re-concatenation if concatenated results already exist
        save_to_file: Whether to save the concatenated results to a file
        
    Returns:
        Dictionary containing the concatenated results
    """
    model_dir = os.path.join(output_dir, dataset_name, model_name)
    if not os.path.exists(model_dir):
        console.print(f"[error]Directory not found: {model_dir}[/error]")
        return {}

    save_path = os.path.join(model_dir, "concatenated_results.json")
    if os.path.exists(save_path) and not force_reconcat:
        console.print(f"[info]Concatenated results already exist at {save_path}. Use force_reconcat=True to regenerate.[/info]")
        if save_to_file:
            return read_json(save_path)
        else:
            return read_json(save_path)

    all_results = {}
    dp_dirs = glob.glob(os.path.join(model_dir, "dp*"))
    
    if not dp_dirs:
        console.print(f"[warning]No DP directories found in {model_dir}[/warning]")
        return {}
        
    console.print(f"Found {len(dp_dirs)} DP directories")
    
    batch_performance = []
    
    for dp_dir in dp_dirs:
        # Extract dp_rank from directory name
        try:
            dp_rank = int(os.path.basename(dp_dir).replace('dp', ''))
        except ValueError:
            console.print(f"[warning]Could not extract dp_rank from {dp_dir}[/warning]")
            continue

        final_results_path = os.path.join(dp_dir, "final_results.json")
        if not os.path.exists(final_results_path):
            console.print(f"[warning]No final_results.json found in {dp_dir}[/warning]")
            continue
            
        data = read_json(final_results_path)
        if not data or "results" not in data:
            console.print(f"[warning]Invalid results format in {final_results_path}[/warning]")
            continue
            
        # Add dp_rank to keys to avoid conflicts
        dp_results = {
            f"dp{dp_rank}_{key}": value 
            for key, value in data["results"].items()
        }
        
        # Track batch performance
        batch_size = len(dp_results)
        batch_avg_pass = 0.0
        if batch_size > 0:
            batch_avg_pass = sum(item["pass_rate"] for item in dp_results.values()) / batch_size
        batch_performance.append({"dp": dp_rank, "samples": batch_size, "avg_pass_rate": batch_avg_pass})
        console.print(f"[info]Batch dp{dp_rank}: {batch_size} samples, avg pass: {batch_avg_pass:.3f}[/info]")
        
        all_results.update(dp_results)

    if not all_results:
        console.print("[error]No valid results found to concatenate[/error]")
        return {}

    result_data = {
        "results": all_results,
        "total_samples": len(all_results),
        "dp_count": len(dp_dirs),
        "batch_performance": batch_performance
    }
    
    if save_to_file:
        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2, default=json_default)
        console.print(f"[success]Concatenated {len(all_results)} samples from {len(dp_dirs)} DPs to {save_path}[/success]")
    else:
        console.print(f"[success]Concatenated {len(all_results)} samples from {len(dp_dirs)} DPs (not saved to file)[/success]")
    
    return result_data

def analyze_dataset_difficulty(output_dir: str, dataset_name: str, model_names: Optional[List[str]] = None, 
                             force_reconcat: bool = False, save_concat: bool = False) -> None:
    """Analyze difficulty distribution for each model separately."""
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        console.print(f"[error]Dataset directory not found: {dataset_dir}[/error]")
        return

    # Get all model directories if model_names not specified
    if model_names is None:
        model_names = [d for d in os.listdir(dataset_dir) 
                      if os.path.isdir(os.path.join(dataset_dir, d))]

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Analyzing models for {dataset_name}", total=len(model_names))
        
        for model in model_names:
            progress.print(f"\n[bold]Analyzing Model: {model}[/bold]")
            model_dir = os.path.join(dataset_dir, model)
            
            # Get concatenated results based on save_concat preference
            results_path = os.path.join(model_dir, "concatenated_results.json")
            data = None
            
            if os.path.exists(results_path) and not force_reconcat:
                data = read_json(results_path)
            
            if data is None or not data or "results" not in data or force_reconcat:
                try:
                    data = concat_model_results(output_dir, dataset_name, model, force_reconcat, save_to_file=save_concat)
                except Exception as e:
                    progress.print(f"[error]Failed to concatenate results for {model}: {e}[/error]")
                    progress.advance(task)
                    continue

            if not data or "results" not in data:
                progress.print(f"[warning]No valid results found for {model}[/warning]")
                progress.advance(task)
                continue

            # Calculate pass rates
            pass_rates = []
            for sample in data["results"].values():
                pass_rates.append(sample["pass_rate"])

            if not pass_rates:
                progress.print(f"[warning]No pass rates found for {model}[/warning]")
                progress.advance(task)
                continue

            # Calculate statistics
            mean_pass_rate = float(np.mean(pass_rates))
            median_pass_rate = float(np.median(pass_rates))
            std_pass_rate = float(np.std(pass_rates))
            
            # Create analysis data structure
            analysis_data = {
                "statistics": {
                    "total_samples": len(pass_rates),
                    "mean_pass_rate": mean_pass_rate,
                    "median_pass_rate": median_pass_rate,
                    "std_pass_rate": std_pass_rate,
                },
                "difficulty_distribution": {}
            }
            
            # Print basic statistics
            stats_table = Table(show_header=True, header_style="bold magenta")
            stats_table.add_column("Metric")
            stats_table.add_column("Value")
            
            stats = {
                "Total samples": len(pass_rates),
                "Mean pass rate": f"{mean_pass_rate:.3f}",
                "Median pass rate": f"{median_pass_rate:.3f}",
                "Std pass rate": f"{std_pass_rate:.3f}",
            }
            
            for metric, value in stats.items():
                stats_table.add_row(metric, str(value))
            
            progress.print(stats_table)

            # Print and calculate difficulty distribution
            dist_table = Table(show_header=True, header_style="bold magenta")
            dist_table.add_column("Difficulty")
            dist_table.add_column("Pass Rate Range")
            dist_table.add_column("Count")
            dist_table.add_column("Percentage")
            
            # Count exact 0s and 1s
            exact_zeros = sum(1 for r in pass_rates if r == 0.0)
            exact_ones = sum(1 for r in pass_rates if r == 1.0)
            
            # Add special categories for exactly 0 and exactly 1
            dist_table.add_row(
                "Impossible",
                "Exactly 0.0",
                str(exact_zeros),
                f"{(exact_zeros / len(pass_rates)) * 100:.1f}%"
            )
            
            # Update bins to exclude exact 0s and 1s
            bins = [(0, 0.2, "Very Hard", True, False), 
                   (0.2, 0.4, "Hard", False, False),
                   (0.4, 0.6, "Medium", False, False),
                   (0.6, 0.8, "Easy", False, False),
                   (0.8, 1.0, "Very Easy", False, True)]
            
            # Add to analysis data for exact 0
            analysis_data["difficulty_distribution"]["Impossible"] = {
                "range": [0.0, 0.0],
                "count": exact_zeros,
                "percentage": (exact_zeros / len(pass_rates)) * 100
            }
            
            for bin_start, bin_end, difficulty, exclude_start, exclude_end in bins:
                # Count items in this bin, respecting exclusions
                count = sum(1 for r in pass_rates if 
                           (bin_start < r if exclude_start else bin_start <= r) and
                           (r < bin_end if exclude_end else r <= bin_end))
                
                percentage = (count / len(pass_rates)) * 100
                
                # Show range notation with appropriate brackets
                range_str = f"{'(' if exclude_start else '['}{bin_start:.1f}-{bin_end:.1f}{')' if exclude_end else ']'}"
                
                dist_table.add_row(
                    difficulty,
                    range_str,
                    str(count),
                    f"{percentage:.1f}%"
                )
                
                # Add to analysis data
                analysis_data["difficulty_distribution"][difficulty] = {
                    "range": [bin_start, bin_end],
                    "exclude_start": exclude_start,
                    "exclude_end": exclude_end,
                    "count": count,
                    "percentage": percentage
                }
            
            # Add special category for exactly 1
            dist_table.add_row(
                "Perfect",
                "Exactly 1.0",
                str(exact_ones),
                f"{(exact_ones / len(pass_rates)) * 100:.1f}%"
            )
            
            # Add to analysis data for exact 1
            analysis_data["difficulty_distribution"]["Perfect"] = {
                "range": [1.0, 1.0],
                "count": exact_ones,
                "percentage": (exact_ones / len(pass_rates)) * 100
            }
            
            progress.print(dist_table)
            
            # Save analysis data to JSON file
            analysis_path = os.path.join(model_dir, "analysis_results.json")
            try:
                with open(analysis_path, 'w') as f:
                    json.dump(analysis_data, f, indent=2, default=json_default)
                progress.print(f"[success]Saved analysis results to {analysis_path}[/success]")
            except Exception as e:
                progress.print(f"[error]Failed to save analysis results for {model}: {e}[/error]")
            
            progress.print()  # Add a blank line between models
            progress.advance(task)

def main():
    parser = argparse.ArgumentParser(description="Dataset difficulty analysis tools")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Concat command
    concat_parser = subparsers.add_parser('concat', help='Concatenate results for a model')
    concat_parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')
    concat_parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    concat_parser.add_argument('--model', type=str, required=True, help='Model name')
    concat_parser.add_argument('--force', action='store_true', help='Force re-concatenation even if file exists')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze dataset difficulty')
    analyze_parser.add_argument('--output_dir', type=str, required=True, help='Base output directory')
    analyze_parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    analyze_parser.add_argument('--models', type=str, nargs='*', help='Optional: specific models to analyze')
    analyze_parser.add_argument('--force_reconcat', action='store_true', help='Force re-concatenation of results')
    analyze_parser.add_argument('--save_concat', action='store_true', help='Save concatenated results after analysis')

    args = parser.parse_args()

    if args.command == 'concat':
        concat_model_results(args.output_dir, args.dataset, args.model, args.force)
    elif args.command == 'analyze':
        analyze_dataset_difficulty(args.output_dir, args.dataset, args.models, 
                                 args.force_reconcat, args.save_concat)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()