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

def concat_model_results(output_dir: str, dataset_name: str, model_name: str, force_reconcat: bool = False) -> None:
    """Concatenate final_results.json from all dp* directories for a model."""
    model_dir = os.path.join(output_dir, dataset_name, model_name)
    if not os.path.exists(model_dir):
        console.print(f"[error]Directory not found: {model_dir}[/error]")
        return

    save_path = os.path.join(model_dir, "concatenated_results.json")
    if os.path.exists(save_path) and not force_reconcat:
        console.print(f"[info]Concatenated results already exist at {save_path}. Use force_reconcat=True to regenerate.[/info]")
        return

    all_results = {}
    dp_dirs = glob.glob(os.path.join(model_dir, "dp*"))
    
    if not dp_dirs:
        console.print(f"[warning]No DP directories found in {model_dir}[/warning]")
        return
        
    console.print(f"Found {len(dp_dirs)} DP directories")
    
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
        all_results.update(dp_results)

    if not all_results:
        console.print("[error]No valid results found to concatenate[/error]")
        return

    with open(save_path, 'w') as f:
        json.dump({
            "results": all_results,
            "total_samples": len(all_results),
            "dp_count": len(dp_dirs)
        }, f, indent=2, default=json_default)
    
    console.print(f"[success]Concatenated {len(all_results)} samples from {len(dp_dirs)} DPs to {save_path}[/success]")

def analyze_dataset_difficulty(output_dir: str, dataset_name: str, model_names: Optional[List[str]] = None, 
                             force_reconcat: bool = False, save_concat: bool = True) -> None:
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
            
            # Try to load concatenated results first
            results_path = os.path.join(model_dir, "concatenated_results.json")
            if not os.path.exists(results_path) or force_reconcat:
                try:
                    concat_model_results(output_dir, dataset_name, model, force_reconcat)
                except Exception as e:
                    progress.print(f"[error]Failed to concatenate results for {model}: {e}[/error]")
                    progress.advance(task)
                    continue

            data = read_json(results_path)
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
            
            bins = [(0, 0.2, "Very Hard"), 
                   (0.2, 0.4, "Hard"),
                   (0.4, 0.6, "Medium"),
                   (0.6, 0.8, "Easy"),
                   (0.8, 1.0, "Very Easy")]
            
            for bin_start, bin_end, difficulty in bins:
                count = sum(1 for r in pass_rates if bin_start <= r < bin_end)
                percentage = (count / len(pass_rates)) * 100
                dist_table.add_row(
                    difficulty,
                    f"{bin_start:.1f}-{bin_end:.1f}",
                    str(count),
                    f"{percentage:.1f}%"
                )
                
                # Add to analysis data
                analysis_data["difficulty_distribution"][difficulty] = {
                    "range": [bin_start, bin_end],
                    "count": count,
                    "percentage": percentage
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
            
            # Optionally remove concatenated results if save_concat is False
            if not save_concat and os.path.exists(results_path):
                try:
                    os.remove(results_path)
                    progress.print(f"[info]Removed concatenated results for {model} (save_concat=False)[/info]")
                except Exception as e:
                    progress.print(f"[warning]Failed to remove concatenated results for {model}: {e}[/warning]")
            
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
    analyze_parser.add_argument('--no_save_concat', action='store_true', help='Don\'t save concatenated results after analysis')

    args = parser.parse_args()

    if args.command == 'concat':
        concat_model_results(args.output_dir, args.dataset, args.model, args.force)
    elif args.command == 'analyze':
        analyze_dataset_difficulty(args.output_dir, args.dataset, args.models, 
                                 args.force_reconcat, not args.no_save_concat)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()