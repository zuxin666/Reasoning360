#!/usr/bin/env python3
"""
Reward-only pass for batches produced by diff_infer.py
Optimised for balanced CPU utilisation.
"""
import argparse
import glob
import json
import os
import time
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from verl.utils.reward_score import _default_compute_score
from model_filtering.utils import console, json_default


# --------------------------------------------------------------------------- #
# Utility helpers                                                             #
# --------------------------------------------------------------------------- #
def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def compute_single_reward(arg_tuple):
    """
    Wrapper so we can pass a tuple through Pool.imap_unordered.
    Returns (idx, detailed_dict)
    """
    idx, response, data_source, ground_truth, extra_info = arg_tuple
    try:
        result = _default_compute_score(
            data_source=data_source,
            solution_str=response,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if isinstance(result, dict):
            detailed = result
            score = detailed.get("score", 0.0)
        else:  # float
            detailed = {"score": float(result)}
            score = float(result)
        detailed["score"] = score
        return idx, detailed
    except Exception as e:  # never crash whole pool
        return idx, {"score": 0.0, "error": str(e)}


# --------------------------------------------------------------------------- #
# Per-rank scoring                                                             #
# --------------------------------------------------------------------------- #
def score_rank_dir(rank_dir: str, args, reward_pool: Pool | None, workers: int):
    batch_files = sorted(glob.glob(os.path.join(rank_dir, "batch_*.json")))
    if not batch_files:
        console.print(f"⚠️ [warning]No batch files found in {rank_dir} — skipping[/warning]")
        return 0.0

    rank_start = time.time()
    global_results = {}
    total_responses = 0

    for batch_file in tqdm(batch_files, desc=f"💯 Scoring {os.path.basename(rank_dir)}", position=0):
        with open(batch_file, "r") as f:
            batch = json.load(f)

        # Build flat task list: (global_index, response, …)
        tasks, lookup = [], {}     # lookup maps global idx → (sample_idx, resp_idx)
        gid = 0
        for s_idx, sample in batch.items():
            ground_truth = sample["ground_truth"]
            data_source  = sample["source"]
            extra_info   = sample["extra_info"]
            responses    = sample["responses"]

            for r_idx, raw_resp in enumerate(responses):
                stripped = raw_resp.split("</think>", 1)[1] if "</think>" in raw_resp else raw_resp
                tasks.append(
                    (gid, stripped, data_source, ground_truth, extra_info)
                )
                lookup[gid] = (s_idx, r_idx)
                gid += 1

        total_responses += len(tasks)

        # Decide chunk size once per batch to minimise overhead
        if reward_pool:
            chunksize = max(1, len(tasks) // (workers * 8))
            results_iter = reward_pool.imap_unordered(compute_single_reward, tasks, chunksize)
        else:
            results_iter = map(compute_single_reward, tasks)

        # Place scores back into samples
        detailed_by_sample = {}
        batch_name = os.path.basename(batch_file)
        
        # Add inner progress bar for individual response scoring
        inner_pbar = tqdm(
            total=len(tasks), 
            desc=f"🧮 Responses in {batch_name}", 
            position=1, 
            leave=False, 
            ncols=80,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
        )
        
        for gidx, detailed in results_iter:
            s_idx, _ = lookup[gidx]
            detailed_by_sample.setdefault(s_idx, []).append(detailed)
            inner_pbar.update(1)
            
        inner_pbar.close()

        # Update samples with scores / pass rate
        for s_idx, sample in batch.items():
            detailed_list = detailed_by_sample.get(s_idx, [])
            scores = [d["score"] for d in detailed_list]
            pass_cnt = sum(s >= args.correct_reward_threshold for s in scores)
            sample.update(
                {
                    "detailed_scores": detailed_list,
                    "scores":          scores,
                    "pass_rate":       pass_cnt / len(scores) if scores else 0.0,
                }
            )
            global_results[f"{os.path.basename(batch_file)}_{s_idx}"] = sample

        # Overwrite batch file with scores added
        with open(batch_file, "w") as f:
            json.dump(batch, f, indent=2, default=json_default)

    elapsed = time.time() - rank_start
    final_path = os.path.join(rank_dir, "final_results.json")
    with open(final_path, "w") as f:
        json.dump(
            {
                "results": global_results,
                "errors":  {},          # kept for parity with earlier format
                "metrics": {
                    "total_time": elapsed,
                    "num_responses": total_responses,
                    "avg_reward_time": elapsed / max(1, total_responses),
                },
            },
            f,
            indent=2,
            default=json_default,
        )
    console.print(f"✅ Saved summary to [highlight]{final_path}[/highlight]")
    return elapsed


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Reward-only pass (efficient, balanced)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_parquet_path", type=str, required=True,
                        help="Only used to locate the dataset-named output directory")
    parser.add_argument("--output_dir", type=str, default="./diff_filter_output")

    parser.add_argument("--reward_workers", type=int, default=16,
                        help="Upper bound on CPU processes. The script will down-scale if necessary.")
    parser.add_argument("--correct_reward_threshold", type=float, default=1.0)
    parser.add_argument("--recalculate_rewards", action="store_true",
                        help="Recompute even if sample already has scores")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Smart worker count selection                                       #
    # ------------------------------------------------------------------ #
    avail = cpu_count() or 1
    if args.reward_workers <= 1:
        workers = 1
    else:
        reserved = max(2, avail // 4)          # keep at least 25 % or 2 cores free
        workers = min(args.reward_workers, max(1, avail - reserved))

    console.rule("[bold]Difficulty Filter — Reward pass", style="cyan")
    console.print(f"⏰  Start : {datetime.now():%Y-%m-%d %H:%M:%S}")
    console.print(f"🖥️  CPUs  : available={avail}, using={workers}")
    console.rule(style="cyan")

    dataset_name = os.path.basename(args.dataset_parquet_path).rsplit(".parquet", 1)[0]
    model_name   = args.model_path.split("/")[-1]
    root_dir     = os.path.join(args.output_dir, dataset_name, model_name)

    rank_dirs = sorted(glob.glob(os.path.join(root_dir, "dp*")))
    if not rank_dirs:
        console.print(f"❌ [error]No dp* directories under {root_dir}")
        return

    # Spawn pool (or run serially)
    reward_pool = Pool(processes=workers) if workers > 1 else None

    total_elapsed = 0.0
    for rd in rank_dirs:
        total_elapsed += score_rank_dir(rd, args, reward_pool, workers)

    if reward_pool:
        reward_pool.close()
        reward_pool.join()

    console.rule(style="cyan")
    console.print(f"🏁 Finished scoring in {format_time(total_elapsed)}")


if __name__ == "__main__":
    main()
