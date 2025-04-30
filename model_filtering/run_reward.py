#!/usr/bin/env python3
"""
Reward-only pass for batches produced by diff_infer.py
Optimised for balanced CPU utilisation and finer-grained task dispatch.
"""
import argparse
import glob
import json
import os
import time
from datetime import datetime, timedelta
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from typing import Optional

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
def score_rank_dir(rank_dir: str, args, reward_pool: Optional[Pool], workers: int):
    batch_files = sorted(glob.glob(os.path.join(rank_dir, "batch_*.json")))
    if not batch_files:
        console.print(f"⚠️  [warning]No batch files found in {rank_dir} — skipping[/warning]")
        return 0.0

    rank_start = time.time()
    rank_results = {}
    total_responses = 0
    total_pass_count = 0
    total_samples_with_scores = 0

    for batch_file in tqdm(batch_files, desc=f"💯 Scoring {os.path.basename(rank_dir)}", position=0):
        with open(batch_file, "r") as f:
            batch = json.load(f)

        tasks = []
        lookup = {}
        batch_name = os.path.basename(batch_file)
        gid = 0
        batch_samples_count = len(batch)
        batch_already_scored = 0

        # Build per-response task list, skipping any sample that already has scores
        for s_idx, sample in batch.items():
            if not args.recalculate_rewards and sample.get("scores"):
                # carry forward existing scored sample
                rank_results[f"{batch_name}_{s_idx}"] = sample
                batch_already_scored += 1
                continue

            ground_truth = sample["ground_truth"]
            data_source = sample["source"]
            extra_info = sample["extra_info"]
            responses = sample["responses"]

            for r_idx, raw_resp in enumerate(responses):
                stripped = raw_resp.split("</think>", 1)[1] if "</think>" in raw_resp else raw_resp
                tasks.append((gid, stripped, data_source, ground_truth, extra_info))
                lookup[gid] = (s_idx, r_idx)
                gid += 1

        total_responses += len(tasks)

        if tasks:
            console.print(f"📊 Processing [highlight]{batch_name}[/highlight]: {len(batch)} samples ({batch_already_scored} already scored)")
            
            # use fine-grained dispatch to avoid long-blocking chunks on timeouts
            if reward_pool:
                results_iter = reward_pool.imap_unordered(compute_single_reward, tasks, chunksize=1)
            else:
                results_iter = map(compute_single_reward, tasks)

            inner_pbar = tqdm(
                total=len(tasks),
                desc=f"🧮 Responses in {batch_name}",
                position=1,
                leave=False,
                ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'
            )

            detailed_by_sample = {}
            for gidx, detailed in results_iter:
                s_idx, _ = lookup[gidx]
                detailed_by_sample.setdefault(s_idx, []).append(detailed)
                inner_pbar.update(1)

            inner_pbar.close()

            # integrate scores back into each sample
            batch_pass_count = 0
            batch_total_scores = 0
            for s_idx, sample in batch.items():
                if s_idx not in detailed_by_sample:
                    continue
                detailed_list = detailed_by_sample[s_idx]
                scores = [d["score"] for d in detailed_list]
                pass_cnt = sum(s >= args.correct_reward_threshold for s in scores)
                batch_pass_count += pass_cnt
                batch_total_scores += len(scores)
                sample.update(
                    {
                        "detailed_scores": detailed_list,
                        "scores":          scores,
                        "pass_rate":       pass_cnt / len(scores) if scores else 0.0,
                    }
                )
                rank_results[f"{batch_name}_{s_idx}"] = sample
                total_samples_with_scores += 1
            
            batch_pass_rate = batch_pass_count / batch_total_scores if batch_total_scores > 0 else 0
            total_pass_count += batch_pass_count
            
            console.print(f"📈 Batch [highlight]{batch_name}[/highlight] pass rate: {batch_pass_rate:.2%} ({batch_pass_count}/{batch_total_scores})")
        else:
            console.print(f"ℹ️  All samples in {batch_name} already scored; skipping computation")

        # persist updated batch
        with open(batch_file, "w") as f:
            json.dump(batch, f, indent=2, default=json_default)
            console.print(f"💾 Updated batch file: [highlight]{batch_file}[/highlight]")

    # write summary for this rank
    elapsed = time.time() - rank_start
    overall_pass_rate = total_pass_count / max(1, total_responses) if total_responses else 0
    
    final_path = os.path.join(rank_dir, "final_results.json")
    with open(final_path, "w") as f:
        json.dump(
            {
                "results":    rank_results,
                "errors":     {},
                "metrics": {
                    "total_time":       elapsed,
                    "num_responses":    total_responses,
                    "avg_reward_time":  elapsed / max(1, total_responses),
                    "overall_pass_rate": overall_pass_rate,
                    "total_pass_count": total_pass_count,
                    "samples_with_scores": total_samples_with_scores,
                },
            },
            f,
            indent=2,
            default=json_default,
        )
    console.print(f"✅ Saved summary to [highlight]{final_path}[/highlight]")
    console.print(f"📊 Rank directory summary for [highlight]{os.path.basename(rank_dir)}[/highlight]:")
    console.print(f"   - Total samples scored: {total_samples_with_scores}")
    console.print(f"   - Total responses processed: {total_responses}")
    console.print(f"   - Overall pass rate: {overall_pass_rate:.2%} ({total_pass_count}/{total_responses})")
    console.print(f"   - Processing time: {format_time(elapsed)}")
    
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
        # reserve fewer cores (≈12.5%) for system overhead instead of 25%
        reserved = max(1, avail // 8)
        workers = min(args.reward_workers, max(1, avail - reserved))

    console.rule("[bold]Difficulty Filter — Reward pass", style="cyan")
    console.print(f"⏰  Start : {datetime.now():%Y-%m-%d %H:%M:%S}")
    console.print(f"🖥️  CPUs  : available={avail}, using={workers}")
    console.print(f"📋 Model : {args.model_path}")
    console.print(f"📁 Output: {args.output_dir}")
    console.print(f"🎯 Correct threshold: {args.correct_reward_threshold}")
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
    all_rank_data = []
    
    for rd in rank_dirs:
        rank_start_time = time.time()
        rank_elapsed = score_rank_dir(rd, args, reward_pool, workers)
        total_elapsed += rank_elapsed
        all_rank_data.append({
            "rank_dir": os.path.basename(rd),
            "elapsed": rank_elapsed,
        })
        console.print(f"Completed [highlight]{os.path.basename(rd)}[/highlight] in {format_time(rank_elapsed)}")

    if reward_pool:
        reward_pool.close()
        reward_pool.join()

    console.rule(style="cyan")
    console.print(f"🏁 Finished scoring {len(rank_dirs)} rank directories in {format_time(total_elapsed)}")
    
    # Write global summary
    summary_path = os.path.join(root_dir, "reward_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "total_time": total_elapsed,
            "rank_data": all_rank_data,
        }, f, indent=2, default=json_default)
    console.print(f"📊 Global summary saved to [highlight]{summary_path}[/highlight]")
    console.print(f"⏰ End time: {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()
