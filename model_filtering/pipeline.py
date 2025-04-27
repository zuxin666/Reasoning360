#!/usr/bin/env python3
import os
import json
import pickle
import time
from datetime import datetime, timedelta
import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from rich.panel import Panel
from multiprocessing import Pool
import glob

from model_filtering.utils import console, json_default
from verl.utils.reward_score import _default_compute_score

class DifficultyFilterPipeline:
    def __init__(self, args):
        self.args = args            
        self.tokenizer = None
        self.model = None
        self.sampling_params = None

        # Parallel reward workers
        self.reward_pool = Pool(processes=args.reward_workers) if args.reward_workers > 1 else None

        # Timing
        self.start_time = time.time()
        self.gen_times = []
        self.reward_times = []

    # ------------- misc helpers -------------------------------------------- #
    def __del__(self):
        if self.reward_pool:
            self.reward_pool.close()
            self.reward_pool.join()

    @staticmethod
    def format_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    # ------------- component init ------------------------------------------ #
    def initialize_components(self):
        console.print(f"🔄 Loading tokenizer from [highlight]{self.args.model_path}[/highlight]...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_path)

        console.print(
            f"🚀 Initializing model from [highlight]{self.args.model_path}[/highlight] "
            f"with TP={self.args.tp_size}..."
        )
        self.model = LLM(
            self.args.model_path,
            tensor_parallel_size=self.args.tp_size,
            enforce_eager=True,
        )

        self.sampling_params = SamplingParams(
            n=self.args.n,
            max_tokens=self.args.max_new_tokens,
            top_k=self.args.top_k,
            top_p=self.args.top_p,
            temperature=self.args.temperature,
            repetition_penalty=self.args.repetition_penalty,
            skip_special_tokens=True,
        )
        console.print("✅ Model initialization [success]complete[/success].")

    # ------------- dataset -------------------------------------------------- #
    def prepare_dataset(self):
        console.print(f"📂 Loading dataset from [highlight]{self.args.dataset_parquet_path}[/highlight]...")
        dataset = Dataset.from_parquet(self.args.dataset_parquet_path)

        # ── debug slice
        if self.args.debug:
            dataset = dataset.select(range(min(48, len(dataset))))
            console.print(f"🐞 [DEBUG] Using first [highlight]{len(dataset)}[/highlight] samples")

        # ── DP split
        if self.args.dp_size > 1:
            total = len(dataset)
            per_rank = total // self.args.dp_size
            start = self.args.dp_rank * per_rank
            end = start + per_rank if self.args.dp_rank != self.args.dp_size - 1 else total
            dataset = dataset.select(range(start, end))
            console.print(
                f"🔢 DP rank [highlight]{self.args.dp_rank}[/highlight] "
                f"processing [highlight]{len(dataset)}[/highlight] / {total}"
            )
        else:
            console.print(f"📊 Dataset loaded with [highlight]{len(dataset)}[/highlight] samples")

        # ── KEEP-ONLY columns actually referenced downstream ---------------- #
        required_cols = {
            "prompt",
            "reward_model",
            "apply_chat_template",
            "data_source",
            "extra_info",
        }
        cols_to_drop = [c for c in dataset.column_names if c not in required_cols]
        if cols_to_drop:
            dataset = dataset.remove_columns(cols_to_drop)
            console.print(f"🧹 Dropped {len(cols_to_drop)} column(s) for easier processing: {', '.join(cols_to_drop)}")

        return dataset

    # ------------- checkpoint paths / I-O ------------------------------ #
    def get_checkpoint_path(self):
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name, f"dp{self.args.dp_rank}")
        os.makedirs(rank_output_dir, exist_ok=True)
        return os.path.join(rank_output_dir, "checkpoint.pkl")

    def load_checkpoint(self):
        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name, f"dp{self.args.dp_rank}")
        os.makedirs(rank_output_dir, exist_ok=True)
        
        # If force_regenerate is set, start from scratch
        if self.args.force_regenerate:
            console.print("🔄 [warning]Force regeneration requested - ignoring existing checkpoints[/warning]")
            return {"current_batch_idx": 0, "global_results": {}, "global_errors": {}}
        
        # Find batch files and extract their numbers
        batch_files = glob.glob(os.path.join(rank_output_dir, "batch_*.json"))
        batch_files_with_idx = []
        
        for batch_file in batch_files:
            try:
                batch_num = int(os.path.basename(batch_file).split("_")[1].split(".")[0])
                batch_files_with_idx.append((batch_num, batch_file))
            except (ValueError, IndexError):
                console.print(f"⚠️ [warning]Could not parse batch number from {batch_file}[/warning]")
        
        # Sort by batch number
        batch_files_with_idx.sort(key=lambda x: x[0])
        
        # Load results from existing batch files
        global_results = {}
        global_errors = {}
        
        if batch_files_with_idx:
            console.print(f"📋 Found {len(batch_files_with_idx)} existing batch files to load")
            
            for batch_idx, batch_file in batch_files_with_idx:
                try:
                    with open(batch_file, 'r') as f:
                        batch_data = json.load(f)
                        
                    # If recalculate_rewards is set, we'll reload the responses but recalculate scores later
                    if self.args.recalculate_rewards:
                        for i, result in batch_data.items():
                            # Keep everything except scores
                            global_results[f"{batch_idx}_{i}"] = {
                                "messages": result["messages"],
                                "question": result["question"],
                                "ground_truth": result["ground_truth"],
                                "source": result["source"],
                                "responses": result["responses"],
                                "extra_info": result["extra_info"],
                                "needs_recalculation": True
                            }
                        console.print(f"🔄 Loaded batch {batch_idx} with {len(batch_data)} results (scores will be recalculated)")
                    else:
                        # Add batch results to global results with proper keys
                        for i, result in batch_data.items():
                            global_results[f"{batch_idx}_{i}"] = result
                        
                        console.print(f"✅ Loaded batch {batch_idx} with {len(batch_data)} results")
                except Exception as e:
                    console.print(f"⚠️ [warning]Failed to load batch file {batch_file}: {e}[/warning]")
        
        # Determine the next batch index to start from
        start_batch_idx = batch_files_with_idx[-1][0] + 1 if batch_files_with_idx else 0
        
        if batch_files_with_idx:
            console.print(f"📋 Resuming from batch {start_batch_idx} with {len(global_results)} loaded results")
        
        return {"current_batch_idx": start_batch_idx, "global_results": global_results, "global_errors": global_errors}

    # ------------- message helpers ----------------------------------------- #
    def extract_messages(self, batch_dict, index):
        msgs = []
        if "prompt" in batch_dict:
            for prompt_dict in batch_dict["prompt"]:
                if index < len(prompt_dict["role"]) and index < len(prompt_dict["content"]):
                    msgs.append(
                        {
                            "role": prompt_dict["role"][index],
                            "content": prompt_dict["content"][index],
                        }
                    )
        return msgs

    # ------------- reward / batch processing  ------------------------------ #
    @staticmethod
    def compute_single_reward(args):
        response, data_source, ground_truth, extra_info = args
        try:
            result = _default_compute_score(
                data_source=data_source,
                solution_str=response,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            
            if isinstance(result, dict):
                return result
            elif isinstance(result, float):
                return {"score": result}
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")
        except Exception as e:
            return {"score": 0.0, "error": str(e)}

    def process_batch_outputs(self, outputs, batch_dict):
        batch_results = {}
        reward_start = time.time()

        reward_pbar = tqdm(total=len(outputs), desc="💯 Computing rewards", leave=False, position=1)
        for i in range(len(outputs)):
            # Get the full original responses
            full_responses = [r.text for r in outputs[i].outputs]
            
            # Process responses to extract content after </think> tag if present
            processed_responses = []
            for resp in full_responses:
                if "</think>" in resp:
                    processed_responses.append(resp.split("</think>", 1)[1])
                else:
                    processed_responses.append(resp)
            
            data_source = batch_dict["data_source"][i]
            ground_truth = batch_dict["reward_model"]["ground_truth"][i]
            messages = self.extract_messages(batch_dict, i)
            question = next((m["content"] for m in messages if m["role"] == "user"), "No question found")
            extra_info = {k: batch_dict["extra_info"][k][i] for k in batch_dict["extra_info"]}

            # Use processed responses (after </think>) for reward calculation
            compute_args = [(r, data_source, ground_truth, extra_info) for r in processed_responses]
            if self.reward_pool:
                detailed = self.reward_pool.map(self.compute_single_reward, compute_args)
            else:
                detailed = [self.compute_single_reward(a) for a in compute_args]
            
            scores = [d["score"] for d in detailed]
            
            pass_cnt = sum(s >= self.args.correct_reward_threshold for s in scores)
            batch_results[i] = {
                "messages": messages,
                "question": question,
                "ground_truth": ground_truth,
                "source": data_source,
                "responses": full_responses,  # Save full responses
                "scores": scores,
                "detailed_scores": detailed,
                "pass_rate": pass_cnt / len(processed_responses) if processed_responses else 0.0,
                "extra_info": extra_info,
            }
            reward_pbar.update(1)
        reward_pbar.close()

        self.reward_times.append(time.time() - reward_start)
        return batch_results

    # ------------- progress ------------------------------------------------- #
    def print_progress_stats(self, idx, total_batches):
        elapsed = time.time() - self.start_time
        eta = "calculating..."
        if idx > 0 and self.gen_times and self.reward_times:
            remain = total_batches - idx - 1
            eta_batch = max(np.mean(self.gen_times[-10:]), np.mean(self.reward_times[-10:]))
            eta = self.format_time(remain * eta_batch)

        console.print()
        console.print(
            Panel(
                f"[bold]Progress: [metric]{idx+1}/{total_batches}[/metric] "
                f"({(idx+1)/total_batches*100:.1f}%)[/bold]",
                title="📊 Summary",
                border_style="cyan",
            )
        )
        console.print(f"⏱️  Elapsed: [time]{self.format_time(elapsed)}[/time] | ETA: [time]{eta}[/time]")
        if self.gen_times:
            console.print(f"⚡ Generation avg (last 10): [metric]{np.mean(self.gen_times[-10:]):.2f}s[/metric]")
        if self.reward_times:
            console.print(f"🧮 Reward avg (last 10): [metric]{np.mean(self.reward_times[-10:]):.2f}s[/metric]")

    # ------------- main loop ------------------------------------------------ #
    def run_inference(self):
        self.initialize_components()
        dataset = self.prepare_dataset()

        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            pin_memory=True,
            num_workers=min(4, os.cpu_count() // max(1, self.args.dp_size)),
            prefetch_factor=8,
        )

        chk = self.load_checkpoint()
        start_batch_idx = chk["current_batch_idx"]
        global_results = chk["global_results"]
        global_errors = chk["global_errors"]

        model_name = self.args.model_path.split("/")[-1]
        dataset_name = os.path.basename(self.args.dataset_parquet_path).rsplit(".parquet", 1)[0]
        rank_output_dir = os.path.join(self.args.output_dir, dataset_name, model_name, f"dp{self.args.dp_rank}")
        os.makedirs(rank_output_dir, exist_ok=True)

        # Check if we need to recalculate rewards for existing results
        if self.args.recalculate_rewards and global_results:
            console.print("\n🔄 Recalculating rewards for existing results...")
            items_to_recalculate = [(k, v) for k, v in global_results.items() if v.get("needs_recalculation", False)]
            
            if items_to_recalculate:
                recalculation_pbar = tqdm(
                    total=len(items_to_recalculate),
                    desc="🔄 Recalculating rewards",
                    position=0
                )
                
                for key, result in items_to_recalculate:
                    batch_idx, sample_idx = key.split("_")
                    batch_idx, sample_idx = int(batch_idx), int(sample_idx)
                    
                    # Compute rewards for this sample
                    data_source = result["source"]
                    ground_truth = result["ground_truth"]
                    responses = result["responses"]
                    extra_info = result["extra_info"]
                    
                    # Process responses to extract content after </think> tag if present
                    processed_responses = []
                    for resp in responses:
                        if "</think>" in resp:
                            processed_responses.append(resp.split("</think>", 1)[1])
                        else:
                            processed_responses.append(resp)
                    
                    compute_args = [(r, data_source, ground_truth, extra_info) for r in processed_responses]
                    if self.reward_pool:
                        detailed = self.reward_pool.map(self.compute_single_reward, compute_args)
                    else:
                        detailed = [self.compute_single_reward(a) for a in compute_args]
                    
                    scores = [d["score"] for d in detailed]
                    pass_cnt = sum(s >= self.args.correct_reward_threshold for s in scores)
                    
                    # Update the result
                    result["scores"] = scores
                    result["detailed_scores"] = detailed
                    result["pass_rate"] = pass_cnt / len(processed_responses) if processed_responses else 0.0
                    result.pop("needs_recalculation", None)
                    
                    # Save the updated batch file
                    batch_results = {}
                    for i, r in global_results.items():
                        curr_batch_idx = int(i.split("_")[0])
                        curr_sample_idx = int(i.split("_")[1])
                        if curr_batch_idx == batch_idx:
                            batch_results[curr_sample_idx] = r
                    
                    batch_out = os.path.join(rank_output_dir, f"batch_{batch_idx:05d}.json")
                    with open(batch_out, "w") as f:
                        json.dump(batch_results, f, indent=2, default=json_default)
                    
                    recalculation_pbar.update(1)
                
                recalculation_pbar.close()
                console.print(f"✅ Recalculated rewards for {len(items_to_recalculate)} samples")
            else:
                console.print("ℹ️ No items found that need reward recalculation")

        progress_bar = tqdm(
            total=len(dataloader),
            initial=start_batch_idx,
            desc=f"🔄 DP{self.args.dp_rank} Processing",
            position=0,
        )

        batch_iter = iter(dataloader)
        for _ in range(start_batch_idx):
            try:
                next(batch_iter)
            except StopIteration:
                console.print(f"⚠️ [warning]Checkpoint index {start_batch_idx} exceeds dataset size[/warning]")
                break

        for idx, batch_dict in enumerate(batch_iter, start=start_batch_idx):
            batch_size = len(batch_dict["reward_model"]["ground_truth"])
            console.print(f"\n🔄 Generating for batch {idx}/{len(dataloader)-1} ({batch_size} samples)…")

            # enforce apply_chat_template==True
            if not all(batch_dict.get("apply_chat_template", [True] * batch_size)):
                raise RuntimeError(
                    "Encountered apply_chat_template=False but raw_prompt column is removed. "
                    "Please ensure all samples set apply_chat_template=True."
                )

            gen_start = time.time()
            outputs = self.model.chat(
                [self.extract_messages(batch_dict, i) for i in range(batch_size)],
                sampling_params=self.sampling_params,
                use_tqdm=False,
            )
            self.gen_times.append(time.time() - gen_start)
            console.print(f"⏱️  Generation took [time]{self.gen_times[-1]:.2f}s[/time]")

            batch_results = self.process_batch_outputs(outputs, batch_dict)
            avg_pass = np.mean([r["pass_rate"] for r in batch_results.values()])
            console.print(f"✅ Average pass rate: [success]{avg_pass:.2f}[/success]")

            global_results.update({f"{idx}_{i}": r for i, r in batch_results.items()})

            batch_out = os.path.join(rank_output_dir, f"batch_{idx:05d}.json")
            with open(batch_out, "w") as f:
                json.dump(batch_results, f, indent=2, default=json_default)
            console.print(f"💾 Saved batch results to [highlight]{batch_out}[/highlight]")

            self.print_progress_stats(idx, len(dataloader))
            progress_bar.update(1)
            console.rule(style="cyan")

        progress_bar.close()
        elapsed_total = time.time() - self.start_time

        console.print()
        console.print(Panel("[bold]Inference completed!", title="🏁 Finished", border_style="green"))
        console.print(f"⏱️  Total time: [time]{self.format_time(elapsed_total)}[/time]")

        final_path = os.path.join(rank_output_dir, "final_results.json")
        with open(final_path, "w") as f:
            json.dump(
                {
                    "results": global_results,
                    "errors": global_errors,
                    "metrics": {
                        "total_time": elapsed_total,
                        "avg_gen_time": np.mean(self.gen_times) if self.gen_times else 0.0,
                        "avg_reward_time": np.mean(self.reward_times) if self.reward_times else 0.0,
                    },
                },
                f,
                indent=2,
                default=json_default,
            )
        console.print(f"💾 Final results saved to [highlight]{final_path}[/highlight]")