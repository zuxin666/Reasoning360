import os
import asyncio
from openai import AsyncOpenAI
import pandas as pd
import json
import time
from collections import deque
from typing import Deque, Optional
import logging
from tqdm.asyncio import tqdm_asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hyperparameters
TIMEOUT_SECONDS = 24000  # 400 minutes
MAX_WINDOW_SIZE = 600
NUM_GENERATIONS = 16


class RateLimiter:
    def __init__(self, max_window_size: int):
        self.max_window_size = max_window_size
        self.semaphore = asyncio.Semaphore(max_window_size)
    
    async def acquire(self):
        await self.semaphore.acquire()
        logger.debug(f"Acquired slot. Available slots: {self.semaphore._value}")
    
    async def release(self):
        self.semaphore.release()
        logger.debug(f"Released slot. Available slots: {self.semaphore._value}")

llm_client = AsyncOpenAI(
            base_url="http://10.24.2.67:8000/v1",
            api_key="token-abc123",
            timeout=TIMEOUT_SECONDS,
)

async def single_call(inputs, model="Qwen/Qwen3-32B", timeout=None):
    try:
        start_time = time.time()
        response = await llm_client.chat.completions.create(
            model=model, 
            messages=inputs,
            timeout=timeout or TIMEOUT_SECONDS
        )
        duration = time.time() - start_time
        
        if duration > TIMEOUT_SECONDS * 0.9:  # If request took more than 90% of timeout
            logger.warning(f"Request took {duration:.2f}s (close to timeout {TIMEOUT_SECONDS}s)")
            
        return response.choices[0].message.content
    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {TIMEOUT_SECONDS}s")
        return None
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return None

async def process_all_inputs(inputs_list, num_generations=1, max_window_size=MAX_WINDOW_SIZE, timeout=None):
    # Initialize rate limiter
    rate_limiter = RateLimiter(max_window_size)
    
    try:
        # Create N tasks for each input
        all_tasks = []
        for inputs in inputs_list:
            tasks = []
            for _ in range(num_generations):
                async def rate_limited_call(inputs):
                    await rate_limiter.acquire()
                    try:
                        return await single_call(inputs, timeout=timeout)
                    finally:
                        await rate_limiter.release()
                tasks.append(rate_limited_call(inputs))
            all_tasks.extend(tasks)
        
        # Run all tasks concurrently with progress bar
        responses = await tqdm_asyncio.gather(*all_tasks, desc="Processing requests")
        
        # Group responses by input
        grouped_responses = []
        for i in range(0, len(responses), num_generations):
            grouped_responses.append(responses[i:i + num_generations])
        
        return grouped_responses
    finally:
        pass  # No need to stop the rate limiter as it's managed by the acquire/release calls


def main():
    """
        Make the input a list of openai chat completion messages.
    """
    def preprocess_input(instance):
        messages = instance["messages"]
        if messages[0]["role"] == "system":
            messages = messages[1:]
        return messages
    
    # Read input data
    data_df = pd.read_json("../libra_eval/datasets/iq_250.jsonl", lines=True)

    # Convert inputs to list for async processing
    inputs_list = data_df.apply(preprocess_input, axis=1).tolist()

    # Process all inputs asynchronously with rate limiting
    responses = asyncio.run(process_all_inputs(
        inputs_list, 
        num_generations=NUM_GENERATIONS,
        max_window_size=MAX_WINDOW_SIZE,
        timeout=TIMEOUT_SECONDS
    ))

    # Add responses to the DataFrame
    data_df['responses'] = responses
    
    # Save to JSONL file
    output_file = "llm_responses.jsonl"
    data_df.to_json(output_file, orient='records', lines=True)

    # Calculate success rate
    successful_generations = sum(len([r for r in resp if r is not None]) for resp in responses)
    total_expected_generations = len(inputs_list) * NUM_GENERATIONS

    print(f"Responses saved to {output_file}")
    print(f"Processed {successful_generations} out of {total_expected_generations} generations successfully")


if __name__ == "__main__":
    main()
