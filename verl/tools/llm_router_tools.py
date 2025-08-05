#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elegant LLM Router Tools using LiteLLM

This module provides a clean, unified router system for multiple LLM providers:
- Single call_model() function for all models
- Rich model specifications as the single source of truth  
- Automatic tool function generation
- Cost tracking and usage statistics
- Support for OpenAI, Together AI, and Google Gemini models

Usage:
    from llm_router_tools import call_model, get_available_models, create_openai_tools
    
    # Direct model calls
    response, metadata = call_model("gpt-4o-mini", messages, {"max_tokens": 100})
    response, metadata = call_model("gemini-2.5-pro", messages, {"max_tokens": 8192})
    
    # Get OpenAI function calling tools
    tools = create_openai_tools()
"""

import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ModelSpec:
    """Comprehensive specification for an LLM model"""
    name: str
    api_alias: str
    provider: str
    input_price_per_million: float  # USD per million tokens
    output_price_per_million: float  # USD per million tokens
    context_window: int
    capabilities: List[str]  # e.g., ["reasoning", "coding", "math", "general"]
    quality_tier: str  # "budget", "standard", "premium", "specialized"
    description: str  # Comprehensive description with capabilities, cost, best use cases

@dataclass
class UsageStats:
    """Track usage statistics for a model"""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    last_used: Optional[str] = None
    
    def update(self, input_tokens: int, output_tokens: int, cost: float):
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.last_used = datetime.now().isoformat()

# Model specifications - Single source of truth
MODEL_SPECS = {
    # OpenAI Models
    "gpt-4o": ModelSpec(
        name="GPT-4o (Omni)",
        api_alias="gpt-4o",
        provider="openai",
        input_price_per_million=2.50,
        output_price_per_million=10.00,
        context_window=128000,
        capabilities=["reasoning", "coding", "math", "general", "multimodal"],
        quality_tier="premium",
        description="""Advanced multimodal model with excellent reasoning and coding capabilities.
        
        Capabilities: Advanced reasoning, coding, math, general tasks, multimodal
        Quality: Premium tier  
        Cost: $2.50/M input tokens, $10.00/M output tokens
        Context: 128K tokens
        
        Best for: Complex reasoning tasks, high-quality code generation, mathematical problems,
        multimodal applications requiring the highest quality output."""
    ),
    
    "gpt-4o-mini": ModelSpec(
        name="GPT-4o Mini",
        api_alias="gpt-4o-mini",
        provider="openai",
        input_price_per_million=0.15,
        output_price_per_million=0.60,
        context_window=128000,
        capabilities=["reasoning", "coding", "math", "general"],
        quality_tier="budget",
        description="""Cost-effective model with good performance across tasks.
        
        Capabilities: Reasoning, coding, math, general tasks
        Quality: Budget tier with good performance
        Cost: $0.15/M input tokens, $0.60/M output tokens
        Context: 128K tokens
        
        Best for: High-volume applications, cost-sensitive projects, general purpose tasks
        where premium quality is not required."""
    ),
    
    "gpt-4.1": ModelSpec(
        name="GPT-4.1",
        api_alias="gpt-4.1",
        provider="openai",
        input_price_per_million=2.00,
        output_price_per_million=8.00,
        context_window=1047576,
        capabilities=["reasoning", "coding", "math", "general", "long_context"],
        quality_tier="premium",
        description="""Latest GPT-4 variant with massive context window.
        
        Capabilities: Advanced reasoning, coding, math, general tasks, long context processing
        Quality: Premium tier
        Cost: $2.00/M input tokens, $8.00/M output tokens
        Context: 1,047,576 tokens (~1M tokens)
        
        Best for: Long document analysis, complex reasoning over large contexts,
        research tasks requiring extensive context understanding."""
    ),
    
    "gpt-4.1-mini": ModelSpec(
        name="GPT-4.1 Mini",
        api_alias="gpt-4.1-mini",
        provider="openai",
        input_price_per_million=0.40,
        output_price_per_million=1.60,
        context_window=1047576,
        capabilities=["reasoning", "coding", "math", "general", "long_context"],
        quality_tier="standard",
        description="""Balanced model with large context window and reasonable cost.
        
        Capabilities: Reasoning, coding, math, general tasks, long context processing
        Quality: Standard tier
        Cost: $0.40/M input tokens, $1.60/M output tokens
        Context: 1,047,576 tokens (~1M tokens)
        
        Best for: Medium-complexity tasks with long input contexts, balanced cost-performance
        ratio for long document processing."""
    ),
    
    "gpt-4.1-nano": ModelSpec(
        name="GPT-4.1 Nano",
        api_alias="gpt-4.1-nano",
        provider="openai",
        input_price_per_million=0.10,
        output_price_per_million=1.40,
        context_window=1047576,
        capabilities=["reasoning", "coding", "math", "general", "long_context"],
        quality_tier="budget",
        description="""Ultra-low cost model with large context window.
        
        Capabilities: Basic reasoning, coding, math, general tasks, long context processing
        Quality: Budget tier
        Cost: $0.10/M input tokens, $1.40/M output tokens
        Context: 1,047,576 tokens (~1M tokens)
        
        Best for: High-volume processing with long contexts, simple tasks over large documents,
        cost-sensitive applications requiring large context windows."""
    ),
    
    "o3": ModelSpec(
        name="o3",
        api_alias="o3",
        provider="openai",
        input_price_per_million=2.00,
        output_price_per_million=8.00,
        context_window=200000,
        capabilities=["reasoning", "math", "science"],
        quality_tier="specialized",
        description="""Advanced reasoning model optimized for mathematical and scientific tasks.
        
        Capabilities: Advanced reasoning, mathematics, scientific analysis
        Quality: Specialized for reasoning tasks
        Cost: $2.00/M input tokens, $8.00/M output tokens
        Context: 200K tokens
        
        Best for: Complex mathematical problems, scientific reasoning, research tasks
        requiring deep analytical thinking and step-by-step problem solving.
        
        Note: O-series models only support temperature=1.0 and do not support top_p.
        Other parameters will be automatically dropped."""
    ),
    
    "o3-pro": ModelSpec(
        name="o3 Pro",
        api_alias="o3-pro",
        provider="openai",
        input_price_per_million=20.00,
        output_price_per_million=80.00,
        context_window=200000,
        capabilities=["reasoning", "math", "science", "research"],
        quality_tier="specialized",
        description="""Premium reasoning model for the most challenging problems.
        
        Capabilities: Premium reasoning, advanced mathematics, scientific research
        Quality: Top-tier specialized model
        Cost: $20.00/M input tokens, $80.00/M output tokens
        Context: 200K tokens
        
        Best for: The most challenging mathematical and scientific problems,
        research requiring highest accuracy, complex analytical tasks where
        cost is less important than quality.
        
        Note: O-series models only support temperature=1.0 and do not support top_p.
        Other parameters will be automatically dropped."""
    ),
    
    "o4-mini": ModelSpec(
        name="o4 Mini",
        api_alias="o4-mini",
        provider="openai",
        input_price_per_million=1.10,
        output_price_per_million=4.40,
        context_window=200000,
        capabilities=["reasoning", "math", "science"],
        quality_tier="standard",
        description="""Balanced reasoning model with good performance-to-cost ratio.
        
        Capabilities: Reasoning, mathematics, scientific analysis
        Quality: Standard tier for reasoning
        Cost: $1.10/M input tokens, $4.40/M output tokens
        Context: 200K tokens
        
        Best for: Mathematical and scientific tasks with balanced cost-performance,
        reasoning problems that don't require the premium o3 models.
        
        Note: O-series models only support temperature=1.0 and do not support top_p.
        Other parameters will be automatically dropped."""
    ),
    
    # Together AI Models
    "qwen3-235b-instruct": ModelSpec(
        name="Qwen3 235B Instruct",
        api_alias="together_ai/Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
        provider="together",
        input_price_per_million=0.20,
        output_price_per_million=0.60,
        context_window=262144,
        capabilities=["reasoning", "coding", "math", "general", "multilingual"],
        quality_tier="premium",
        description="""Large-scale Chinese model with excellent multilingual capabilities.
        
        Capabilities: Reasoning, coding, math, general tasks, multilingual
        Quality: Premium tier with excellent multilingual support
        Cost: $0.20/M input tokens, $0.60/M output tokens
        Context: 262,144 tokens
        
        Best for: Multilingual applications, cost-effective high-quality inference,
        general purpose tasks across multiple languages and domains."""
    ),
    
    "qwen3-235b-thinking": ModelSpec(
        name="Qwen3 235B Thinking",
        api_alias="together_ai/Qwen/Qwen3-235B-A22B-Thinking-2507",
        provider="together",
        input_price_per_million=0.65,
        output_price_per_million=3.00,
        context_window=262144,
        capabilities=["reasoning", "math", "analysis", "chain_of_thought", "thinking"],
        quality_tier="specialized",
        description="""Specialized reasoning model with built-in chain-of-thought capabilities.
        
        Capabilities: Advanced reasoning, mathematics, analysis, chain-of-thought
        Quality: Specialized for reasoning with built-in CoT
        Cost: $0.65/M input tokens, $3.00/M output tokens
        Context: 262,144 tokens
        
        Best for: Complex analytical problems requiring step-by-step reasoning,
        mathematical proofs, logical analysis with explicit reasoning chains."""
    ),
    
    "qwen3-coder-480b": ModelSpec(
        name="Qwen3 Coder 480B",
        api_alias="together_ai/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        provider="together",
        input_price_per_million=2.00,
        output_price_per_million=2.00,
        context_window=262144,
        capabilities=["coding", "debugging", "code_review", "programming"],
        quality_tier="specialized",
        description="""Massive coding-specialized model with 480B parameters.
        
        Capabilities: Advanced coding, debugging, code review, programming
        Quality: Specialized coding model - largest available
        Cost: $2.00/M input tokens, $2.00/M output tokens
        Context: 262,144 tokens
        
        Best for: Complex code generation, large-scale refactoring, advanced debugging,
        architectural code reviews, systems programming tasks."""
    ),
    
    "qwen2.5-72b": ModelSpec(
        name="Qwen2.5 72B Turbo",
        api_alias="together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        provider="together",
        input_price_per_million=1.20,
        output_price_per_million=1.20,
        context_window=131072,
        capabilities=["reasoning", "coding", "math", "general"],
        quality_tier="standard",
        description="""Fast and efficient model with balanced performance.
        
        Capabilities: Reasoning, coding, math, general tasks
        Quality: Standard tier with fast inference
        Cost: $1.20/M input tokens, $1.20/M output tokens
        Context: 131,072 tokens
        
        Best for: General purpose applications requiring balanced performance and speed,
        production workloads with consistent quality needs."""
    ),
    
    "qwen2.5-7b": ModelSpec(
        name="Qwen2.5 7B Turbo",
        api_alias="together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo",
        provider="together",
        input_price_per_million=0.30,
        output_price_per_million=0.30,
        context_window=32768,
        capabilities=["reasoning", "coding", "math", "general"],
        quality_tier="budget",
        description="""Compact and fast model with good performance for simple to medium tasks.
        
        Capabilities: Basic reasoning, coding, math, general tasks
        Quality: Budget tier with good speed
        Cost: $0.30/M input tokens, $0.30/M output tokens
        Context: 32,768 tokens
        
        Best for: High-volume applications, simple tasks, rapid prototyping,
        cost-sensitive deployments with moderate quality requirements."""
    ),
    
    "qwen2.5-coder-32b": ModelSpec(
        name="Qwen2.5 Coder 32B",
        api_alias="together_ai/Qwen/Qwen2.5-Coder-32B-Instruct",
        provider="together",
        input_price_per_million=0.80,
        output_price_per_million=0.80,
        context_window=32768,
        capabilities=["coding", "debugging", "programming"],
        quality_tier="standard",
        description="""Coding-focused model with strong programming capabilities.
        
        Capabilities: Coding, debugging, programming
        Quality: Standard tier for coding tasks
        Cost: $0.80/M input tokens, $0.80/M output tokens
        Context: 32,768 tokens
        
        Best for: Standard coding tasks, code review, debugging, educational programming,
        moderate complexity software development."""
    ),
    
    "kimi-k2": ModelSpec(
        name="Kimi K2 Instruct",
        api_alias="together_ai/moonshotai/Kimi-K2-Instruct",
        provider="together",
        input_price_per_million=1.00,
        output_price_per_million=3.00,
        context_window=131072,
        capabilities=["reasoning", "math", "analysis", "multilingual"],
        quality_tier="premium",
        description="""Advanced reasoning model with strong analytical capabilities.
        
        Capabilities: Advanced reasoning, mathematics, analysis, multilingual
        Quality: Premium tier with strong analytical capabilities
        Cost: $1.00/M input tokens, $3.00/M output tokens
        Context: 131,072 tokens
        
        Best for: Complex analytical tasks, mathematical reasoning, multilingual
        applications requiring deep understanding and analysis."""
    ),
    
    "deepseek-r1": ModelSpec(
        name="DeepSeek R1",
        api_alias="together_ai/deepseek-ai/DeepSeek-R1",
        provider="together",
        input_price_per_million=3.00,
        output_price_per_million=7.00,
        context_window=163840,
        capabilities=["reasoning", "math", "science", "research"],
        quality_tier="specialized",
        description="""Research-focused model with exceptional reasoning capabilities.
        
        Capabilities: Advanced reasoning, mathematics, scientific research
        Quality: Specialized for research and reasoning
        Cost: $3.00/M input tokens, $7.00/M output tokens
        Context: 163,840 tokens
        
        Best for: Scientific research, advanced mathematical problems, complex reasoning
        tasks requiring deep analytical capabilities and research-level thinking."""
    ),
    
    "deepseek-r1-tput": ModelSpec(
        name="DeepSeek R1 Throughput",
        api_alias="together_ai/deepseek-ai/DeepSeek-R1-0528-tput",
        provider="together",
        input_price_per_million=0.55,
        output_price_per_million=2.19,
        context_window=163840,
        capabilities=["reasoning", "math", "science"],
        quality_tier="standard",
        description="""High-throughput version of DeepSeek R1 with optimized cost.
        
        Capabilities: Reasoning, mathematics, science with optimized speed
        Quality: Standard tier with better cost-performance
        Cost: $0.55/M input tokens, $2.19/M output tokens
        Context: 163,840 tokens
        
        Best for: Reasoning tasks requiring good performance at better cost,
        mathematical problems where speed and cost matter more than peak accuracy."""
    ),
    
    "deepseek-v3": ModelSpec(
        name="DeepSeek V3",
        api_alias="together_ai/deepseek-ai/DeepSeek-V3",
        provider="together",
        input_price_per_million=1.25,
        output_price_per_million=1.25,
        context_window=131072,
        capabilities=["reasoning", "coding", "math", "general"],
        quality_tier="standard",
        description="""Balanced model with strong performance across multiple domains.
        
        Capabilities: Reasoning, coding, mathematics, general tasks
        Quality: Standard tier with balanced performance
        Cost: $1.25/M input tokens, $1.25/M output tokens
        Context: 131,072 tokens
        
        Best for: Balanced applications requiring both reasoning and coding,
        general purpose tasks with consistent quality across multiple domains."""
    ),
    
    # Google Gemini Models
    "gemini-2.5-pro": ModelSpec(
        name="Gemini 2.5 Pro",
        api_alias="gemini/gemini-2.5-pro",
        provider="google",
        input_price_per_million=1.25,  # ≤200K tokens, $2.50 for >200K
        output_price_per_million=10.00,  # ≤200K tokens, $15.00 for >200K
        context_window=1048576,  # 1M tokens
        capabilities=["reasoning", "coding", "math", "general", "multimodal", "long_context", "thinking"],
        quality_tier="premium",
        description="""Google's most advanced model with thinking capabilities and massive context.
        
        Capabilities: Advanced reasoning, coding, math, general, multimodal, long context, thinking
        Quality: Premium tier with thinking capabilities
        Cost: $1.25/M input tokens, $10.00/M output tokens (≤200K), higher for >200K
        Context: 1,048,576 tokens (~1M tokens)
        
        Best for: Complex reasoning tasks, long document analysis, multimodal applications,
        thinking-enabled tasks requiring the highest quality and massive context.
        
        Note: Thinking model requires high token limits (8192+) to account for internal reasoning."""
    ),
    
    "gemini-2.5-flash": ModelSpec(
        name="Gemini 2.5 Flash",
        api_alias="gemini/gemini-2.5-flash",
        provider="google",
        input_price_per_million=0.30,
        output_price_per_million=2.50,
        context_window=1048576,  # 1M tokens
        capabilities=["reasoning", "coding", "math", "general", "multimodal", "long_context", "hybrid_reasoning"],
        quality_tier="standard",
        description="""Fast and balanced Gemini model with hybrid reasoning capabilities.
        
        Capabilities: Reasoning, coding, math, general, multimodal, long context, hybrid reasoning
        Quality: Standard tier with balanced performance
        Cost: $0.30/M input tokens, $2.50/M output tokens
        Context: 1,048,576 tokens (~1M tokens)
        
        Best for: Balanced applications requiring good performance with large context,
        hybrid reasoning tasks, general purpose with cost efficiency."""
    ),
    
    "gemini-2.5-flash-lite": ModelSpec(
        name="Gemini 2.5 Flash Lite",
        api_alias="gemini/gemini-2.5-flash-lite",
        provider="google",
        input_price_per_million=0.10,
        output_price_per_million=0.40,
        context_window=1048576,  # 1M tokens
        capabilities=["reasoning", "coding", "math", "general", "multimodal", "long_context", "high_throughput"],
        quality_tier="budget",
        description="""High-throughput Gemini model optimized for cost-efficiency.
        
        Capabilities: Reasoning, coding, math, general, multimodal, long context, high throughput
        Quality: Budget tier optimized for cost and throughput
        Cost: $0.10/M input tokens, $0.40/M output tokens
        Context: 1,048,576 tokens (~1M tokens)
        
        Best for: High-volume applications, cost-sensitive projects requiring large context,
        high-throughput processing with good performance."""
    ),
}

class LLMRouter:
    """Unified router system for managing multiple LLM models"""
    
    def __init__(self):
        self.usage_stats = {model_id: UsageStats() for model_id in MODEL_SPECS.keys()}
        self._lock = threading.Lock()
        
        # Set up LiteLLM
        litellm.set_verbose = False  # Reduce logging noise
        
        # Verify API keys
        self._check_api_keys()
    
    def _check_api_keys(self):
        """Verify that necessary API keys are available"""
        openai_key = os.getenv("OPENAI_API_KEY")
        together_key = os.getenv("TOGETHER_API_KEY")
        google_key = os.getenv("GEMINI_API_KEY")
        
        if not openai_key:
            print("WARNING: OPENAI_API_KEY not found. OpenAI models will not work.")
        else:
            print("✓ OPENAI_API_KEY found")
        
        if not together_key:
            print("WARNING: TOGETHER_API_KEY not found. Together AI models will not work.")
        else:
            print("✓ TOGETHER_API_KEY found")
            
        if not google_key:
            print("WARNING: GEMINI_API_KEY not found. Google Gemini models will not work.")
        else:
            print("✓ GEMINI_API_KEY found")
            
        if not (openai_key or together_key or google_key):
            raise ValueError("At least one API key (OPENAI_API_KEY, TOGETHER_API_KEY, or GEMINI_API_KEY) must be set")
    
    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Rough estimation of token count for cost calculation"""
        text = " ".join([msg.get("content", "") for msg in messages])
        return len(text) // 4  # Rough approximation: 4 chars per token
    
    def _calculate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for a model call"""
        spec = MODEL_SPECS[model_id]
        input_cost = (input_tokens / 1_000_000) * spec.input_price_per_million
        output_cost = (output_tokens / 1_000_000) * spec.output_price_per_million
        return input_cost + output_cost
    
    def call_model(self, model_id: str, messages: List[Dict[str, str]], 
                   sampling_params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Call any model by ID with unified interface.
        
        Args:
            model_id: Model identifier from MODEL_SPECS
            messages: List of message dicts with 'role' and 'content' keys
            sampling_params: Optional dict with 'temperature', 'max_tokens', 'top_p'
            
        Returns:
            Tuple of (response_content, metadata_dict)
        """
        if model_id not in MODEL_SPECS:
            raise ValueError(f"Unknown model_id '{model_id}'. Available: {list(MODEL_SPECS.keys())}")
            
        spec = MODEL_SPECS[model_id]
        
        # Prepare parameters with intelligent defaults
        default_max_tokens = 2048
        
        # Special handling for thinking models (need MUCH more tokens for internal reasoning)
        if "thinking" in spec.capabilities or "gemini-2.5-pro" in spec.api_alias:
            default_max_tokens = 8192  # Very high token limit to account for extensive internal reasoning
        
        params = {
            "model": spec.api_alias,
            "messages": messages,
            "temperature": sampling_params.get("temperature", 1.0) if sampling_params else 1.0,
            "max_tokens": sampling_params.get("max_tokens", default_max_tokens) if sampling_params else default_max_tokens,
            "drop_params": True,  # for litellm, always drop params that are not supported by the model
        }
        
        # Only add top_p for models that support it (not o3/o4 models)
        if not spec.api_alias.startswith("o3") and not spec.api_alias.startswith("o4"):
            params["top_p"] = sampling_params.get("top_p", 1.0) if sampling_params else 1.0
        
        # Add provider-specific parameters
        if spec.provider == "together":
            os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY", "")
        elif spec.provider == "google":
            # Ensure Google API key is available for Gemini models
            if os.getenv("GEMINI_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        
        start_time = time.time()
        
        try:
            response = litellm.completion(**params)
            
            # Extract response details
            content = response.choices[0].message.content
            usage = response.usage
            
            # Handle thinking models and empty responses
            if content is None:
                # Check if this is a thinking model with reasoning content
                message = response.choices[0].message
                thinking_content = getattr(message, 'reasoning_content', None)
                
                if thinking_content:
                    # For thinking models, use reasoning content if available
                    content = f"[Thinking: {thinking_content}]"
                elif hasattr(usage, 'completion_tokens_details') and hasattr(usage.completion_tokens_details, 'reasoning_tokens'):
                    # Model used reasoning tokens but no visible output
                    reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
                    if reasoning_tokens > 0:
                        content = f"[Thinking model used {reasoning_tokens} reasoning tokens but no visible output. Use max_tokens=8192+ to allow space for both thinking and response.]"
                    else:
                        content = "[Empty response - thinking model may need different parameters or much higher token limits (8192+)]"
                else:
                    content = "[Empty response - model returned None content. For Gemini Pro, try max_tokens=8192+ to account for internal reasoning.]"
            
            # Get token counts
            input_tokens = getattr(usage, 'prompt_tokens', self._estimate_tokens(messages))
            output_tokens = getattr(usage, 'completion_tokens', len(content) // 4)
            
            # Calculate cost and update stats
            cost = self._calculate_cost(model_id, input_tokens, output_tokens)
            
            with self._lock:
                self.usage_stats[model_id].update(input_tokens, output_tokens, cost)
            
            metadata = {
                "model_id": model_id,
                "model_name": spec.name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
                "latency": time.time() - start_time,
                "provider": spec.provider
            }
            
            return content, metadata
            
        except Exception as e:
            raise RuntimeError(f"Error calling {spec.name}: {str(e)}")

# Create global router instance
router = LLMRouter()

def call_model(model_id: str, messages: List[Dict[str, str]], 
               sampling_params: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Unified function to call any model by ID.
    
    Args:
        model_id: Model identifier from MODEL_SPECS
        messages: List of message dicts with 'role' and 'content' keys
        sampling_params: Optional dict with 'temperature', 'max_tokens', 'top_p'
        
    Returns:
        Tuple of (response_content, metadata_dict)
    """
    return router.call_model(model_id, messages, sampling_params)

def get_available_models() -> Dict[str, ModelSpec]:
    """Get all available model specifications"""
    return MODEL_SPECS.copy()

def get_models_by_capability(capability: str) -> Dict[str, ModelSpec]:
    """Get models that have a specific capability"""
    return {
        model_id: spec for model_id, spec in MODEL_SPECS.items()
        if capability in spec.capabilities
    }

def get_models_by_tier(tier: str) -> Dict[str, ModelSpec]:
    """Get models in a specific quality tier"""
    return {
        model_id: spec for model_id, spec in MODEL_SPECS.items()
        if spec.quality_tier == tier
    }

def get_models_by_provider(provider: str) -> Dict[str, ModelSpec]:
    """Get models from a specific provider"""
    return {
        model_id: spec for model_id, spec in MODEL_SPECS.items()
        if spec.provider == provider
    }

# MODEL_FUNCTIONS removed - use call_model() directly


def get_usage_stats() -> Dict[str, Dict[str, Any]]:
    """Get usage statistics for all models"""
    return {
        model_id: asdict(stats) for model_id, stats in router.usage_stats.items()
        if stats.total_calls > 0
    }

def print_usage_summary():
    """Print a summary of model usage"""
    stats = get_usage_stats()
    if not stats:
        print("No usage recorded yet.")
        return
    
    print("\n=== LLM Usage Summary ===")
    total_cost = sum(s["total_cost"] for s in stats.values())
    total_calls = sum(s["total_calls"] for s in stats.values())
    
    print(f"Total calls: {total_calls}")
    print(f"Total cost: ${total_cost:.4f}")
    print("\nPer-model breakdown:")
    
    for model_id, stat in sorted(stats.items(), key=lambda x: x[1]["total_cost"], reverse=True):
        spec = MODEL_SPECS[model_id]
        print(f"\n{spec.name} ({model_id}):")
        print(f"  Calls: {stat['total_calls']}")
        print(f"  Input tokens: {stat['total_input_tokens']:,}")
        print(f"  Output tokens: {stat['total_output_tokens']:,}")
        print(f"  Cost: ${stat['total_cost']:.4f}")
        if stat['last_used']:
            print(f"  Last used: {stat['last_used']}")

if __name__ == "__main__":
    # Simple test to demonstrate the unified API
    print("=== Unified LLM Router Test ===\n")
    
    # Show available models
    print(f"Available models: {len(MODEL_SPECS)}")
    for provider in ["openai", "together", "google"]:
        models = get_models_by_provider(provider)
        print(f"  {provider}: {len(models)} models")
    
    # Show model capabilities
    print(f"\nBy capability:")
    for capability in ["reasoning", "coding", "math", "thinking"]:
        models = get_models_by_capability(capability)
        print(f"  {capability}: {len(models)} models")
    
    # Example usage (commented out to avoid actual API calls)
    # print(f"\n--- Example Usage ---")
    # response, metadata = call_model("gpt-4o-mini", [{"role": "user", "content": "What is 2+2?"}])
    # print(f"Response: {response}")
    # print(f"Cost: ${metadata['cost']:.6f}")
    
    print(f"\n=== Router Ready ===")
    print("Use call_model(model_id, messages, sampling_params) for any model!")