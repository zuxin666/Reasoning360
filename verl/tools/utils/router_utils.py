# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from openai import AsyncOpenAI


async def call_gpt_api(
    system_prompt: str,
    user_prompt: str,
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
    timeout: float = 30.0
) -> Tuple[str, int]:
    """
    Call GPT API with the given prompts and parameters.
    
    Args:
        system_prompt: The system-level prompt
        user_prompt: The user prompt/question
        model: The model to use (default: gpt-4o)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (response_text, tokens_used)
        
    Raises:
        Exception: If API call fails
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY env var not set")
    
    client = AsyncOpenAI(
        api_key=api_key,
        timeout=timeout
    )
    
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response_text = response.choices[0].message.content
        tokens_used = response.usage.total_tokens
        
        return response_text, tokens_used
        
    except Exception as e:
        raise Exception(f"GPT-4o API call failed: {str(e)}")


async def simulate_user_response(
    content: str,
    conversation_history: List[Dict[str, Any]],
    model: str = "gpt-4o",
    api_key: Optional[str] = None
) -> str:
    """
    Simulate a user response using GPT-4o based on the content and conversation history.
    
    Args:
        content: The content/question presented to the user
        conversation_history: Previous interactions in this conversation (list of dicts with route_to, arguments, feedback)
        model: The model to use for simulation
        api_key: OpenAI API key
        
    Returns:
        Simulated user response
    """
    # Build context from conversation history
    history_context = ""
    if conversation_history:
        recent_interactions = conversation_history[-3:]  # Last 3 interactions for context
        for i, interaction in enumerate(recent_interactions):
            history_context += f"Previous interaction {i+1}:\n"
            history_context += f"Route: {interaction['route_to']}\n"
            history_context += f"Feedback: {interaction['feedback']}\n\n"
    
    system_prompt = """You are simulating a helpful user who is engaged in a conversation with an AI system. 
The system is presenting you with content and asking for your input, clarification, or response.
You should respond naturally and helpfully as a real user would.

Consider the conversation history and provide responses that are:
- Contextually appropriate
- Helpful and constructive
- Realistic for a human user
- Concise but informative

If the system is asking for clarification, provide it.
If the system is asking for confirmation, give a reasonable yes/no with brief explanation.
If the system is asking for your opinion or judgment, provide a thoughtful response."""

    user_prompt = f"""Based on the conversation history below, please respond to the current content as a user would.

Conversation History:
{history_context}

Current content from system:
{content}

Please provide a natural user response:"""

    try:
        response, _ = await call_gpt4o_api(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=0.8,  # Slightly higher temperature for more natural responses
            max_tokens=200,   # Keep responses concise
            api_key=api_key
        )
        return response.strip()
    except Exception as e:
        # Fallback to simple response if simulation fails
        return f"I understand. Could you provide more details about: {content[:100]}{'...' if len(content) > 100 else ''}?"


async def call_claude_api(
    system_prompt: str,
    user_prompt: str,
    model: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    region_name: str = "us-west-2",
    timeout: float = 30.0
) -> Tuple[str, int]:
    """
    Call Claude API via AWS Bedrock with the given prompts and parameters.
    
    Args:
        system_prompt: The system-level prompt
        user_prompt: The user prompt/question
        model: The Claude model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        region_name: AWS region for Bedrock
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (response_text, tokens_used)
        
    Raises:
        Exception: If API call fails
    """
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=region_name)
        
        # Format messages for Bedrock
        system_message = [{"text": system_prompt}]
        messages = [{"role": "user", "content": [{"text": user_prompt}]}]
        
        response = client.converse(
            modelId=model,
            messages=messages,
            system=system_message,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        )
        
        # Extract response text
        response_text = response["output"]["message"]["content"][0]["text"]
        tokens_used = response["usage"]["inputTokens"] + response["usage"]["outputTokens"]
        
        return response_text, tokens_used
        
    except Exception as e:
        raise Exception(f"Claude API call failed: {str(e)}")


async def call_gemini_api(
    system_prompt: str,
    user_prompt: str,
    model: str = "gemini-1.5-pro",
    temperature: float = 0.7,
    max_tokens: int = 1000,
    api_key: Optional[str] = None,
    timeout: float = 30.0
) -> Tuple[str, int]:
    """
    Call Gemini API with the given prompts and parameters.
    
    Args:
        system_prompt: The system-level prompt
        user_prompt: The user prompt/question
        model: The Gemini model to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        api_key: Google API key (if None, uses GENAI_API_KEY env var)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (response_text, tokens_used)
        
    Raises:
        Exception: If API call fails
    """
    if api_key is None:
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise ValueError("Google API key not provided and GENAI_API_KEY env var not set")
    
    try:
        from google.genai import Client, types
        client = Client(api_key=api_key)
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        
        # Combine system and user prompts
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        contents = [types.Content(role="user", parts=[types.Part(text=combined_prompt)])]
        
        response = await client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )
        
        response_text = response.candidates[0].content.parts[0].text
        # Gemini doesn't provide detailed token usage in basic calls
        tokens_used = len(combined_prompt.split()) + len(response_text.split())  # Rough estimate
        
        return response_text, tokens_used
        
    except Exception as e:
        raise Exception(f"Gemini API call failed: {str(e)}")


async def call_local_api(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "dummy",
    timeout: float = 30.0
) -> Tuple[str, int]:
    """
    Call local hosted model API with OpenAI-compatible interface.
    
    Args:
        system_prompt: The system-level prompt
        user_prompt: The user prompt/question
        model: The model name/identifier
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        base_url: Base URL for the local API
        api_key: API key (usually dummy for local models)
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (response_text, tokens_used)
        
    Raises:
        Exception: If API call fails
    """
    try:
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response_text = response.choices[0].message.content
        tokens_used = getattr(response.usage, 'total_tokens', 0)
        
        return response_text, tokens_used
        
    except Exception as e:
        raise Exception(f"Local API call failed for {model}: {str(e)}")

