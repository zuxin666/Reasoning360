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
import time
import json
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema
from .utils.router_utils import call_gpt_api, simulate_user_response
from .conversation_context import conversation_context


class GPTTool(BaseTool):
    """A tool for routing requests to GPT for intelligent processing."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        # Default configuration for GPT-4o calls
        self.default_model = config.get("model", "gpt-4o")
        self.default_temperature = config.get("temperature", 0.7)
        self.default_max_tokens = config.get("max_tokens", 1000)
        self.api_key = config.get("api_key")  # Will use env var if not provided

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance for a conversation."""
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute a GPT call with the provided parameters."""
        # Extract parameters
        system_prompt = parameters.get("system_prompt", "")
        user_prompt = parameters.get("user_prompt", "")
        temperature = parameters.get("temperature", self.default_temperature)
        
        if not system_prompt or not user_prompt:
            raise ValueError("Both system_prompt and user_prompt are required")
        
        try:
            # Call GPT API
            start_time = time.time()
            response, tokens_used = await call_gpt_api(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.default_model,
                temperature=temperature,
                max_tokens=self.default_max_tokens,
                api_key=self.api_key
            )
            end_time = time.time()
            
            # Append interaction to conversation context
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to=self.default_model,
                arguments=parameters,
                feedback=response
            )
            
            # Calculate reward
            reward = 0.0
            
            # Prepare metrics
            metrics = {
                "tokens_used": tokens_used,
                "latency": end_time - start_time
            }
            
            return response, reward, self.default_model, json.dumps(parameters), metrics
            
        except Exception as e:
            # Record failed attempt
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to=self.default_model,
                arguments=parameters,
                feedback=f"Error: {str(e)}"
            )
            
            return f"Error calling {self.default_model}: {str(e)}", 0.0, self.default_model, json.dumps(parameters), {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward for the conversation."""
        context = conversation_context.get_context(instance_id)
        
        # Calculate reward based on successful calls
        calls = [c for c in context if c["route_to"] == self.default_model]
        if not calls:
            return 0.0
        
        successful_calls = sum(1 for call in calls 
                             if not call["feedback"].startswith("Error:"))
        
        return successful_calls / len(calls)

    async def release(self, instance_id: str, **kwargs) -> None:
        # Release the context
        conversation_context.clear_context(instance_id)


class UserTool(BaseTool):
    """A tool for routing requests to the user for clarification or input."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        
        # Configuration options
        self.simulate_user = config.get("simulate_user", True)
        self.simulation_model = config.get("simulation_model", "gpt-4o")
        self.default_response_timeout = config.get("response_timeout", 30.0)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        """Create a tool instance for user interaction."""
        if instance_id is None:
            instance_id = str(uuid4())
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Route content to user or simulate user response."""
        # Extract parameters
        content = parameters.get("content", "")
        
        if not content:
            raise ValueError("Content parameter is required")
        
        try:
            start_time = time.time()
            
            if self.simulate_user:
                # Get conversation history for user simulation
                context = conversation_context.get_context(instance_id)
                user_response = await simulate_user_response(
                    content=content,
                    conversation_history=context,
                    model=self.simulation_model
                )
            else:
                # In real scenario, this would wait for actual user input
                user_response = f"[User input requested for: {content}]"
            
            end_time = time.time()
            
            # Append interaction to conversation context
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to="user",
                arguments=parameters,
                feedback=user_response
            )
            
            # Calculate reward
            reward = 0.0
            
            # Prepare metrics
            metrics = {
                "latency": end_time - start_time,
                "simulated": self.simulate_user
            }
            
            return user_response, reward, self.default_model, json.dumps(parameters), metrics
            
        except Exception as e:
            # Record failed attempt
            conversation_context.append_interaction(
                request_id=instance_id,
                route_to="user",
                arguments=parameters,
                feedback=f"Error: {str(e)}"
            )
            
            return f"Error in user interaction: {str(e)}", 0.0, self.default_model, json.dumps(parameters), {"error": str(e)}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate final reward for user interactions."""
        context = conversation_context.get_context(instance_id)
        
        # Calculate reward based on successful user interactions
        user_calls = [c for c in context if c["route_to"] == "user"]
        if not user_calls:
            return 0.0
        
        successful_calls = sum(1 for call in user_calls 
                             if not call["feedback"].startswith("Error:"))
        
        return successful_calls / len(user_calls)

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance."""
        # Release the context
        conversation_context.clear_context(instance_id)

