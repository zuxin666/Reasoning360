#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced LLM Router using OpenAI Tools Format

This module provides a sophisticated router system that:
1. Imports selected model functions from llm_router_tools
2. Converts them to OpenAI tools format
3. Uses an LLM to intelligently select and execute the best model
4. Provides feedback and summarization capabilities

Usage:
    from test_routing import ToolBasedRouter
    
    router = ToolBasedRouter()
    response, metadata = router.route(
        "Solve this complex math problem: 2x + 5 = 15"
    )
"""

import json
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm

# Import model specs and call_model from llm_router_tools
from verl.tools.llm_router_tools import MODEL_SPECS, call_model


# Auto-generate OpenAI-compatible function tools
def create_openai_tools(selected_models: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Create OpenAI function calling tools from model specifications.
    
    Args:
        selected_models: List of model IDs to include, or None for all models
        
    Returns:
        List of OpenAI function tool definitions
    """
    if selected_models is None:
        selected_models = list(MODEL_SPECS.keys())
    
    tools = []
    
    for model_id in selected_models:
        if model_id not in MODEL_SPECS:
            continue
            
        spec = MODEL_SPECS[model_id]
        func_name = f"call_{model_id.replace('-', '_').replace('.', '_')}"
        
        tool = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": f"Call {spec.name} model. {spec.description}",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "revised_system_prompt": {
                            "type": "string",
                            "description": "Optimized system prompt for this specific model and task"
                        },
                        "max_tokens": {
                            "type": "integer",
                            "description": "Maximum tokens to generate",
                            "default": 8192 if "thinking" in spec.capabilities else 2048
                        },
                        "temperature": {
                            "type": "number",
                            "description": "Sampling temperature (0.0 to 2.0)",
                            "default": 1.0
                        }
                    },
                    "required": ["revised_system_prompt"]
                },
                "metadata": {
                    "model_id": model_id,
                    "model_name": spec.name,
                    "provider": spec.provider,
                    "capabilities": spec.capabilities,
                    "quality_tier": spec.quality_tier,
                    "cost": f"${spec.input_price_per_million}/M input, ${spec.output_price_per_million}/M output"
                }
            }
        }
        tools.append(tool)
    
    return tools

class ToolBasedRouter:
    """
    Advanced router that uses OpenAI tools format for intelligent model selection
    and execution. The router acts as a prompt engineer, selecting the best model
    and creating optimized system prompts for each task.
    """
    
    def __init__(self, router_model: str = "gpt-4o-mini", selected_models: Optional[List[str]] = None):
        """
        Initialize the tool-based router.
        
        Args:
            router_model: The model to use for routing decisions
            selected_models: List of model IDs to include, or None for default
        """
        self.router_model = router_model
        self.conversation_history = []
        self.execution_history = []
        
        # Default to a curated set of models if none specified
        if selected_models is None:
            selected_models = [
                "gpt-4o-mini", "qwen2.5-7b", "gemini-2.5-flash-lite", 
                "deepseek-v3", "qwen2.5-coder-32b", "gpt-4.1"
            ]
        
        # Filter selected models to only include those that exist
        self.selected_model_ids = [
            model_id for model_id in selected_models 
            if model_id in MODEL_SPECS
        ]
        
        # Convert models to OpenAI tools format using helper function
        self.tools = self._create_tools()
        
    def _create_tools(self) -> List[Dict[str, Any]]:
        """Convert selected models to OpenAI tools format."""
        return create_openai_tools(self.selected_model_ids)
    
    def _tool_name_to_model_id(self, tool_name: str) -> str:
        """Convert tool name back to model ID."""
        # Extract model_id from tool_name (convert back from call_xxx format)
        model_id = tool_name.replace("call_", "")
        
        # Handle special cases for dots in model names
        # Convert back by checking against available models
        for available_model_id in self.selected_model_ids:
            # Create the expected function name for this model
            expected_func_name = available_model_id.replace('-', '_').replace('.', '_')
            if model_id == expected_func_name:
                return available_model_id
        
        # Fallback to simple underscore to hyphen conversion
        model_id = model_id.replace("_", "-")
        return model_id
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any], original_messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        try:
            # Create simple mapping from tool name to model ID
            model_id = self._tool_name_to_model_id(tool_name)
            
            if model_id not in self.selected_model_ids:
                return {
                    "error": f"Model not found for tool name {tool_name}. Available models: {self.selected_model_ids}",
                    "success": False
                }
            
            # Use unified call_model function directly
            def model_func(messages, sampling_params=None):
                return call_model(model_id, messages, sampling_params)
            
            # Prepare messages with revised system prompt
            revised_system_prompt = arguments["revised_system_prompt"]
            
            # Create new messages list, replacing any existing system message with the revised one
            messages = []
            system_added = False
            
            for msg in original_messages:
                if msg["role"] == "system":
                    # Replace existing system message with revised one
                    messages.append({"role": "system", "content": revised_system_prompt})
                    system_added = True
                else:
                    messages.append(msg)
            
            # Add revised system prompt if no system message was found
            if not system_added:
                messages.insert(0, {"role": "system", "content": revised_system_prompt})
            
            # Prepare sampling parameters
            sampling_params = {}
            if "max_tokens" in arguments:
                sampling_params["max_tokens"] = arguments["max_tokens"]
            if "temperature" in arguments:
                sampling_params["temperature"] = arguments["temperature"]
            if "top_p" in arguments:
                sampling_params["top_p"] = arguments["top_p"]
            
            # Execute the model call
            start_time = time.time()
            response, metadata = model_func(messages, sampling_params)
            execution_time = time.time() - start_time
            
            # Record execution
            execution_record = {
                "tool_name": tool_name,
                "model_id": model_id,
                "arguments": arguments,
                "response": response,
                "metadata": metadata,
                "execution_time": execution_time,
                "timestamp": time.time()
            }
            self.execution_history.append(execution_record)
            
            return {
                "success": True,
                "response": response,
                "metadata": metadata,
                "execution_time": execution_time,
                "model_used": model_id
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "tool_name": tool_name
            }
    
    def _create_router_prompt(self, original_messages: List[Dict[str, str]]) -> str:
        """Create a prompt for the router model."""
        
        prompt = f"""You are an intelligent LLM router and prompt engineer. Your job is to:

1. Analyze the conversation history
2. Select the most appropriate model from the available tools
3. Create an optimized system prompt for that specific model and task
4. Set appropriate sampling parameters

IMPORTANT: You MUST call one of the available model functions. Do not provide a direct response.

INSTRUCTIONS:
- Choose the best model based on task complexity and required capabilities
- For simple tasks, prefer budget models
- For complex reasoning/math, prefer specialized models
- For coding tasks, prefer coding-specialized models
- For general tasks, prefer balanced models
- Use lower temperature (0.1-0.5) for precise tasks, higher (0.7-1.0) for creative tasks
- Create a system prompt that optimizes the model's performance for this specific task
- Consider cost efficiency - choose the most cost-effective model that can solve the task well

SYSTEM PROMPT GUIDELINES:
- Be specific about the task requirements
- Include any relevant context or constraints
- Optimize for the model's strengths and capabilities
- Keep it concise but comprehensive
- For coding tasks: include coding standards and best practices
- For math tasks: emphasize step-by-step reasoning and precision
- For educational tasks: include appropriate explanation level
- For creative tasks: encourage creativity and style

USER MESSAGES: {original_messages}

Now please select the best model and create an optimized system prompt for that specific model and task. You MUST call one of the available functions."""
        
        return prompt
    
    def route(self, user_query: str, max_iterations: int = 3) -> Tuple[str, Dict[str, Any]]:
        """
        Route a user query using the tool-based approach.
        
        Args:
            user_query: The user's request
            max_iterations: Maximum number of routing iterations
            
        Returns:
            Tuple of (final_response, metadata)
        """
        # Initialize conversation with original user query
        original_messages = [{"role": "user", "content": user_query}]

        # Create initial router prompt
        router_prompt = self._create_router_prompt(original_messages)
        
        # Router conversation
        router_messages = [
            {"role": "user", "content": router_prompt}
        ]
        
        iteration = 0
        final_response = None
        total_metadata = {
            "iterations": [],
            "total_cost": 0.0,
            "models_used": [],
            "execution_times": []
        }
        
        while iteration < max_iterations:
            iteration += 1
            print(f"\n=== Router Iteration {iteration} ===")
            
            try:
                # For first iteration, force tool calling; for subsequent iterations, allow choice
                tool_choice = "auto"
                
                # Call the router model with tools using vanilla litellm
                router_response = litellm.completion(
                    model=self.router_model,
                    messages=router_messages,
                    max_tokens=8192,
                    temperature=0.1,
                    tools=self.tools,
                    tool_choice=tool_choice
                )
                
                # Check if the router wants to use a tool
                if hasattr(router_response, 'choices') and router_response.choices:
                    choice = router_response.choices[0]
                    
                    if hasattr(choice, 'message') and hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                        tool_calls = choice.message.tool_calls
                        
                        for tool_call in tool_calls:
                            # Execute the tool
                            tool_name = tool_call.function.name
                            arguments = json.loads(tool_call.function.arguments)

                            print(f"Tool name: {tool_name}")
                            print(f"Arguments: {arguments}")
                            
                            result = self._execute_tool(tool_name, arguments, original_messages)
                            
                            if result["success"]:
                                # Add tool result to conversation
                                router_messages.append({
                                    "role": "assistant", 
                                    "content": None,
                                    "tool_calls": [tool_call]
                                })
                                router_messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": result["response"]
                                })
                                
                                # Update metadata
                                total_metadata["iterations"].append({
                                    "iteration": iteration,
                                    "model_used": result["model_used"],
                                    "cost": result["metadata"]["cost"],
                                    "execution_time": result["execution_time"]
                                })
                                total_metadata["total_cost"] += result["metadata"]["cost"]
                                total_metadata["models_used"].append(result["model_used"])
                                total_metadata["execution_times"].append(result["execution_time"])
                                
                                # Set final response
                                final_response = result["response"]
                                
                                # Ask router to summarize if needed
                                if iteration < max_iterations:
                                    router_messages.append({
                                        "role": "user",
                                        "content": "The model has provided a response. Please summarize the key points and provide a clear answer to the user's original question."
                                    })
                            else:
                                # Tool execution failed
                                router_messages.append({
                                    "role": "assistant",
                                    "content": f"I tried to use a model but encountered an error: {result['error']}"
                                })
                                break
                        
                        if final_response:
                            break
                    else:
                        # No function call made - check if there's message content
                        if hasattr(choice, 'message') and choice.message and hasattr(choice.message, 'content'):
                            final_response = choice.message.content
                            if final_response:
                                print(f"Router provided direct response: {final_response}")
                                break
                        
                        # If still no response and it's first iteration, force tool calling
                        if iteration == 1:
                            print("First iteration didn't call tool, retrying with stronger prompt...")
                            router_messages.append({
                                "role": "user",
                                "content": "You MUST select and call one of the available model functions. Do not provide a direct response. Please choose the most appropriate model and execute it with an optimized system prompt."
                            })
                            continue
                        else:
                            # Fallback response
                            final_response = "I apologize, but I was unable to process your request properly."
                            break
                else:
                    # Fallback to direct response
                    final_response = router_response
                    break
                    
            except Exception as e:
                print(f"Error in iteration {iteration}: {e}")
                final_response = f"An error occurred during routing: {str(e)}"
                break
        
        # Add final metadata
        total_metadata["final_response"] = final_response
        total_metadata["total_iterations"] = iteration
        
        return final_response, total_metadata
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history."""
        return self.execution_history
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get a summary of tool usage."""
        if not self.execution_history:
            return {"message": "No executions recorded"}
        
        total_cost = sum(exec["metadata"]["cost"] for exec in self.execution_history)
        total_calls = len(self.execution_history)
        models_used = {}
        
        for exec_record in self.execution_history:
            model_id = exec_record["model_id"]
            if model_id not in models_used:
                models_used[model_id] = {
                    "calls": 0,
                    "total_cost": 0.0,
                    "total_tokens": 0
                }
            
            models_used[model_id]["calls"] += 1
            models_used[model_id]["total_cost"] += exec_record["metadata"]["cost"]
            models_used[model_id]["total_tokens"] += (
                exec_record["metadata"]["input_tokens"] + 
                exec_record["metadata"]["output_tokens"]
            )
        
        return {
            "total_calls": total_calls,
            "total_cost": total_cost,
            "models_used": models_used,
            "average_cost_per_call": total_cost / total_calls if total_calls > 0 else 0
        }


def create_simple_router() -> ToolBasedRouter:
    """Create a simple router with basic models for testing."""
    # Just use the default models from the constructor
    return ToolBasedRouter()


if __name__ == "__main__":
    print("=== Tool-Based LLM Router Test ===\n")
    
    # Create router
    router = create_simple_router()
    
    # Test cases
    test_queries = [
        {
            "query": "What is 15 * 23?",
            "description": "Simple math problem"
        },
        {
            "query": "Write a Python function to calculate fibonacci numbers",
            "description": "Coding task"
        },
        {
            "query": "Explain the concept of quantum entanglement in simple terms",
            "description": "Complex reasoning task"
        },
        {
            "query": "Summarize the benefits of renewable energy",
            "description": "General information task"
        }
    ]
    
    # Test each query
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test_case['description']} ---")
        print(f"Query: {test_case['query']}")
        
        try:
            response, metadata = router.route(test_case["query"])
            
            print(f"Response: {response}")
            print(f"Models used: {metadata['models_used']}")
            print(f"Total cost: ${metadata['total_cost']:.4f}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        print("-" * 60)
    
    # Print usage summary
    print("\n=== Usage Summary ===")
    summary = router.get_usage_summary()
    print(json.dumps(summary, indent=2))
    
    print("\n=== Router Test Complete ===")