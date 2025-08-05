#!/usr/bin/env python3
"""
Test script to verify all LLM model connections work properly.

This script tests each model in the router with a simple query to ensure:
1. API keys are working
2. Model endpoints are accessible
3. Response format is correct
4. Cost calculation is working
"""

import json
import os
import sys
import time
from typing import Dict, List, Tuple

# Add the project root to the path
sys.path.append('/fsx/home/zuxin.liu/1_project/Reasoning360')

from verl.tools.llm_router_tools import (MODEL_SPECS, call_model,
                                         print_usage_summary, get_available_models)


class ModelTester:
    """Test all models with connection and functionality verification"""
    
    def __init__(self):
        self.test_message = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
        self.test_params = {"max_tokens": 512, "temperature": 0.1}
        self.results = {}
        
    def test_single_model(self, model_id: str) -> Dict[str, any]:
        """Test a single model and return results"""
        spec = MODEL_SPECS[model_id]
        result = {
            "model_id": model_id,
            "model_name": spec.name,
            "provider": spec.provider,
            "status": "unknown",
            "response": "",
            "cost": 0.0,
            "latency": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "error": ""
        }
        
        print(f"Testing {spec.name} ({model_id})...")
        print(f"  Provider: {spec.provider}")
        print(f"  Expected cost: ${spec.input_price_per_million}/M input, ${spec.output_price_per_million}/M output")
        
        try:
            start_time = time.time()
            response, metadata = call_model(model_id, self.test_message, self.test_params)
            end_time = time.time()
            
            result.update({
                "status": "success",
                "response": response.strip(),
                "cost": metadata.get("cost", 0.0),
                "latency": end_time - start_time,
                "input_tokens": metadata.get("input_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0)
            })
            
            print(f"  ✅ Success: '{response.strip()}'")
            print(f"  💰 Cost: ${result['cost']:.6f}")
            print(f"  ⏱️  Latency: {result['latency']:.2f}s")
            print(f"  📊 Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
            
        except Exception as e:
            result.update({
                "status": "failed",
                "error": str(e)
            })
            print(f"  ❌ Failed: {str(e)}")
        
        print()
        return result
    
    def test_all_models(self) -> Dict[str, Dict]:
        """Test all models and return comprehensive results"""
        print("🚀 Testing All LLM Model Connections\n")
        print(f"Testing {len(MODEL_SPECS)} models with query: '{self.test_message[0]['content']}'")
        print("="*80 + "\n")
        
        for model_id in MODEL_SPECS.keys():
            self.results[model_id] = self.test_single_model(model_id)
            
            # Small delay between tests to be respectful to APIs
            time.sleep(1)
        
        return self.results
    
    def print_summary(self):
        """Print summary of all test results"""
        print("\n" + "="*80)
        print("📊 TEST SUMMARY")
        print("="*80)
        
        # Count results by status
        success_count = sum(1 for r in self.results.values() if r["status"] == "success")
        failed_count = sum(1 for r in self.results.values() if r["status"] == "failed")
        total_cost = sum(r["cost"] for r in self.results.values())
        
        print(f"\nOverall Results:")
        print(f"  ✅ Successful: {success_count}/{len(self.results)}")
        print(f"  ❌ Failed: {failed_count}/{len(self.results)}")
        print(f"  💰 Total Cost: ${total_cost:.6f}")
        
        # Group by provider
        providers = {}
        for result in self.results.values():
            provider = result["provider"]
            if provider not in providers:
                providers[provider] = {"success": 0, "failed": 0, "cost": 0.0}
            
            providers[provider][result["status"]] += 1
            providers[provider]["cost"] += result["cost"]
        
        print(f"\nBy Provider:")
        for provider, stats in providers.items():
            total = stats["success"] + stats["failed"]
            print(f"  {provider.upper()}:")
            print(f"    ✅ {stats['success']}/{total} successful")
            print(f"    💰 ${stats['cost']:.6f} total cost")
        
        # Show failed models
        if failed_count > 0:
            print(f"\nFailed Models:")
            for model_id, result in self.results.items():
                if result["status"] == "failed":
                    print(f"  ❌ {result['model_name']} ({model_id})")
                    print(f"     Error: {result['error']}")
        
        # Show successful models with responses
        if success_count > 0:
            print(f"\nSuccessful Models:")
            for model_id, result in self.results.items():
                if result["status"] == "success":
                    print(f"  ✅ {result['model_name']}: '{result['response']}'")
                    print(f"     Cost: ${result['cost']:.6f}, Latency: {result['latency']:.2f}s")
    
    def save_detailed_results(self, filename: str = "model_test_results.json"):
        """Save detailed results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to: {filename}")

def test_api_keys():
    """Test if API keys are available"""
    print("🔑 Checking API Keys...")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    keys_found = 0
    
    if openai_key:
        print(f"  ✅ OPENAI_API_KEY found (ends with: ...{openai_key[-4:]})")
        keys_found += 1
    else:
        print("  ❌ OPENAI_API_KEY not found")
    
    if together_key:
        print(f"  ✅ TOGETHER_API_KEY found (ends with: ...{together_key[-4:]})")
        keys_found += 1
    else:
        print("  ❌ TOGETHER_API_KEY not found")
        
    if gemini_key:
        print(f"  ✅ GEMINI_API_KEY found (ends with: ...{gemini_key[-4:]})")
        keys_found += 1
    else:
        print("  ❌ GEMINI_API_KEY not found")
    
    if keys_found == 0:
        print("  🚨 No API keys found! Tests will fail.")
        return False
    
    print(f"  📊 {keys_found}/3 API keys available")
    print()
    return True

def main():
    """Main test function"""
    print("🧪 LLM Router Tools - Model Connection Test")
    print("="*80)
    
    # Test API keys first
    if not test_api_keys():
        print("❌ API key check failed. Please set OPENAI_API_KEY and/or TOGETHER_API_KEY")
        return
    
    # Initialize and run tester
    tester = ModelTester()
    
    # Run all tests
    results = tester.test_all_models()
    
    # Print summary
    tester.print_summary()
    
    # Print router usage stats
    print("\n" + "="*80)
    print("📈 ROUTER USAGE STATISTICS")
    print("="*80)
    print_usage_summary()
    
    # Save detailed results
    tester.save_detailed_results()
    
    # Return success/failure based on results
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    if success_count == len(results):
        print("\n🎉 All models tested successfully!")
        return 0
    else:
        failed_count = len(results) - success_count
        print(f"\n⚠️  {failed_count} models failed testing.")
        return 1

if __name__ == "__main__":
    exit_code = main()