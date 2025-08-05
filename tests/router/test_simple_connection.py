#!/usr/bin/env python3
"""
Simple test of all model connections using direct import
"""

import importlib.util
import json
import os
import sys
import time

from dotenv import load_dotenv

# Load environment
load_dotenv()

def load_router_module():
    """Directly load the router module"""
    spec = importlib.util.spec_from_file_location(
        "llm_router_tools", 
        "/fsx/home/zuxin.liu/1_project/Reasoning360/verl/tools/llm_router_tools.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_all_models():
    """Test all models using the unified API"""
    print("🧪 LLM Router Tools - Model Connection Test")
    print("="*80)
    
    # Load router module
    router = load_router_module()
    
    # Check API keys
    print("🔑 Checking API Keys...")
    openai_key = os.getenv("OPENAI_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    keys_found = 0
    if openai_key:
        print(f"  ✅ OPENAI_API_KEY found")
        keys_found += 1
    if together_key:
        print(f"  ✅ TOGETHER_API_KEY found")
        keys_found += 1
    if gemini_key:
        print(f"  ✅ GEMINI_API_KEY found")
        keys_found += 1
    
    print(f"  📊 {keys_found}/3 API keys available\n")
    
    # Test parameters
    test_message = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    test_params = {"max_tokens": 2048, "temperature": 0.1}
    
    print(f"🚀 Testing {len(router.MODEL_SPECS)} models with query: '{test_message[0]['content']}'")
    print("="*80 + "\n")
    
    results = {}
    
    for model_id, spec in router.MODEL_SPECS.items():
        print(f"Testing {spec.name} ({model_id})...")
        print(f"  Provider: {spec.provider}")
        
        # Skip gemini-2.5-flash due to known rate limit issues
        if model_id == "gemini-2.5-flash":
            results[model_id] = {
                "model_id": model_id,
                "model_name": spec.name,
                "provider": spec.provider,
                "status": "skipped",
                "response": "",
                "cost": 0.0,
                "latency": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": "Skipped due to rate limit issues"
            }
            print(f"  ⚠️  Skipped: Known rate limit issues")
            print()
            continue
        
        try:
            start_time = time.time()
            response, metadata = router.call_model(model_id, test_message, test_params)
            end_time = time.time()
            
            results[model_id] = {
                "model_id": model_id,
                "model_name": spec.name,
                "provider": spec.provider,
                "status": "success",
                "response": response.strip(),
                "cost": metadata.get("cost", 0.0),
                "latency": end_time - start_time,
                "input_tokens": metadata.get("input_tokens", 0),
                "output_tokens": metadata.get("output_tokens", 0),
                "error": ""
            }
            
            print(f"  ✅ Success: '{response.strip()}'")
            print(f"  💰 Cost: ${results[model_id]['cost']:.6f}")
            print(f"  ⏱️  Latency: {results[model_id]['latency']:.2f}s")
            print(f"  📊 Tokens: {results[model_id]['input_tokens']} in, {results[model_id]['output_tokens']} out")
            
        except Exception as e:
            results[model_id] = {
                "model_id": model_id,
                "model_name": spec.name,
                "provider": spec.provider,
                "status": "failed",
                "response": "",
                "cost": 0.0,
                "latency": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "error": str(e)
            }
            print(f"  ❌ Failed: {str(e)}")
        
        print()
        time.sleep(1)  # Rate limiting
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    success_count = sum(1 for r in results.values() if r["status"] == "success")
    failed_count = sum(1 for r in results.values() if r["status"] == "failed")
    skipped_count = sum(1 for r in results.values() if r["status"] == "skipped")
    total_cost = sum(r["cost"] for r in results.values())
    
    print(f"\nOverall Results:")
    print(f"  ✅ Successful: {success_count}/{len(results)}")
    print(f"  ❌ Failed: {failed_count}/{len(results)}")
    print(f"  ⚠️  Skipped: {skipped_count}/{len(results)}")
    print(f"  💰 Total Cost: ${total_cost:.6f}")
    
    # Group by provider
    providers = {}
    for result in results.values():
        provider = result["provider"]
        if provider not in providers:
            providers[provider] = {"success": 0, "failed": 0, "skipped": 0, "cost": 0.0}
        
        providers[provider][result["status"]] += 1
        providers[provider]["cost"] += result["cost"]
    
    print(f"\nBy Provider:")
    for provider, stats in providers.items():
        total = stats["success"] + stats["failed"] + stats["skipped"]
        print(f"  {provider.upper()}:")
        print(f"    ✅ {stats['success']}/{total} successful")
        if stats["failed"] > 0:
            print(f"    ❌ {stats['failed']} failed")
        if stats["skipped"] > 0:
            print(f"    ⚠️  {stats['skipped']} skipped")
        print(f"    💰 ${stats['cost']:.6f} total cost")
    
    # Show failed models
    if failed_count > 0:
        print(f"\nFailed Models:")
        for model_id, result in results.items():
            if result["status"] == "failed":
                print(f"  ❌ {result['model_name']} ({model_id})")
                print(f"     Error: {result['error']}")
    
    # Save results
    with open('model_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: model_test_results.json")
    
    # Final assessment
    expected_working = len(results) - 1  # All except gemini-2.5-flash
    working_models = success_count
    
    if working_models >= expected_working - 2:  # Allow for 2 failures
        print(f"\n🎉 EXCELLENT: {working_models}/{expected_working} expected models working!")
        print("The unified router system is functioning well across all providers.")
    elif working_models >= expected_working // 2:
        print(f"\n✅ GOOD: {working_models}/{expected_working} expected models working.")
        print("Most models are functional with some issues to investigate.")
    else:
        print(f"\n⚠️  ISSUES: Only {working_models}/{expected_working} expected models working.")
        print("Significant problems need to be addressed.")
    
    return results

if __name__ == "__main__":
    test_all_models()