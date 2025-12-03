#!/usr/bin/env python3
"""
Test script to check Ollama is working properly
"""

import requests
import json
import time

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "llama3:latest"

print("🧪 Testing Ollama Connection and Response")
print("=" * 50)
print()

# Test 1: Can we reach Ollama?
print("Test 1: Checking if Ollama API is reachable...")
try:
    response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    if response.status_code == 200:
        print("✅ Ollama API is reachable")
        models = response.json().get('models', [])
        print(f"   Found {len(models)} models:")
        for model in models:
            print(f"   - {model.get('name', 'unknown')}")
    else:
        print(f"❌ Ollama API returned status {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ Cannot reach Ollama: {e}")
    print("   Make sure Ollama is running: ollama serve")
    exit(1)

print()

# Test 2: Can we generate a simple response?
print(f"Test 2: Testing generation with {MODEL}...")
print("Sending prompt: 'Say hello in one word'")
print()

try:
    start_time = time.time()
    
    payload = {
        "model": MODEL,
        "prompt": "Say hello in one word",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 10
        }
    }
    
    print("⏳ Waiting for response (this may take 5-30 seconds on first run)...")
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=120
    )
    
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        answer = result.get("response", "")
        
        print(f"✅ Got response in {elapsed:.2f} seconds")
        print(f"   Response: '{answer}'")
        print(f"   Length: {len(answer)} characters")
        
        if len(answer) > 0:
            print()
            print("🎉 SUCCESS! Ollama is working correctly!")
            print()
            print("💡 Next steps:")
            print("   1. Keep this terminal open with: ollama run llama3:latest")
            print("   2. In another terminal run: streamlit run simple_reliable_chatbot.py")
        else:
            print()
            print("⚠️ WARNING: Ollama responded but answer is empty")
            print("   This might indicate the model isn't fully loaded")
    else:
        print(f"❌ Request failed with status {response.status_code}")
        print(f"   Response: {response.text}")
        
except requests.exceptions.Timeout:
    print(f"❌ Request timed out after 120 seconds")
    print()
    print("🔧 Possible fixes:")
    print("   1. Your model is too large for your RAM")
    print("   2. Try a smaller model: ollama pull llama3.2:3b")
    print("   3. Close other applications to free up memory")
    print("   4. Check RAM usage: free -h")
    
except Exception as e:
    print(f"❌ Error: {e}")

print()
print("=" * 50)
print("🔍 Debug complete")