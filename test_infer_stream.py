#!/usr/bin/env python3
"""
Simple test script to verify infer_stream method works correctly
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import Engine

def test_infer_stream():
    """Test the infer_stream method with a simple prompt"""
    print("=== Testing infer_stream method ===")

    # Initialize engine
    print("1. Initializing engine...")
    try:
        engine = Engine()
        print(f"   Engine initialized successfully with model: {engine.model_id}")
    except Exception as e:
        print(f"   Error initializing engine: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test messages
    messages = [
        {"role": "user", "content": "Hello, can you count to 3?"}
    ]

    print("\n2. Starting streaming test...")
    print(f"   Input messages: {messages}")

    try:
        piece_count = 0
        generated_text = ""

        for piece in engine.infer_stream(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        ):
            piece_count += 1
            print(f"   Piece {piece_count}: '{piece}'")
            generated_text += piece

        print(f"\n3. Streaming completed!")
        print(f"   Total pieces: {piece_count}")
        print(f"   Generated text: '{generated_text}'")

        if piece_count == 0:
            print("   ❌ ERROR: No pieces were generated!")
            return False
        else:
            print("   ✅ SUCCESS: Streaming worked correctly!")
            return True

    except Exception as e:
        print(f"   ❌ ERROR during streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_infer_non_stream():
    """Test the regular infer method for comparison"""
    print("\n=== Testing infer method (non-streaming) ===")

    try:
        engine = Engine()
        messages = [
            {"role": "user", "content": "Hello, can you count to 3?"}
        ]

        print("1. Starting non-streaming inference...")
        result = engine.infer(
            messages=messages,
            max_tokens=50,
            temperature=0.7
        )

        print(f"2. Non-streaming result: '{result}'")
        print("   ✅ Non-streaming inference works!")
        return True

    except Exception as e:
        print(f"   ❌ ERROR during non-streaming: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing inference methods...")
    print("=" * 50)

    # Test non-streaming first (should work)
    non_stream_ok = test_infer_non_stream()

    # Test streaming (currently problematic)
    stream_ok = test_infer_stream()

    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Non-streaming: {'✅ PASS' if non_stream_ok else '❌ FAIL'}")
    print(f"Streaming: {'✅ PASS' if stream_ok else '❌ FAIL'}")

    if not stream_ok:
        print("\n🔍 Debugging notes:")
        print("- Check the detailed logs above for where streaming fails")
        print("- Verify the model.generate() call in the thread")
        print("- Check if TextIteratorStreamer receives any tokens")
        print("- Look for template or tokenization issues")