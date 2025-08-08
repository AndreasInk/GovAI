#!/usr/bin/env python3
"""
Test script for LLM judge drift detection
=========================================

This script demonstrates how the LLM judge works for detecting semantic drift
between summary sentences and their source text.
"""

from pathlib import Path
import sys

# Ensure local imports work when invoked directly
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))


def test_llm_judge():
    """Test the LLM judge with some example cases."""
    
    # Test cases: (summary, source, expected_drift)
    test_cases = [
        (
            "Homeowners must pay a $50 fine for parking violations.",
            "Homeowners may be subject to a fine of up to $50 for parking violations.",
            True  # "must" vs "may" - significant legal difference
        ),
        (
            "The Board can form committees to set rules for facility use.",
            "The Board can form committees (e.g. House, Sports) to set rules for facility use.",
            False  # Minor omission, acceptable paraphrasing
        ),
        (
            "Residents are required to maintain their property.",
            "Property owners must maintain their property in good condition.",
            False  # Acceptable paraphrasing
        ),
        (
            "Violations result in immediate suspension of privileges.",
            "Violations may result in suspension of privileges after written notice.",
            True  # "immediate" vs "after written notice" - significant difference
        ),
        (
            "The annual assessment is $500.",
            "The annual assessment is $500 per lot.",
            True  # Missing "per lot" - important detail
        )
    ]
    
    print("🤖 Testing LLM Judge Drift Detection")
    print("=" * 50)
    
    for i, (summary, source, expected_drift) in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}:")
        print(f"Summary: {summary}")
        print(f"Source:  {source}")
        print(f"Expected drift: {expected_drift}")
        
        # Call the LLM judge
        from ingest import _llm_judge_drift
        is_drift, confidence, reasoning = _llm_judge_drift(summary, source)
        
        print("🤖 LLM Result:")
        print(f"  - Drift detected: {is_drift}")
        print(f"  - Confidence: {confidence:.2f}")
        print(f"  - Reasoning: {reasoning}")
        
        # Check if result matches expectation
        if is_drift == expected_drift:
            print("✅ CORRECT - LLM judge matched expectation")
        else:
            print("❌ INCORRECT - LLM judge disagreed with expectation")
        
        print("-" * 50)

def test_with_json_draft():
    """Test with an actual JSON draft file if available."""
    draft_path = Path("draft.json")
    if not draft_path.exists():
        print("No draft.json found. Create one first by running:")
        print("python research-with-mcp.py --wait --out draft.json")
        return
    
    print("\n📄 Testing with actual draft.json")
    print("=" * 50)
    
    from ingest import _load_json_sentences, _llm_judge_drift
    
    try:
        pairs = _load_json_sentences(draft_path)
        print(f"Found {len(pairs)} summary/source pairs")
        
        # Test first few pairs
        for i, (summary, source) in enumerate(pairs[:3]):
            if source is None:
                print(f"\n📋 Pair {i+1}: Executive Summary (no source)")
                print(f"Summary: {summary[:100]}...")
                continue
                
            print(f"\n📋 Pair {i+1}:")
            print(f"Summary: {summary}")
            print(f"Source:  {source[:200]}...")
            
            is_drift, confidence, reasoning = _llm_judge_drift(summary, source)
            
            print("🤖 LLM Result:")
            print(f"  - Drift detected: {is_drift}")
            print(f"  - Confidence: {confidence:.2f}")
            print(f"  - Reasoning: {reasoning}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error testing with draft.json: {e}")

if __name__ == "__main__":
    test_llm_judge()
    test_with_json_draft() 