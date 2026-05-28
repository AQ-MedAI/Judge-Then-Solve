"""
Simple test to verify final answer extraction logic.
"""

def extract_final_answer(response: str) -> str:
    """Extract final answer from response.

    For reasoning models that use <think>...</think> tags, this extracts
    only the content after </think> tag as the final answer.
    For other models, returns the full response.
    """
    # Check if response contains </think> tag (reasoning model output)
    if "</think>" in response:
        # Extract everything after the last </think> tag
        final_answer = response.split("</think>")[-1].strip()
        return final_answer

    # For non-reasoning models, return the full response
    return response


def test_extraction():
    print("=" * 80)
    print("Test Case 1: Response with reasoning tags")
    print("=" * 80)

    reasoning_response = """<think>
Let me calculate this step by step.
2 + 2 = 4
This is a simple arithmetic problem.
</think>

The answer is 4."""

    extracted = extract_final_answer(reasoning_response)
    print(f"Original response:\n{reasoning_response}\n")
    print(f"Extracted final answer:\n{extracted}\n")
    print(f"Contains </think>: {'</think>' in extracted}")
    assert "</think>" not in extracted, "Final answer should not contain </think>"
    print("✓ Test passed!\n")

    # Test case 2
    print("=" * 80)
    print("Test Case 2: Response without reasoning tags")
    print("=" * 80)

    simple_response = "The answer is 4."
    extracted_simple = extract_final_answer(simple_response)
    print(f"Original response:\n{simple_response}\n")
    print(f"Extracted final answer:\n{extracted_simple}\n")
    print(f"Should be identical: {simple_response == extracted_simple}")
    assert simple_response == extracted_simple, "Should return full response"
    print("✓ Test passed!\n")

    # Test case 3
    print("=" * 80)
    print("Test Case 3: Multiple </think> tags")
    print("=" * 80)

    multi_think_response = """<think>
First reasoning step
</think>
Some intermediate text
<think>
Second reasoning step
</think>

Final answer: 42"""

    extracted_multi = extract_final_answer(multi_think_response)
    print(f"Original response:\n{multi_think_response}\n")
    print(f"Extracted final answer:\n{extracted_multi}\n")
    print(f"Contains </think>: {'</think>' in extracted_multi}")
    assert "</think>" not in extracted_multi, "Final answer should not contain </think>"
    print("✓ Test passed!\n")

    print("=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_extraction()
