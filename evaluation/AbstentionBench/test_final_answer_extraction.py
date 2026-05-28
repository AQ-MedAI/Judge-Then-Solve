"""
Test script to verify that DirectAbstention correctly extracts final answer
from reasoning model outputs (after </think> tag).
"""

from recipe.abstention import DirectAbstention
from recipe.inference import RawResponse
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt


def test_final_answer_extraction():
    """Test that final answer is correctly extracted from </think> tag."""

    # Create a mock DirectAbstention instance
    abstention_method = DirectAbstention(save_dir="/tmp/test")

    # Test case 1: Response with <think>...</think> tags
    print("=" * 80)
    print("Test Case 1: Response with reasoning tags")
    print("=" * 80)

    mock_prompt = Prompt(
        question="What is 2+2?",
        should_abstain=False,
        reference_answers=["4"]
    )

    reasoning_response = """<think>
Let me calculate this step by step.
2 + 2 = 4
This is a simple arithmetic problem.
</think>

The answer is 4."""

    raw_response_with_reasoning = RawResponse(
        prompt=mock_prompt,
        response=reasoning_response,
        reasoning=None,
        response_with_reasoning=None
    )

    extracted = abstention_method.respond_or_abstain(raw_response_with_reasoning)
    print(f"Original response:\n{reasoning_response}\n")
    print(f"Extracted final answer:\n{extracted}\n")
    print(f"Contains </think>: {'</think>' in extracted}")
    print()

    # Test case 2: Response without reasoning tags
    print("=" * 80)
    print("Test Case 2: Response without reasoning tags")
    print("=" * 80)

    simple_response = "The answer is 4."

    raw_response_simple = RawResponse(
        prompt=mock_prompt,
        response=simple_response,
        reasoning=None,
        response_with_reasoning=None
    )

    extracted_simple = abstention_method.respond_or_abstain(raw_response_simple)
    print(f"Original response:\n{simple_response}\n")
    print(f"Extracted final answer:\n{extracted_simple}\n")
    print(f"Should be identical: {simple_response == extracted_simple}")
    print()

    # Test case 3: Multiple </think> tags (edge case)
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

    raw_response_multi = RawResponse(
        prompt=mock_prompt,
        response=multi_think_response,
        reasoning=None,
        response_with_reasoning=None
    )

    extracted_multi = abstention_method.respond_or_abstain(raw_response_multi)
    print(f"Original response:\n{multi_think_response}\n")
    print(f"Extracted final answer:\n{extracted_multi}\n")
    print(f"Contains </think>: {'</think>' in extracted_multi}")
    print()

    print("=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_final_answer_extraction()
