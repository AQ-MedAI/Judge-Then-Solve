"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pytest

from recipe.abstention import Response
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.evaluation import LLMJudgeCorrectnessEvaluator
from recipe.models import Llama3_1_8B_Instruct


class TestLLMJudgeCorrectnessEvaluator:

    @pytest.mark.slow
    def test_is_abstention(self):
        judge_model = Llama3_1_8B_Instruct()
        correctness_evaluator = LLMJudgeCorrectnessEvaluator(
            judge_model, save_dir="save_dir"
        )
        prompt = Prompt(
            question="What color is the sky?",
            should_abstain=False,
            reference_answers=["blue"],
            metadata={},
        )

        correct_response = Response(
            prompt=prompt,
            response="The sky is blue",
            response_or_abstention="The sky is blue",
        )

        is_correct, _ = correctness_evaluator.is_response_correct(correct_response)

        assert is_correct

        incorrect_response = Response(
            prompt=prompt,
            response="The sky is red and green stripes",
            response_or_abstention="The sky is red and green stripes",
        )

        is_correct, _ = correctness_evaluator.is_response_correct(incorrect_response)

        assert not is_correct
