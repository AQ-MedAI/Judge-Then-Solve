"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from recipe.evaluation import ContainsAbstentionKeywordAbstentionDetector, LLMJudgeAbstentionDetector
from recipe.abstention import Response
from recipe.abstention_datasets.abstract_abstention_dataset import Prompt
from recipe.models import Llama3_1_8B_Instruct
import pytest


class TestContainsAbstentionKeywordAbstentionDetector:
    def test_is_abstention(self):
        abstention_detector = ContainsAbstentionKeywordAbstentionDetector("save_dir")
        prompt = Prompt(
            question="What color is the sky?",
            should_abstain=False,
            reference_answers=["blue", "sky blue"],
            metadata={"question_type": "silly"},
        )
        response1 = Response(
            prompt=prompt,
            response="The sky is blue",
            response_or_abstention="The sky is blue",
            )
        is_abstention1, detector_response1 = abstention_detector.detect_abstention(response1)
        assert is_abstention1 == False

        response2 = Response(
            prompt=prompt,
            response="I don't know",
            response_or_abstention="I don't know",
        )
        is_abstention2, detector_response2 = abstention_detector.detect_abstention(response2)
        assert is_abstention2 == True


class TestLLMJudgeAbstentionDetector:

    @pytest.mark.slow
    def test_is_abstention(self):
        judge_model = Llama3_1_8B_Instruct()
        abstention_detector = LLMJudgeAbstentionDetector(judge_model, use_gt_in_prompt=True, save_dir="save_dir")
        prompt = Prompt(
            question="What color is the sky?",
            should_abstain=False,
            reference_answers=["blue", "sky blue"],
            metadata={"question_type": "silly"},
        )
        response1 = Response(
            prompt=prompt,
            response="The sky is blue",
            response_or_abstention="The sky is blue",
            )
        is_abstention1, detector_response1 = abstention_detector.detect_abstention(response1)
        assert is_abstention1 == False
        response2 = Response(
            prompt=prompt,
            response="I don't know",
            response_or_abstention="I don't know",
        )
        is_abstention2, detector_response2 = abstention_detector.detect_abstention(response2)
        assert is_abstention2 == True
