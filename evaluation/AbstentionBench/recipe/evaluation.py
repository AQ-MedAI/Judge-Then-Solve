"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
"""
evaluates whether a response is an abstention 
and whether it's correct to abstain
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

from recipe.abstention import Response, Responses
from recipe.abstention_keywords import ABSTENTION_KEYWORDS
from recipe.evaluation_judge_prompts import (
    LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT,
    LLM_JUDGE_CORRECTNESS_PROMPT,
    LLM_JUDGE_MATH_CORRECTNESS_PROMPT,
)
from recipe.models import InferenceModel


class AbstentionEvaluator(ABC):
    """Determines whether the model should abstain given the question.
    Really only meaningful for correctness;
    otherwise, ground-truth comes from the dataset
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def is_abstention_correct(self, response: Response) -> bool:
        pass

    def run(self, responses: Responses) -> Responses:
        responses_list = []
        for response in responses.responses:
            response.is_abstention_correct = self.is_abstention_correct(response)
            responses_list.append(response)

        responses = Responses(responses=responses_list)
        responses.save(self.save_dir, self.__class__.__name__)
        return responses


class GroundTruthAbstentionEvaluator(AbstentionEvaluator):
    def is_abstention_correct(self, response: Response) -> bool:
        return response.is_abstention == response.prompt.should_abstain


# This class is needed due to saving logic of AbstentionEvaluator being
# dependent on the class name.
class GroundTruthAbstentionEvaluatorWithReasoning(GroundTruthAbstentionEvaluator):
    pass


class AbstentionDetector(ABC):
    """Determines whether a response constitutes an abstention"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def detect_abstention(
        self, response: Response
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Detects abstention in model's response, returns is_abstention label
        which is bool or None (in case detector wasn't able to determine abstention label)
        and optionally a longer AbstentionDetector's response (relevant for LLM judge).
        """
        pass

    def run(self, responses: Responses) -> Responses:
        responses_list = []
        for response in responses.responses:
            response.is_abstention, response.full_judge_response = (
                self.detect_abstention(response)
            )
            responses_list.append(response)

        responses = Responses(responses=responses_list)
        responses.save(self.save_dir, self.__class__.__name__)
        return responses


class ContainsAbstentionKeywordAbstentionDetector(AbstentionDetector):

    def detect_abstention(self, response: Response) -> Tuple[bool, Optional[str]]:
        """search in response for keywords"""
        response_lower = response.response_or_abstention.lower()
        for keyword in ABSTENTION_KEYWORDS:
            if keyword.lower() in response_lower:
                # Second returned value corresponds to
                # full judge response, returning None
                # since this is not an LLM judge.
                return True, None

        return False, None


class LLMJudgeAbstentionDetector(AbstentionDetector):
    USE_RESPONSE_WITH_REASONING = False

    def __init__(
        self, judge_model: InferenceModel, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._judge_model = judge_model

    def detect_abstention(self, response: Response) -> Tuple[Optional[bool], str]:
        # Reference answers are list or None
        if response.prompt.reference_answers:
            reference_answers = (
                "[" + "; ".join(response.prompt.reference_answers) + "]"
            )
        else:
            reference_answers = "[]"

        model_answer = (
            response.response_with_reasoning
            if self.USE_RESPONSE_WITH_REASONING
            else response.response_or_abstention
        )
        judge_prompt = LLM_JUDGE_ABSTENTION_COCONOT_STYLE_PROMPT.format(
            question=response.prompt.question,
            ref_answer=reference_answers,
            abstention_label=response.prompt.should_abstain,
            model_answer=model_answer,
        )

        # Temporary workaround for batched inference
        judge_response = self._judge_model.respond([judge_prompt])[0]
        judge_response = judge_response.lower().strip(" .,\n")
        if judge_response not in ["yes", "no"]:
            logger.warning(
                f"\nUnexpected judge response:\n{judge_response}"
                f"\nJudge prompt:\n{judge_prompt}"
            )
            return None, judge_response
        return judge_response.lower() == "yes", judge_response


class LLMJudgeAbstentionDetectorWithReasoning(LLMJudgeAbstentionDetector):
    USE_RESPONSE_WITH_REASONING = True


class CorrectnessEvaluator(ABC):
    """Determines whether a response is correct, according to the dataset's reference answers"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir

    @abstractmethod
    def is_response_correct(
        self, response: Response
    ) -> Tuple[Optional[bool], Optional[str]]:
        """Determine's whether the response is correct, returning is_response correct label
        which is bool or None (in case no reference answers are given), and optionally a
        longer AbstentionDetector's response (relevant for LLM judge).
        """
        pass

    def run(self, responses: Responses) -> Responses:

        responses_list = []
        for response in responses.responses:
            response.is_response_correct, response.full_correctness_judge_response = (
                self.is_response_correct(response)
            )
            responses_list.append(response)

        responses = Responses(responses=responses_list)
        responses.save(self.save_dir, self.__class__.__name__)


class LLMJudgeCorrectnessEvaluator(CorrectnessEvaluator):

    def __init__(self, judge_model: InferenceModel, math_mode=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._judge_model = judge_model
        self.math_mode = math_mode

    def is_response_correct(self, response: Response) -> Tuple[Optional[bool], str]:
        if response.prompt.reference_answers is None:
            return (None, "")

        if not self.math_mode:
            reference_answers = "\n".join(response.prompt.reference_answers)

            judge_prompt = LLM_JUDGE_CORRECTNESS_PROMPT.format(
                question=response.prompt.question,
                ref_answer=reference_answers,
                model_answer=response.response_or_abstention,
            )
        else:
            reference_answers = "; ".join(response.prompt.reference_answers)

            judge_prompt = LLM_JUDGE_MATH_CORRECTNESS_PROMPT.format(
                question=response.prompt.question,
                ref_answer=reference_answers.strip(" \n"),
                model_answer=response.response_or_abstention.strip(" \n"),
            )

        # Temporary workaround for batched inference
        judge_response = self._judge_model.respond([judge_prompt])[0]
        judge_response = judge_response.lower().strip(" .,\n")
        if judge_response not in ["correct", "incorrect"]:
            logger.warning(
                f"\nUnexpected judge response:\n{judge_response}"
                f"\nJudge prompt:\n{judge_prompt}"
            )
            return None, judge_response
        return judge_response.lower() == "correct", judge_response
