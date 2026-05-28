"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pytest

from recipe.models import (
    DummyModel,
    Gemini15ProAPI,
    GPT4oAPI,
    Llama3_1_8B_Base,
    Llama3_1_8B_Instruct,
    Llama3_1_8B_Tulu_3_DPO,
    Llama3_1_8B_Tulu_3_PPO_RLVF,
    Llama3_1_8B_Tulu_3_SFT,
    Mistral_7B_Instruct_v0_3,
    OLMo_7B_0724_Instruct,
    TinyLlamaChat,
    o1API,
)


class TestModelResponses:
    faster_model_classes = [
        DummyModel,
        TinyLlamaChat,
    ]
    slower_model_classes = [
        Llama3_1_8B_Instruct,
        OLMo_7B_0724_Instruct,
        Mistral_7B_Instruct_v0_3,
        Llama3_1_8B_Base,
        Llama3_1_8B_Instruct,
        Llama3_1_8B_Tulu_3_SFT,
        Llama3_1_8B_Tulu_3_DPO,
        Llama3_1_8B_Tulu_3_PPO_RLVF,
    ]
    remote_api_model_classes = [
        GPT4oAPI,
        o1API,
        Gemini15ProAPI,
    ]

    def check_respond(self, model_class):
        model = model_class()
        responses = model.respond(["Why is the sky blue?", "Why is the grass green?"])
        assert isinstance(responses, list)
        assert isinstance(responses[0], str)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_class", slower_model_classes)
    def test_bigger_model_respond(self, model_class):
        self.check_respond(model_class)

    @pytest.mark.parametrize("model_class", faster_model_classes)
    def test_smaller_model_respond(self, model_class):
        self.check_respond(model_class)

    @pytest.mark.api
    @pytest.mark.parametrize("model_class", remote_api_model_classes)
    def test_remote_api_model_respond(self, model_class):
        self.check_respond(model_class)

