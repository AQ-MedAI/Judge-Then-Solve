"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from recipe.abstention_datasets.abstract_abstention_dataset import DummyDataset, Prompt
from recipe.inference import InferencePipeline, RawResponse, RawResponses
from recipe.models import DummyModel


class TestResponses:
    prompt = Prompt(
        question="Is the sky blue?",
        reference_answers=["yes"],
        should_abstain=False,
        metadata={"test": "test"},
    )

    def test_response_creation(self):
        raw_response = RawResponse(prompt=self.prompt, response="maybe")
        raw_response2 = RawResponse(prompt=self.prompt, response="yes")

        raw_responses = RawResponses(responses=[raw_response, raw_response2])
        assert isinstance(raw_responses.responses[1], RawResponse)

    def test_response_saving(self, tmp_path):
        raw_response = RawResponse(prompt=self.prompt, response="maybe")
        raw_response2 = RawResponse(prompt=self.prompt, response="yes")

        raw_responses = RawResponses(responses=[raw_response, raw_response2])
        raw_responses.save(tmp_path)

        file_path = tmp_path / "RawResponses.json"
        assert file_path.exists()

    def test_loading(self, tmp_path):
        raw_response = RawResponse(prompt=self.prompt, response="maybe")
        raw_response2 = RawResponse(prompt=self.prompt, response="yes")

        raw_responses = RawResponses(responses=[raw_response, raw_response2])
        raw_responses.save(tmp_path)

        loaded_raw_responses = RawResponses.load(tmp_path / "RawResponses.json")
        assert loaded_raw_responses.responses[0].response == "maybe"


class TestInferencePipeline:
    def test_pipeline_run(self, tmp_path):
        model = DummyModel()
        dataset = DummyDataset()
        inference_pipeline = InferencePipeline(
            model, dataset=dataset, save_dir=tmp_path
        )
        raw_responses = inference_pipeline.run()
        assert isinstance(raw_responses, RawResponses)
        assert isinstance(raw_responses.responses[1].response, str)

    def test_pipeline_run_without_indices_subset(self, tmp_path):
        model = DummyModel()

        dataset = DummyDataset()  # Has 100 samples

        inference_pipeline = InferencePipeline(
            model, dataset=dataset, save_dir=tmp_path
        )

        raw_responses = inference_pipeline.run()

        assert len(raw_responses.responses) == 100

    def test_pipeline_run_with_indices_subset(self, tmp_path):
        model = DummyModel()

        dataset = DummyDataset()

        inference_pipeline = InferencePipeline(
            model, dataset=dataset, save_dir=tmp_path
        )

        raw_responses = inference_pipeline.run(indices_subset=[0, 5, 10])

        assert len(raw_responses.responses) == 3
