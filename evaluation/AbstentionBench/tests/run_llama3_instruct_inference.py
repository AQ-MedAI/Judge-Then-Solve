"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import torch

from recipe.models import Llama3_1_8B_Instruct

torch.cuda.empty_cache()

llm = Llama3_1_8B_Instruct(convert_prompt_to_chat=True)
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

for prompt in prompts:
    output = llm.respond(prompt)
    print(f"Prompt: {prompt!r}, Generated text: {output!r}")
