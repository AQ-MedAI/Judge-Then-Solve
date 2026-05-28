"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from vllm import LLM, SamplingParams
import torch

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "facebook/opt-125m"
# float16 becuase local devfair Quadro GP100 GPU has compute capability 6.0
# which doesn't support bfloat

# on SLURM 
# FlashAttention-2 backend for Volta doesn't work either
# a helpful site https://zilliz.com/blog/building-rag-milvus-vllm-llama-3-1
torch.cuda.empty_cache()

llm = LLM(
    model=model_name,
    dtype=torch.float16,
    gpu_memory_utilization=0.75,
    max_model_len=1000,
    max_num_batched_tokens=3000,
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
