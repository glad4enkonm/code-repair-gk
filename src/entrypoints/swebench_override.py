import openai
import tiktoken


def apply_swebench_overrides():
    from swebench.inference.run_api import (
        MODEL_COST_PER_INPUT,
        MODEL_COST_PER_OUTPUT,
        MODEL_LIMITS,
    )

    MODEL_LIMITS["qwen3-coder-next-Q8_0"] = 128_000
    MODEL_COST_PER_INPUT["qwen3-coder-next-Q8_0"] = 0.00000025
    MODEL_COST_PER_OUTPUT["qwen3-coder-next-Q8_0"] = 0.00000125
    tiktoken.model.MODEL_TO_ENCODING["qwen3-coder-next-Q8_0"] = "cl100k_base"

    MODEL_LIMITS["gemma-4-26B-A4B-it-Q8_0"] = 128_000
    MODEL_COST_PER_INPUT["gemma-4-26B-A4B-it-Q8_0"] = 0.00000025
    MODEL_COST_PER_OUTPUT["gemma-4-26B-A4B-it-Q8_0"] = 0.00000125
    tiktoken.model.MODEL_TO_ENCODING["gemma-4-26B-A4B-it-Q8_0"] = "cl100k_base"

    MODEL_LIMITS["deepseek/deepseek-v4-flash"] = 128_000
    MODEL_COST_PER_INPUT["deepseek/deepseek-v4-flash"] = 0.00000025
    MODEL_COST_PER_OUTPUT["deepseek/deepseek-v4-flash"] = 0.00000125
    tiktoken.model.MODEL_TO_ENCODING["deepseek/deepseek-v4-flash"] = "cl100k_base"

    openai.timeout = 7200
