"""Test the swebench override entry point."""

from unittest.mock import patch


def test_apply_swebench_overrides_injects_model():
    from swebench.inference.run_api import (
        MODEL_COST_PER_INPUT,
        MODEL_COST_PER_OUTPUT,
        MODEL_LIMITS,
    )

    with (
        patch.dict(MODEL_LIMITS, clear=True),
        patch.dict(MODEL_COST_PER_INPUT, clear=True),
        patch.dict(MODEL_COST_PER_OUTPUT, clear=True),
    ):

        from entrypoints.swebench_override import apply_swebench_overrides

        apply_swebench_overrides()

        assert MODEL_LIMITS["qwen3-coder-next-Q8_0"] == 128_000
        assert MODEL_COST_PER_INPUT["qwen3-coder-next-Q8_0"] == 0.00000025
        assert MODEL_COST_PER_OUTPUT["qwen3-coder-next-Q8_0"] == 0.00000125

        assert MODEL_LIMITS["gemma-4-26B-A4B-it-Q8_0"] == 128_000
        assert MODEL_COST_PER_INPUT["gemma-4-26B-A4B-it-Q8_0"] == 0.00000025
        assert MODEL_COST_PER_OUTPUT["gemma-4-26B-A4B-it-Q8_0"] == 0.00000125


def test_config_loads_defaults():
    from kg.config import KGConfig

    c = KGConfig()
    assert c.api_base == "http://localhost:8081/v1"
    assert c.model == "qwen3-coder-next-Q8_0"
