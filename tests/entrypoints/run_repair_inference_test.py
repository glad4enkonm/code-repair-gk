"""Tests for run_repair_inference.call_llm thinking-disable plumbing.

Gemma-style reasoning models spend the completion budget on reasoning
before emitting the SEARCH/REPLACE block; ``--disable-thinking`` must
attach the llama.cpp chat-template kwarg (verified on
gemma-4-26B-A4B-it-Q8_0) while leaving default request bodies untouched.
"""

import importlib.util
import types
import sys
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = (
    Path(__file__).parent.parent.parent
    / "src"
    / "entrypoints"
    / "run_repair_inference.py"
)
_SRC = _SCRIPT.parent.parent


class _Recorder:
    """OpenAI-like module stub recording chat.completions.create kwargs.

    ``responses`` optionally scripts the content returned per call
    (popped in order); the default answers every call with "ok".
    """

    def __init__(self, responses=None):
        self.requests = []
        self._responses = list(responses) if responses is not None else None

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        content = "ok"
        if self._responses is not None:
            content = self._responses.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    def install(self, monkeypatch):
        fake_openai = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self._create))
        )
        # datasets stub: module-level __getattr__ yields any name imported
        # anywhere in the entrypoint's import chain (load_from_disk, Dataset…)
        fake_datasets = types.ModuleType("datasets")
        fake_datasets.__getattr__ = lambda name: SimpleNamespace()
        # tiktoken is only module-imported by swebench_override (its C
        # extension is unavailable in some sandboxes); a bare stub is
        # enough for the import to succeed. Same for numpy, pulled in by
        # the kg import chain.
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
        monkeypatch.setitem(sys.modules, "tiktoken", SimpleNamespace())
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.__getattr__ = lambda name: SimpleNamespace()
        monkeypatch.setitem(sys.modules, "numpy", fake_numpy)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "run_repair_inference",
        _SCRIPT,
        submodule_search_locations=[],
    )
    # entrypoints.swebench_override must resolve: put src on sys.path
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_call_llm_disable_thinking_adds_extra_body(monkeypatch):
    recorder = _Recorder()
    recorder.install(monkeypatch)
    module = _load_module()
    module.call_llm(
        "gemma4-26b-A4B-Q8_0", "sys\nuser", 8192, 0.0, disable_thinking=True
    )
    assert recorder.requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_call_llm_default_leaves_body_untouched(monkeypatch):
    recorder = _Recorder()
    recorder.install(monkeypatch)
    module = _load_module()
    module.call_llm("qwen3-coder-next-Q8_0", "sys\nuser", 8192, 0.0)
    assert "extra_body" not in recorder.requests[0]


def test_generate_fallback_retries_with_disabled_thinking(monkeypatch):
    recorder = _Recorder(
        responses=["   ", "<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE"]
    )
    recorder.install(monkeypatch)
    module = _load_module()
    response, retried = module.generate_with_thinking_fallback(
        "gemma4-26b-A4B-Q8_0", "sys\nuser", 12288, 0.0, disable_thinking=False
    )
    assert retried is True
    assert response.startswith("<<<<<<< SEARCH")
    assert len(recorder.requests) == 2
    assert "extra_body" not in recorder.requests[0]
    assert recorder.requests[1]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_generate_no_fallback_when_content_present(monkeypatch):
    recorder = _Recorder()
    recorder.install(monkeypatch)
    module = _load_module()
    response, retried = module.generate_with_thinking_fallback(
        "gemma4-26b-A4B-Q8_0", "sys\nuser", 12288, 0.0, disable_thinking=False
    )
    assert retried is False
    assert response == "ok"
    assert len(recorder.requests) == 1


def test_generate_no_fallback_when_thinking_already_disabled(monkeypatch):
    recorder = _Recorder(responses=[""])
    recorder.install(monkeypatch)
    module = _load_module()
    response, retried = module.generate_with_thinking_fallback(
        "gemma4-26b-A4B-Q8_0", "sys\nuser", 12288, 0.0, disable_thinking=True
    )
    assert retried is False
    assert response == ""
    assert len(recorder.requests) == 1
    assert recorder.requests[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
