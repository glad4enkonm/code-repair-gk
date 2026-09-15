from kg.config import KGConfig


def test_default_config():
    c = KGConfig()
    assert c.model == "qwen3-coder-next-Q8_0"
    assert c.max_rounds is not None
    assert c.max_tool_calls is not None
    assert c.max_context_tokens == 128_000
    assert c.prompt_style == "style-3"


def test_config_from_file(tmp_path):
    import json

    d = {
        "model": "test-model",
        "max_rounds": 4,
        "temperature": 0.5,
        "max_tool_calls": 7,
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(d))
    c = KGConfig.from_file(str(p))
    assert c.model == "test-model"
    assert c.max_rounds == 4
    assert c.temperature == 0.5
    assert c.max_tool_calls == 7  # overridden by file, not default


def test_meta_path():
    c = KGConfig(meta_schema_dir="foo/bar")
    assert c.meta_path == "foo/bar/meta.json"
    assert c.origin_path == "foo/bar/origin.json"
