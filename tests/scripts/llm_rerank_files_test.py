"""Tests for llm-rerank-files.py (comparator, ranking, checkpointing, loaders)."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "llm-rerank-files.py"
_spec = importlib.util.spec_from_file_location("lrf", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


# --- fakes ---
class FakeClient:
    """OpenAI-like client returning canned completion contents in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        content = self.responses.pop(0) if self.responses else "FILE_A"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _mk_comparator(client, **kwargs):
    params = {"retries": 3, "max_file_lines": 400}
    params.update(kwargs)
    return _mod.LLMComparator(client=client, model="m", **params)


# --- resolve_kg_input_path ---
def test_resolve_kg_input_path_defaults_to_cfl():
    default = _mod.cfl.kg_progress_path("file_level", "embed")
    assert _mod.resolve_kg_input_path(None, "embed") == default


def test_resolve_kg_input_path_explicit_overrides():
    explicit = "outputs/gemma26b/data__swe-bench_lite__style-3__fs-kg__k-auto__embed.progress.jsonl"
    assert _mod.resolve_kg_input_path(explicit, "embed") == explicit


# --- extract_files_with_content ---
def test_extract_files_with_content_order_and_readme_filter():
    text = (
        "[start of README.md]\nreadme\n[end of README.md]\n"
        "[start of src/foo.py]\nfoo line\n[end of src/foo.py]\n"
        "[start of src/bar.py]\nbar line\n[end of src/bar.py]\n"
    )
    result = _mod.extract_files_with_content(text)
    assert result == [
        ("src/foo.py", "foo line"),
        ("src/bar.py", "bar line"),
    ]


def test_extract_files_multiline_content():
    text = "[start of a.py]\nline1\nline2\nline3\n[end of a.py]\n"
    result = _mod.extract_files_with_content(text)
    assert result == [("a.py", "line1\nline2\nline3")]


def test_extract_files_missing_end_marker_keeps_empty_content():
    text = "[start of a.py]\ncode without end\n[start of b.py]\nok\n[end of b.py]\n"
    result = _mod.extract_files_with_content(text)
    assert result == [("a.py", ""), ("b.py", "ok")]


def test_extract_files_empty_text():
    assert _mod.extract_files_with_content("") == []


# --- truncate_content ---
def test_truncate_content_caps_lines():
    content = "\n".join(f"line{i}" for i in range(10))
    truncated = _mod.truncate_content(content, 3)
    assert truncated == "line0\nline1\nline2"


def test_truncate_content_short_untouched():
    assert _mod.truncate_content("a\nb", 5) == "a\nb"


# --- parse_comparator_choice ---
def test_parse_choice_plain():
    assert _mod.parse_comparator_choice("FILE_A") == -1
    assert _mod.parse_comparator_choice("FILE_B") == 1


def test_parse_choice_case_and_decoration():
    assert _mod.parse_comparator_choice("file_a") == -1
    assert _mod.parse_comparator_choice("**FILE_B**") == 1
    assert _mod.parse_comparator_choice("Answer: FILE_A") == -1


def test_parse_choice_bare_letter_exact():
    assert _mod.parse_comparator_choice("A") == -1
    assert _mod.parse_comparator_choice(" b ") == 1


def test_parse_choice_garbage():
    assert _mod.parse_comparator_choice("both are relevant") is None
    assert _mod.parse_comparator_choice("FILE_A and FILE_B") is None
    assert _mod.parse_comparator_choice("") is None
    assert _mod.parse_comparator_choice(None) is None


# --- LLMComparator ---
def test_comparator_file_a_wins():
    client = FakeClient(["FILE_A"])
    comparator = _mk_comparator(client)
    result = comparator.compare("issue", ("a.py", "x"), ("b.py", "y"))
    assert result == -1
    assert client.calls[0]["model"] == "m"
    assert comparator.n_llm_calls == 1


def test_comparator_max_tokens_allows_reasoning_models():
    client = FakeClient(["FILE_A"])
    comparator = _mk_comparator(client)
    comparator.compare("issue", ("a.py", "x"), ("b.py", "y"))
    assert client.calls[0]["max_tokens"] == _mod._MAX_COMPLETION_TOKENS
    # Thinking models (e.g. gemma) spend the budget on reasoning_content
    # before emitting the one-token answer; 64 was too small for them.
    assert _mod._MAX_COMPLETION_TOKENS >= 512


def test_comparator_retry_ladder_disables_thinking_then_raises_temp():
    client = FakeClient(["garbage", "garbage", "FILE_B"])
    comparator = _mk_comparator(client, retries=3)
    assert comparator.compare("issue", ("a.py", "x"), ("b.py", "y")) == 1
    temps = [c["temperature"] for c in client.calls]
    extras = [c.get("extra_body") for c in client.calls]
    # attempt 1: plain, deterministic
    # attempt 2: thinking disabled (reasoning can exhaust the completion
    #            budget on thinking models before the answer token)
    # attempt 3+: thinking disabled + resampled at higher temperature
    assert temps == [0.0, 0.0, 0.7]
    assert extras[0] is None
    assert extras[1] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert extras[2] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_comparator_nothink_fallback_recovers_on_second_attempt():
    client = FakeClient(["garbage", "FILE_B"])
    comparator = _mk_comparator(client, retries=3)
    assert comparator.compare("issue", ("a.py", "x"), ("b.py", "y")) == 1
    assert len(client.calls) == 2
    assert client.calls[1]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    assert client.calls[1]["temperature"] == 0.0


def test_comparator_api_error_is_retried_not_fatal():
    class FlakyClient(FakeClient):
        def _create(self, **kwargs):
            if not self.calls:
                self.calls.append(kwargs)
                raise RuntimeError("endpoint 500")
            return super()._create(**kwargs)

    client = FlakyClient(["FILE_A"])
    comparator = _mk_comparator(client, retries=3)
    assert comparator.compare("issue", ("a.py", "x"), ("b.py", "y")) == -1
    assert len(client.calls) == 2


def test_comparator_pair_cache_reversed_pair_no_second_call():
    client = FakeClient(["FILE_B"])
    comparator = _mk_comparator(client)
    assert comparator.compare("issue", ("a.py", "x"), ("b.py", "y")) == 1
    assert comparator.compare("issue", ("b.py", "y"), ("a.py", "x")) == -1
    assert len(client.calls) == 1


def test_comparator_retry_then_success():
    client = FakeClient(["garbage", "FILE_B"])
    comparator = _mk_comparator(client, retries=3)
    assert comparator.compare("issue", ("a.py", "x"), ("b.py", "y")) == 1
    assert len(client.calls) == 2


def test_comparator_gives_up_after_all_retries():
    client = FakeClient(["nope", "nope"])
    comparator = _mk_comparator(client, retries=2)
    try:
        comparator.compare("issue", ("a.py", "x"), ("b.py", "y"))
        raise AssertionError("expected ComparatorError")
    except _mod.ComparatorError:
        pass
    assert len(client.calls) == 2


def test_comparator_identical_paths_no_call():
    client = FakeClient([])
    comparator = _mk_comparator(client)
    assert comparator.compare("issue", ("a.py", "x"), ("a.py", "x")) == 0
    assert client.calls == []


def test_comparator_prompt_contains_issue_paths_truncated_content():
    client = FakeClient(["FILE_A"])
    comparator = _mk_comparator(client, max_file_lines=2)
    long_content = "\n".join(f"l{i}" for i in range(5))
    comparator.compare("the issue text", ("a.py", long_content), ("b.py", "c"))
    prompt = client.calls[0]["messages"][-1]["content"]
    assert "the issue text" in prompt
    assert "a.py" in prompt and "b.py" in prompt
    assert "l0" in prompt and "l1" in prompt
    assert "l2" not in prompt
    assert "FILE_A" in prompt and "FILE_B" in prompt


def test_comparator_prompt_has_few_shot_examples():
    client = FakeClient(["FILE_A"])
    comparator = _mk_comparator(client)
    comparator.compare("issue", ("a.py", "x"), ("b.py", "y"))
    prompt = client.calls[0]["messages"][-1]["content"]
    # correct example answers
    assert "Correct: FILE_A" in prompt
    assert "Correct: FILE_B" in prompt
    # incorrect answer patterns called out explicitly
    assert "Neither file" in prompt
    assert "FILE_A because" in prompt
    assert "FILE_A FILE_B" in prompt
    # forced choice rule for irrelevant pairs
    assert "MORE RELATED" in prompt


# --- rank_files ---
class StubComparator:
    """Duck-typed comparator: prefers the path containing 'core'."""

    def __init__(self, fail_after=0):
        self.fail_after = fail_after
        self.calls = 0

    def compare(self, issue, file_a, file_b):
        self.calls += 1
        if self.fail_after and self.calls >= self.fail_after:
            raise _mod.ComparatorError("boom")
        a_win = "core" in file_a[0]
        b_win = "core" in file_b[0]
        if a_win == b_win:
            return 0
        return -1 if a_win else 1


def test_rank_files_orders_by_comparator():
    files = [("src/util.py", "u"), ("src/core.py", "c"), ("docs/x.py", "x")]
    result = _mod.rank_files("issue", files, StubComparator())
    assert result["ranked"][0] == "src/core.py"
    assert result["failed"] is False
    assert result["n_comparisons"] > 0


def test_rank_files_failure_keeps_original_order_and_stops_calls():
    files = [("a.py", "1"), ("b.py", "2"), ("c.py", "3"), ("d.py", "4")]
    stub = StubComparator(fail_after=2)
    result = _mod.rank_files("issue", files, stub)
    assert result["failed"] is True
    assert result["ranked"] == ["a.py", "b.py", "c.py", "d.py"]
    calls_at_failure = stub.calls
    assert calls_at_failure == 2


def test_rank_files_single_file_no_comparisons():
    result = _mod.rank_files("issue", [("a.py", "1")], StubComparator())
    assert result["ranked"] == ["a.py"]
    assert result["n_comparisons"] == 0
    assert result["failed"] is False


def test_rank_files_all_equal_is_stable():
    files = [("z.py", "1"), ("m.py", "2"), ("a.py", "3")]
    result = _mod.rank_files("issue", files, StubComparator())
    assert result["ranked"] == ["z.py", "m.py", "a.py"]


# --- estimate_comparisons ---
def test_estimate_comparisons():
    assert _mod.estimate_comparisons(0) == 0
    assert _mod.estimate_comparisons(1) == 0
    assert _mod.estimate_comparisons(4) > _mod.estimate_comparisons(2)
    assert _mod.estimate_comparisons(2) >= 1


# --- load_rerank ---
def _write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_load_rerank(tmp_path):
    path = tmp_path / "rerank.progress.jsonl"
    _write_jsonl(
        path,
        [
            {"instance_id": "i1", "ranked_files": ["b.py", "a.py"]},
            {"instance_id": "i2", "ranked_files": ["c.py"], "failed": True},
        ],
    )
    result = _mod.load_rerank(str(path))
    assert result == {"i1": ["b.py", "a.py"], "i2": ["c.py"]}


def test_load_rerank_skips_blank_lines(tmp_path):
    path = tmp_path / "rerank.progress.jsonl"
    path.write_text('{"instance_id": "i1", "ranked_files": ["a.py"]}\n\n')
    assert _mod.load_rerank(str(path)) == {"i1": ["a.py"]}


# --- process_instances: checkpointing + resume ---
def test_process_instances_writes_progress_and_resumes(tmp_path):
    progress = tmp_path / "rerank.progress.jsonl"
    rows = [
        {
            "instance_id": "i1",
            "problem_statement": "issue one",
            "text_inputs": "[start of a.py]\nx\n[end of a.py]\n",
        },
        {
            "instance_id": "i2",
            "problem_statement": "issue two",
            "text_inputs": "[start of b.py]\ny\n[end of b.py]\n",
        },
    ]
    factory_calls = []

    def factory():
        factory_calls.append(1)
        return StubComparator()

    _mod.process_instances(rows, factory, str(progress), limit=1)
    lines = progress.read_text().strip().splitlines()
    assert len(lines) == 1
    first = json.loads(lines[0])
    assert first["instance_id"] == "i1"
    assert first["ranked_files"] == ["a.py"]
    assert first["failed"] is False
    assert first["n_files"] == 1

    done_before = len(factory_calls)
    _mod.process_instances(rows, factory, str(progress), limit=0)
    assert progress.read_text().count("\n") == 2
    assert len(factory_calls) == done_before + 1


def test_process_instances_dry_run_no_llm_no_file(tmp_path):
    progress = tmp_path / "rerank.progress.jsonl"
    rows = [
        {
            "instance_id": "i1",
            "problem_statement": "issue",
            "text_inputs": "[start of a.py]\nx\n[end of a.py]\n[start of b.py]\ny\n[end of b.py]\n",
        },
    ]

    def factory():
        raise AssertionError("factory must not be called in dry-run")

    stats = _mod.process_instances(rows, factory, str(progress), limit=0, dry_run=True)
    assert stats["instances"] == 1
    assert stats["estimated_comparisons"] >= 1
    assert not progress.exists()


def test_process_instances_records_failure_row(tmp_path):
    progress = tmp_path / "rerank.progress.jsonl"
    rows = [
        {
            "instance_id": "i1",
            "problem_statement": "issue",
            "text_inputs": "[start of a.py]\nx\n[end of a.py]\n[start of b.py]\ny\n[end of b.py]\n",
        },
    ]
    _mod.process_instances(
        rows, lambda: StubComparator(fail_after=1), str(progress), limit=0
    )
    row = json.loads(progress.read_text().strip())
    assert row["failed"] is True
    assert row["ranked_files"] == ["a.py", "b.py"]


# --- evaluation wiring (integration with compare-file-localization) ---
def test_rerank_improves_recall_via_cfl_metrics(tmp_path):
    path = tmp_path / "rerank.progress.jsonl"
    _write_jsonl(
        path,
        [
            {"instance_id": "i1", "ranked_files": ["b.py", "a.py"]},
        ],
    )
    rerank = _mod.load_rerank(str(path))
    gt = {"i1": {"b.py"}}
    original = {"i1": ["a.py", "b.py"]}
    orig_rows = _mod.cfl.evaluate_method(original, gt, ["i1"], "kg_file_level")
    rerank_rows = _mod.cfl.evaluate_method(rerank, gt, ["i1"], "kg_file_level_llm")
    assert _mod.cfl.aggregate(orig_rows)["recall@1"] == 0.0
    assert _mod.cfl.aggregate(rerank_rows)["recall@1"] == 1.0


# --- filter_rows_by_instance (debugging aid for --instance) ---
def test_filter_rows_by_instance():
    rows = [
        {"instance_id": "a", "text_inputs": ""},
        {"instance_id": "b", "text_inputs": ""},
    ]
    assert _mod.filter_rows_by_instance(rows, None) == rows
    assert _mod.filter_rows_by_instance(rows, "b") == [rows[1]]
    assert _mod.filter_rows_by_instance(rows, "zz") == []


# --- select_rankable_rows (restart-row filtering) ---
def test_select_rankable_rows_drops_empty_restart_attempts():
    rows = [
        {"instance_id": "a", "text_inputs": "", "touched_nodes": []},
        {"instance_id": "a", "text_inputs": "ctx", "touched_nodes": ["n1"]},
        {"instance_id": "b", "text_inputs": "ctx2", "touched_nodes": ["n2"]},
        {"instance_id": "c", "text_inputs": "", "touched_nodes": []},
    ]
    picked = _mod.select_rankable_rows(rows)
    assert [r["instance_id"] for r in picked] == ["a", "b"]
    assert picked[0]["touched_nodes"] == ["n1"]


def test_select_rankable_rows_last_attempt_wins():
    rows = [
        {"instance_id": "a", "text_inputs": "old", "touched_nodes": ["old"]},
        {"instance_id": "a", "text_inputs": "new", "touched_nodes": ["new"]},
    ]
    picked = _mod.select_rankable_rows(rows)
    assert picked == [
        {"instance_id": "a", "text_inputs": "new", "touched_nodes": ["new"]}
    ]


def test_select_rankable_rows_empty_input():
    assert _mod.select_rankable_rows([]) == []


# --- defaults from kg_config.json ---
def test_load_defaults_reads_kg_config():
    defaults = _mod._load_defaults()
    assert defaults.get("model") == "qwen3-coder-next-Q8_0"
    assert defaults.get("api_base") == "http://localhost:8081/v1"


# --- _sanitize_model_name ---
def test_sanitize_model_name_plain_name_untouched():
    assert (
        _mod._sanitize_model_name("gemma-4-26B-A4B-it-Q8_0")
        == "gemma-4-26B-A4B-it-Q8_0"
    )
    assert _mod._sanitize_model_name("qwen3-coder-next-Q8_0") == "qwen3-coder-next-Q8_0"


def test_sanitize_model_name_slash_becomes_dash():
    assert (
        _mod._sanitize_model_name("deepseek/deepseek-v4-flash")
        == "deepseek-deepseek-v4-flash"
    )


def test_sanitize_model_name_empty_and_unsafe_chars():
    assert _mod._sanitize_model_name("") == "model"
    assert _mod._sanitize_model_name("a b:c") == "a-b-c"


# --- default progress path carries the model ---
def test_default_progress_path_includes_split_and_model():
    path = _mod.default_progress_path("dev", "gemma-4-26B-A4B-it-Q8_0")
    assert path.endswith(
        "outputs/llm_rerank__file_level__dev__gemma-4-26B-A4B-it-Q8_0.progress.jsonl"
    )


def test_default_progress_path_sanitizes_model():
    path = _mod.default_progress_path("dev", "deepseek/deepseek-v4-flash")
    assert path.endswith(
        "outputs/llm_rerank__file_level__dev__"
        "deepseek-deepseek-v4-flash.progress.jsonl"
    )


# --- progress rows record model provenance ---
def test_process_instances_records_model_in_rows(tmp_path):
    progress = tmp_path / "rerank.progress.jsonl"
    rows = [
        {
            "instance_id": "i1",
            "problem_statement": "issue",
            "text_inputs": "[start of a.py]\nx\n[end of a.py]\n",
        },
    ]
    _mod.process_instances(
        rows, StubComparator, str(progress), limit=0, model="gemma-4-26B-A4B-it-Q8_0"
    )
    row = json.loads(progress.read_text().strip())
    assert row["model"] == "gemma-4-26B-A4B-it-Q8_0"


def test_load_rerank_models_reports_provenance(tmp_path):
    path = tmp_path / "rerank.progress.jsonl"
    _write_jsonl(
        path,
        [
            {
                "instance_id": "i1",
                "ranked_files": ["a.py"],
                "model": "qwen3-coder-next-Q8_0",
            },
            {
                "instance_id": "i2",
                "ranked_files": ["b.py"],
                "model": "gemma-4-26B-A4B-it-Q8_0",
            },
            {"instance_id": "i3", "ranked_files": ["c.py"]},
        ],
    )
    assert _mod.load_rerank_models(str(path)) == {
        "qwen3-coder-next-Q8_0",
        "gemma-4-26B-A4B-it-Q8_0",
    }


# ===========================================================================
# List mode (--mode list): signals spec, assembly, prompt, ranking
# ===========================================================================


# --- parse_signals ---
def test_parse_signals_default_expands_sim():
    # "sim" is the umbrella token for both similarity aggregations
    assert _mod.parse_signals("freq,sim,parts") == [
        "freq",
        "sim_max",
        "sim_mean",
        "parts",
    ]


def test_parse_signals_granular_sim_tokens():
    assert _mod.parse_signals("freq,sim_max,parts") == ["freq", "sim_max", "parts"]
    assert _mod.parse_signals("freq,sim_mean") == ["freq", "sim_mean"]


def test_parse_signals_strips_whitespace_and_normalizes_order():
    assert _mod.parse_signals("parts, freq ,sim") == [
        "freq",
        "sim_max",
        "sim_mean",
        "parts",
    ]


def test_parse_signals_rejects_invalid_token():
    import pytest

    with pytest.raises(ValueError, match="unknown signal"):
        _mod.parse_signals("freq,bogus")


def test_parse_signals_empty_string_rejected():
    import pytest

    with pytest.raises(ValueError):
        _mod.parse_signals("")


# --- signals tag (filenames / eval labels) ---
def test_signals_tag_joins_normalized_subset():
    assert (
        _mod.signals_tag(["freq", "sim_max", "sim_mean", "parts"])
        == "freq-simmax-simmean-parts"
    )
    assert _mod.signals_tag(["freq", "parts"]) == "freq-parts"
    assert _mod.signals_tag(["freq"]) == "freq"


def test_default_progress_path_list_mode_encodes_signals():
    path = _mod.default_progress_path(
        "dev",
        "qwen3-coder-next-Q8_0",
        variant="embed",
        mode="list",
        signals=["freq", "sim_max", "sim_mean", "parts"],
    )
    assert path.endswith(
        "outputs/llm_rerank__file_level__embed__list__"
        "freq-simmax-simmean-parts__dev__qwen3-coder-next-Q8_0.progress.jsonl"
    )


def test_default_progress_path_pairwise_mode_unchanged():
    # Backwards compatibility: pairwise keeps the existing filename shape
    path = _mod.default_progress_path("dev", "m", variant="plain")
    assert path.endswith("outputs/llm_rerank__file_level__dev__m.progress.jsonl")


def test_list_eval_method_label():
    assert (
        _mod.list_eval_method("kg_file_level", ["freq", "sim_max", "parts"])
        == "kg_file_level_llm_list__freq-simmax-parts"
    )
    assert (
        _mod.list_eval_method("kg_embed_file_level", ["freq"])
        == "kg_embed_file_level_llm_list__freq"
    )


# --- graph dir derivation ---
def test_graph_dir_for_row_mirrors_build_graphs_key():
    row = {"repo": "sqlfluff/sqlfluff", "base_commit": "abc123"}
    assert _mod.graph_dir_for_row(row, "outputs/kg_graphs") == (
        "outputs/kg_graphs/sqlfluff__sqlfluff__abc123"
    )


# --- assemble_file_signals ---
def _write_mini_graph(graphs_dir: Path) -> str:
    """Two files; foo.py holds Class Auth + Method login, bar.py holds f."""
    graph_dir = graphs_dir / "demo__repo__abc123"
    graph_dir.mkdir(parents=True)
    data = {
        "nodes": [
            {"id": "File-foo.py", "label": "foo.py"},
            {"id": "Class-foo.py-Auth", "label": "Auth"},
            {"id": "Method-foo.py-Auth-login", "label": "login"},
            {"id": "File-bar.py", "label": "bar.py"},
            {"id": "Function-bar.py-f", "label": "f"},
        ],
        "edges": [
            {
                "id": "e1",
                "label": "CONTAINS",
                "source": "File-foo.py",
                "target": "Class-foo.py-Auth",
            },
            {
                "id": "e2",
                "label": "CONTAINS",
                "source": "Class-foo.py-Auth",
                "target": "Method-foo.py-Auth-login",
            },
            {
                "id": "e3",
                "label": "CONTAINS",
                "source": "File-bar.py",
                "target": "Function-bar.py-f",
            },
        ],
        "allValues": {
            "File-foo.py": {"meta_type": "File", "relative_path": "foo.py"},
            "Class-foo.py-Auth": {
                "meta_type": "Class",
                "name": "Auth",
                "docstring": "auth helpers\nlong doc",
                "source_code": "class Auth:\n    pass",
                "line_start": 1,
                "line_end": 2,
            },
            "Method-foo.py-Auth-login": {
                "meta_type": "Method",
                "name": "login",
                "docstring": "auth login",
                "source_code": "def login(self):\n    pass",
                "line_start": 3,
                "line_end": 4,
            },
            "File-bar.py": {"meta_type": "File", "relative_path": "bar.py"},
            "Function-bar.py-f": {
                "meta_type": "Function",
                "name": "f",
                "source_code": "def f():\n    pass",
                "line_start": 1,
                "line_end": 2,
            },
        },
    }
    (graph_dir / "data.json").write_text(json.dumps(data))
    return str(graph_dir)


def _mini_row():
    return {
        "instance_id": "demo__repo-1",
        "repo": "demo/repo",
        "base_commit": "abc123",
        "problem_statement": "auth is broken",
        "text_inputs": (
            "[start of foo.py]\nx\n[end of foo.py]\n[start of bar.py]\ny\n[end of bar.py]\n"
        ),
        "touched_nodes": [
            "Class-foo.py-Auth",
            "Method-foo.py-Auth-login",
            "Function-bar.py-f",
        ],
    }


def test_assemble_file_signals_freq_and_parts(tmp_path):
    graph_dir = _write_mini_graph(tmp_path)
    signals = _mod.assemble_file_signals(
        _mini_row(), graph_dir, subparts=3, src_chars=120, sim_provider=None
    )
    by_path = {fs["path"]: fs for fs in signals}
    assert by_path["foo.py"]["touch_freq"] == 2
    assert by_path["bar.py"]["touch_freq"] == 1
    # parts carry type, name, docstring first line, line range, source
    auth = next(p for p in by_path["foo.py"]["parts"] if p["name"] == "Auth")
    assert auth["meta_type"] == "Class"
    assert auth["docstring"] == "auth helpers"
    assert auth["line_start"] == 1 and auth["line_end"] == 2
    assert auth["source"].startswith("class Auth:")


def test_assemble_file_signals_subparts_cap(tmp_path):
    graph_dir = _write_mini_graph(tmp_path)
    signals = _mod.assemble_file_signals(
        _mini_row(), graph_dir, subparts=1, src_chars=120, sim_provider=None
    )
    by_path = {fs["path"]: fs for fs in signals}
    assert len(by_path["foo.py"]["parts"]) == 1
    # freq order decides which node survives the cap: Auth (freq from both
    # nodes is per-file; node order is traversal order — Class first)
    assert by_path["foo.py"]["parts"][0]["name"] == "Auth"


def test_assemble_file_signals_src_truncated(tmp_path):
    graph_dir = _write_mini_graph(tmp_path)
    signals = _mod.assemble_file_signals(
        _mini_row(), graph_dir, subparts=3, src_chars=10, sim_provider=None
    )
    by_path = {fs["path"]: fs for fs in signals}
    for part in by_path["foo.py"]["parts"]:
        assert len(part["source"]) <= 10


def test_assemble_file_signals_sim_aggregation(tmp_path, monkeypatch):
    graph_dir = _write_mini_graph(tmp_path)

    def fake_provider(problem_statement, graph_dir_arg):
        # sims per node: login=0.9, Auth=0.5, f=0.3
        return {
            "Method-foo.py-Auth-login": 0.9,
            "Class-foo.py-Auth": 0.5,
            "Function-bar.py-f": 0.3,
        }

    signals = _mod.assemble_file_signals(
        _mini_row(),
        graph_dir,
        subparts=3,
        src_chars=120,
        sim_provider=fake_provider,
    )
    by_path = {fs["path"]: fs for fs in signals}
    assert by_path["foo.py"]["sim_max"] == 0.9
    assert by_path["foo.py"]["sim_mean"] == round((0.9 + 0.5) / 2, 4)
    assert by_path["bar.py"]["sim_max"] == 0.3
    assert by_path["bar.py"]["sim_mean"] == 0.3


def test_assemble_file_signals_sim_provider_failure_omits_sims(tmp_path, monkeypatch):
    graph_dir = _write_mini_graph(tmp_path)

    def broken_provider(problem_statement, graph_dir_arg):
        raise RuntimeError("chroma missing")

    signals = _mod.assemble_file_signals(
        _mini_row(),
        graph_dir,
        subparts=3,
        src_chars=120,
        sim_provider=broken_provider,
    )
    for file_signal in signals:
        assert "sim_max" not in file_signal
        assert "sim_mean" not in file_signal


def test_assemble_file_signals_sorted_by_freq_desc(tmp_path):
    graph_dir = _write_mini_graph(tmp_path)
    signals = _mod.assemble_file_signals(
        _mini_row(), graph_dir, subparts=3, src_chars=120, sim_provider=None
    )
    freqs = [fs["touch_freq"] for fs in signals]
    assert freqs == sorted(freqs, reverse=True)


# --- build_list_prompt ---
def _assembled(tmp_path, **kwargs):
    graph_dir = _write_mini_graph(tmp_path)
    params = {"subparts": 3, "src_chars": 120, "sim_provider": None}
    params.update(kwargs)
    return _mod.assemble_file_signals(_mini_row(), graph_dir, **params)


def test_build_list_prompt_contains_signals_and_parts(tmp_path):
    signals = _assembled(tmp_path)
    prompt = _mod.build_list_prompt(
        "auth is broken", signals, ["freq", "sim_max", "sim_mean", "parts"], top_k=5
    )
    assert "auth is broken" in prompt
    assert "### foo.py" in prompt and "### bar.py" in prompt
    assert "touch_freq: 2" in prompt
    assert "Class Auth" in prompt
    assert "top 5" in prompt
    assert "JSON array" in prompt


def test_build_list_prompt_omits_disabled_signals(tmp_path):
    signals = _assembled(tmp_path)
    prompt = _mod.build_list_prompt("issue", signals, ["freq"], top_k=5)
    assert "touch_freq:" in prompt
    assert "sim_max" not in prompt
    assert "sim_top3_mean" not in prompt
    assert "Class Auth" not in prompt  # parts disabled


def test_build_list_prompt_missing_sims_field_skipped(tmp_path):
    # sim enabled but provider returned no sims -> header must not show them
    signals = _assembled(tmp_path, sim_provider=lambda ps, gd: {})
    prompt = _mod.build_list_prompt(
        "issue", signals, ["freq", "sim_max", "sim_mean", "parts"], top_k=5
    )
    assert "touch_freq:" in prompt
    assert "sim_max:" not in prompt


# --- parse_list_answer ---
def test_parse_list_answer_plain_json():
    answer = _mod.parse_list_answer('["foo.py", "bar.py"]')
    assert answer == ["foo.py", "bar.py"]


def test_parse_list_answer_strips_markdown_fence():
    answer = _mod.parse_list_answer('```json\n["foo.py"]\n```')
    assert answer == ["foo.py"]


def test_parse_list_answer_extracts_array_from_prose():
    answer = _mod.parse_list_answer('Here you go:\n["foo.py"]\nhope it helps')
    assert answer == ["foo.py"]


def test_parse_list_answer_garbage_returns_none():
    assert _mod.parse_list_answer("no array here") is None
    assert _mod.parse_list_answer("") is None
    assert _mod.parse_list_answer('["foo.py", 42]') is None


# --- rank_files_list ---
def _list_engine(client):
    return _mod.ListRanker(client=client, model="m", retries=3)


def test_rank_files_list_single_call_parses_order(tmp_path):
    signals = _assembled(tmp_path)
    client = FakeClient(['["bar.py", "foo.py"]'])
    result = _mod.rank_files_list(
        "issue",
        [("foo.py", "x"), ("bar.py", "y")],
        signals,
        top_k=5,
        engine=_list_engine(client),
    )
    assert result["ranked"] == ["bar.py", "foo.py"]
    assert result["failed"] is False
    assert result["n_calls"] == 1


def test_rank_files_list_filters_unknown_and_pads_with_freq_tail(tmp_path):
    signals = _assembled(tmp_path)
    client = FakeClient(['["mystery.py", "bar.py"]'])
    result = _mod.rank_files_list(
        "issue",
        [("foo.py", "x"), ("bar.py", "y")],
        signals,
        top_k=5,
        engine=_list_engine(client),
    )
    # mystery.py dropped; foo.py appended as freq tail -> full ranking kept
    assert result["ranked"] == ["bar.py", "foo.py"]
    assert result["failed"] is False


def test_rank_files_list_retry_ladder_then_success(tmp_path):
    signals = _assembled(tmp_path)
    client = FakeClient(["garbage", '["foo.py", "bar.py"]'])
    result = _mod.rank_files_list(
        "issue",
        [("foo.py", "x"), ("bar.py", "y")],
        signals,
        top_k=5,
        engine=_list_engine(client),
    )
    assert result["failed"] is False
    assert result["n_calls"] == 2
    # second attempt disables thinking (same ladder as pairwise comparator)
    assert client.calls[1]["extra_body"] == _mod._NO_THINK_BODY


def test_rank_files_list_gives_up_and_falls_back_to_freq(tmp_path):
    signals = _assembled(tmp_path)  # foo.py freq 2, bar.py freq 1
    client = FakeClient(["garbage", "worse", "nope"])
    result = _mod.rank_files_list(
        "issue",
        [("foo.py", "x"), ("bar.py", "y")],
        signals,
        top_k=5,
        engine=_list_engine(client),
    )
    assert result["failed"] is True
    assert result["ranked"] == ["foo.py", "bar.py"]
    assert result["n_calls"] == 3


def test_rank_files_list_single_file_no_llm_call(tmp_path):
    signals = _assembled(tmp_path)
    client = FakeClient([])
    result = _mod.rank_files_list(
        "issue", [("foo.py", "x")], signals, top_k=5, engine=_list_engine(client)
    )
    assert result["ranked"] == ["foo.py"]
    assert result["failed"] is False
    assert result["n_calls"] == 0


# --- rank_files_list chunking (prompt exceeds rerank context) ---
def _chunk_cap_for_single_file_chunks(issue, prompt_signals, signals):
    """Cap under which the greedy chunker emits one file per chunk.

    The chunker closes a chunk when adding the next block would exceed
    the cap, so capping at template overhead + 2*smallest block + 3
    guarantees no pair fits while singles are always kept (greedy
    keeps at least one file per chunk).
    """
    empty = len(_mod.build_list_prompt(issue, [], signals, top_k=5))
    blocks = [len(_mod._render_file_block(fs, signals)) for fs in prompt_signals]
    return empty + 2 * min(blocks) + 3


def test_rank_files_list_splits_oversized_prompt_into_chunks(tmp_path):
    """Files that would overflow the rerank context are ranked chunk by
    chunk (freq order preserved); winners merge round-robin, the freq
    tail pads the rest."""
    signals = _assembled(tmp_path)  # foo.py freq 2, bar.py freq 1
    prompt_signals = [fs for fs in signals] + [{"path": "zed.py", "touch_freq": 0}]
    issue = "issue"
    cap = _chunk_cap_for_single_file_chunks(issue, prompt_signals, ["freq", "parts"])
    client = FakeClient(['["foo.py"]', '["bar.py"]', '["zed.py"]'])
    result = _mod.rank_files_list(
        issue,
        [("foo.py", "x"), ("bar.py", "y"), ("zed.py", "z")],
        prompt_signals,
        top_k=5,
        engine=_list_engine(client),
        signals=["freq", "parts"],
        max_prompt_chars=cap,
    )
    # one chunk per file: three separate LLM calls, each with one file
    assert result["n_calls"] == 3
    for call in client.calls:
        assert call["messages"][0]["content"].count("\n### ") == 1
    # round-robin over chunk winners, no leftovers to pad
    assert result["ranked"] == ["foo.py", "bar.py", "zed.py"]
    assert result["failed"] is False


def test_rank_files_list_chunk_failure_keeps_other_chunks(tmp_path):
    """A failed chunk only demotes its own files to the freq tail;
    successful chunks still contribute their winners."""
    signals = _assembled(tmp_path)  # foo.py freq 2, bar.py freq 1
    prompt_signals = signals + [{"path": "zed.py", "touch_freq": 0}]
    issue = "issue"
    cap = _chunk_cap_for_single_file_chunks(issue, prompt_signals, ["freq", "parts"])
    client = FakeClient(["garbage", '["bar.py"]', '["zed.py"]'])
    engine = _mod.ListRanker(client=client, model="m", retries=1)
    result = _mod.rank_files_list(
        issue,
        [("foo.py", "x"), ("bar.py", "y"), ("zed.py", "z")],
        prompt_signals,
        top_k=5,
        engine=engine,
        signals=["freq", "parts"],
        max_prompt_chars=cap,
    )
    assert result["failed"] is False
    assert result["n_calls"] == 3
    # foo.py's chunk failed -> freq tail; bar/zed keep their chunk wins
    assert result["ranked"] == ["bar.py", "zed.py", "foo.py"]


def test_rank_files_list_all_chunks_fail_falls_back_to_freq(tmp_path):
    signals = _assembled(tmp_path)
    prompt_signals = signals + [{"path": "zed.py", "touch_freq": 0}]
    issue = "issue"
    cap = _chunk_cap_for_single_file_chunks(issue, prompt_signals, ["freq", "parts"])
    client = FakeClient(["garbage", "garbage", "garbage"])
    engine = _mod.ListRanker(client=client, model="m", retries=1)
    result = _mod.rank_files_list(
        issue,
        [("foo.py", "x"), ("bar.py", "y"), ("zed.py", "z")],
        prompt_signals,
        top_k=5,
        engine=engine,
        signals=["freq", "parts"],
        max_prompt_chars=cap,
    )
    assert result["failed"] is True
    assert result["ranked"] == ["foo.py", "bar.py", "zed.py"]
    assert result["n_calls"] == 3
