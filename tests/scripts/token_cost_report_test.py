"""Tests for token-cost-report.py (KG vs BM25 vs Agentless token costs)."""

import importlib.util
import json
from pathlib import Path

_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "token-cost-report.py"
_spec = importlib.util.spec_from_file_location("tcr", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class _StubTokenizer:
    """Deterministic fake: one token per character."""

    class _Enc:
        def __init__(self, ids):
            self.ids = ids

    def encode(self, text):
        return self._Enc(list(range(len(text))))


# --- counters ---


def test_heuristic_counter_counts_quarters():
    counter = _mod.CharsHeuristicCounter()
    assert counter.count("abcd" * 3) == 3
    assert counter.count("") == 0
    assert counter.count("ab") == 0
    assert counter.describe().startswith("chars/4")


def test_token_counter_uses_encode_ids():
    counter = _mod.TokenCounter(_StubTokenizer())
    assert counter.count("hello") == 5
    assert counter.count("") == 0


def test_make_counter_falls_back_without_tokenizer():
    counter = _mod.make_counter(None)
    assert isinstance(counter, _mod.CharsHeuristicCounter)


def test_make_counter_uses_tokenizer_object():
    counter = _mod.make_counter(tokenizer=_StubTokenizer())
    assert counter.count("abc") == 3


# --- parsing ---


def test_pool_files_extracts_ordered_paths():
    text = (
        "header\n[start of lib/foo.py]\ncontent\n[end of lib/foo.py]\n"
        "middle\n[start of lib/bar.py]\nmore\n[end of lib/bar.py]\n"
    )
    assert _mod.pool_files(text) == ["lib/foo.py", "lib/bar.py"]


def test_gt_files_dedupes_diff_headers():
    patch = (
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
        "--- a/y.py\n+++ b/y.py\n@@ -2 +2 @@\n+++ b/x.py\n"
    )
    assert _mod.gt_files(patch) == ["x.py", "y.py"]


# --- loaders ---


def test_load_kg_rows_skips_first_n(tmp_path):
    path = tmp_path / "kg.progress.jsonl"
    rows = [
        {
            "instance_id": f"i{n}",
            "repo": "r",
            "problem_statement": "p",
            "patch": "+++ b/a.py\n",
            "text_inputs": "[start of a.py]\nx\n[end of a.py]\n",
        }
        for n in range(3)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    loaded = _mod.load_kg_rows(str(path), skip=1)
    assert [r["instance_id"] for r in loaded] == ["i1", "i2"]
    assert loaded[0]["text_inputs"].startswith("[start of")


def test_load_bm25_texts_from_jsonl(tmp_path):
    path = tmp_path / "bm25.jsonl"
    path.write_text(
        json.dumps({"instance_id": "i1", "text": "t1"})
        + "\n"
        + json.dumps({"instance_id": "i2", "text": "t2"})
        + "\n"
    )
    assert _mod.load_bm25_texts(jsonl_path=str(path)) == {"i1": "t1", "i2": "t2"}


def test_load_agentless_found(tmp_path):
    path = tmp_path / "loc_outputs.jsonl"
    path.write_text(
        json.dumps({"instance_id": "i1", "found_files": ["a.py", "b.py"]})
        + "\n"
        + json.dumps({"instance_id": "i2", "found_files": []})
        + "\n"
    )
    found = _mod.load_agentless_found(str(path))
    assert found["i1"] == ["a.py", "b.py"]
    assert found["i2"] == []


def test_load_corpus_reads_documents_jsonl(tmp_path):
    index_dir = tmp_path / "idx" / "i1"
    index_dir.mkdir(parents=True)
    (index_dir / "documents.jsonl").write_text(
        json.dumps({"id": "a.py", "contents": "print(1)\n"})
        + "\n"
        + json.dumps({"id": "b.py", "contents": "x = 2\n"})
        + "\n"
    )
    corpus = _mod.load_corpus(str(tmp_path / "idx"), "i1")
    assert corpus == {"a.py": "print(1)\n", "b.py": "x = 2\n"}


# --- metrics ---


def test_token_stats_mean_median_p90():
    stats = _mod.token_stats(list(range(1, 11)))
    assert stats["mean"] == 5.5
    assert stats["median"] == 5.5
    assert stats["p90"] == 9  # nearest-rank: sorted[ceil(0.9*10)-1]


def test_token_stats_empty_returns_none():
    stats = _mod.token_stats([])
    assert stats == {"mean": None, "median": None, "p90": None}


def test_evaluate_method_computes_acc_and_tokens_per_hit():
    instances = [
        {"instance_id": "i1", "tokens": 100, "files": 4, "hit": True},
        {"instance_id": "i2", "tokens": 200, "files": 6, "hit": False},
        {"instance_id": "i3", "tokens": 300, "files": 2, "hit": True},
    ]
    summary = _mod.evaluate_method(instances)
    assert summary["n"] == 3
    assert summary["acc"] == 2 / 3
    assert summary["avg_files"] == 4.0
    assert summary["mean_tokens"] == 200.0
    assert summary["tokens_per_hit"] == 600 / 2


def test_evaluate_method_without_hits_column():
    instances = [{"instance_id": "i1", "tokens": 10, "files": 1}]
    summary = _mod.evaluate_method(instances)
    assert summary["acc"] is None
    assert summary["tokens_per_hit"] is None


# --- agentless estimation ---


def test_agentless_context_tokens_sums_and_reports_missing():
    counter = _mod.CharsHeuristicCounter()
    tokens, missing = _mod.agentless_context_tokens(
        found_files=["a.py", "gone.py"],
        problem_statement="abcd",
        corpus={"a.py": "x" * 40},
        counter=counter,
    )
    # 40 chars -> 10 tokens, problem statement 4 chars -> 1 token
    assert tokens == 11
    assert missing == ["gone.py"]


# --- rerank estimation (pure, injected callables) ---


def _rerank_case(n_calls, chunk_count):
    signals = [{"path": f"f{n}.py", "touch_freq": n} for n in range(4)]
    prompts = [f"prompt {i}" for i in range(chunk_count)]
    chunks = [signals[i::2] for i in range(2)][:chunk_count]

    def build_prompt(issue, chunk, signal_names, top_k):
        return prompts[chunks.index(chunk) if chunk in chunks else 0]

    def chunk_prompts(issue, prompt_signals, signal_names, top_k, cap):
        return chunks

    return _mod.estimate_rerank_tokens(
        issue="issue",
        prompt_signals=signals,
        signals=["freq"],
        top_k=5,
        max_prompt_chars=1000,
        n_calls=n_calls,
        counter=_mod.TokenCounter(_StubTokenizer()),
        build_prompt=build_prompt,
        chunk_prompts=chunk_prompts,
    )


def test_estimate_rerank_tokens_single_chunk_no_retries():
    result = _rerank_case(n_calls=1, chunk_count=1)
    # prompt "prompt 0" = 8 chars = 8 stub tokens, output ~json answer estimate
    assert result["input_tokens"] == 8
    assert result["n_chunks"] == 1
    assert result["retry_multiplier"] == 1.0
    assert result["output_tokens_estimate"] > 0


def test_estimate_rerank_tokens_retry_inflation():
    result = _rerank_case(n_calls=3, chunk_count=1)
    assert result["input_tokens"] == 8 * 3
    assert result["retry_multiplier"] == 3.0


def test_estimate_rerank_tokens_chunking_sums_prompts():
    result = _rerank_case(n_calls=2, chunk_count=2)
    assert result["n_chunks"] == 2
    # "prompt 0" + "prompt 1" = 16 tokens, one call per chunk
    assert result["input_tokens"] == 16


# --- report assembly ---


def test_agentless_paper_numbers_are_cited():
    paper = _mod.AGENTLESS_PAPER_NUMBERS
    assert paper["avg_tokens_full_pipeline"] == 78166
    assert paper["file_level_stage_cost_usd"] == 0.06
    assert "arxiv.org/abs/2407.01489" in paper["source"]


# --- KG solver-prompt method (exact sr predictions) ---


def test_load_sr_predictions_reads_prompt_and_output():
    path = _tmp_sr_file(
        {
            "instance_id": "i1",
            "text": "solve this",
            "full_output": "the patch",
            "model_patch": "diff --git",
        }
    )
    loaded = _mod.load_sr_predictions(str(path))
    assert loaded["i1"]["prompt"] == "solve this"
    assert loaded["i1"]["output"] == "the patch"


def _tmp_sr_file(*rows, tmp_path=None):
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    path = tmp_path / "sr.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_build_report_kg_sr_uses_exact_prompts_with_pool_fallback(tmp_path):
    counter = _mod.CharsHeuristicCounter()
    kg_instances = [
        {
            "instance_id": "i1",
            "repo": "r",
            "problem_statement": "ps ps ps ps",
            "patch": "+++ b/a.py\n",
            "text_inputs": "[start of a.py]\n" + "x" * 40 + "\n[end of a.py]\n",
        },
        {
            "instance_id": "i2",
            "repo": "r",
            "problem_statement": "ps ps ps ps",
            "patch": "+++ b/c.py\n",
            "text_inputs": "[start of c.py]\n" + "x" * 80 + "\n[end of c.py]\n",
        },
    ]
    sr_prompts = {
        "i1": {
            "prompt": "[start of a.py]\n" + "y" * 20 + "\n[end of a.py]\n",
            "output": "patch text",
        },
    }
    report = _mod.build_report(
        kg_instances=kg_instances,
        bm25_texts={},
        agentless_found={},
        corpus_dir=str(tmp_path),
        counter=counter,
        sr_prompts=sr_prompts,
    )
    kg_sr = report["methods"]["kg_sr"]
    assert kg_sr["n"] == 2
    assert kg_sr["n_fallback"] == 1
    # i1: 20+~30 chars block text; i2 fallback: 80+~30 chars pool text
    tokens = {r["instance_id"]: r["tokens"] for r in report["instances"]["kg_sr"]}
    assert tokens["i1"] == counter.count(sr_prompts["i1"]["prompt"])
    assert tokens["i2"] == counter.count(kg_instances[1]["text_inputs"])
    assert report["instances"]["kg_sr"][1]["from_pool_fallback"] is True
    assert kg_sr["acc"] == 1.0
    assert kg_sr["output_mean_tokens"] == 2  # "patch text": 10 chars / 4
    assert kg_sr["output_n"] == 1


def test_build_report_without_sr_prompts_keeps_pool_only(tmp_path):
    counter = _mod.CharsHeuristicCounter()
    kg_instances = [
        {
            "instance_id": "i1",
            "repo": "r",
            "problem_statement": "ps",
            "patch": "+++ b/a.py\n",
            "text_inputs": "[start of a.py]\nx\n[end of a.py]\n",
        },
    ]
    report = _mod.build_report(
        kg_instances=kg_instances,
        bm25_texts={},
        agentless_found={},
        corpus_dir=str(tmp_path),
        counter=counter,
    )
    assert "kg_sr" not in report["methods"]


def test_render_markdown_includes_kg_sr_row_and_output_tokens():
    report = {
        "counter": "stub",
        "instance_count": 2,
        "methods": {
            "kg_pool": {
                "n": 2,
                "avg_files": 10.0,
                "acc": 0.5,
                "mean_tokens": 1000,
                "median_tokens": 900,
                "p90_tokens": 1200,
                "tokens_per_hit": 2000,
            },
            "kg_sr": {
                "n": 2,
                "avg_files": 5.0,
                "acc": 0.5,
                "mean_tokens": 500,
                "median_tokens": 450,
                "p90_tokens": 600,
                "tokens_per_hit": 1000,
                "n_fallback": 1,
                "output_mean_tokens": 747.0,
                "output_median_tokens": 700,
                "output_p90_tokens": 900,
                "output_n": 1,
            },
            "bm25_k5": {
                "n": 2,
                "avg_files": 5.0,
                "acc": 0.4,
                "mean_tokens": 800,
                "median_tokens": 700,
                "p90_tokens": 1000,
                "tokens_per_hit": 2000,
            },
            "agentless": {
                "n": 2,
                "avg_files": 3.0,
                "acc": 0.4,
                "mean_tokens": 900,
                "median_tokens": 800,
                "p90_tokens": 1100,
                "tokens_per_hit": 2250,
            },
        },
    }
    md = _mod.render_markdown(report)
    assert "KG sr (rerank top-5)" in md
    assert "output tokens (exact" in md
    assert "fallback: 1" in md


def test_build_report_renders_all_methods(tmp_path):
    counter = _mod.CharsHeuristicCounter()
    kg_instances = [
        {
            "instance_id": "i1",
            "repo": "r",
            "problem_statement": "ps ps ps ps",
            "patch": "+++ b/a.py\n",
            "text_inputs": "[start of a.py]\n" + "x" * 40 + "\n[end of a.py]\n",
        },
    ]
    corpus_dir = tmp_path / "idx"
    for iid in ("i1",):
        d = corpus_dir / iid
        d.mkdir(parents=True)
        (d / "documents.jsonl").write_text(
            json.dumps({"id": "a.py", "contents": "y" * 80}) + "\n"
        )
    report = _mod.build_report(
        kg_instances=kg_instances,
        bm25_texts={"i1": "[start of a.py]\n" + "z" * 120 + "\n[end of a.py]\n"},
        agentless_found={"i1": ["a.py"]},
        corpus_dir=str(corpus_dir),
        counter=counter,
    )
    assert set(report["methods"]) == {"kg_pool", "bm25_k5", "agentless"}
    kg = report["methods"]["kg_pool"]
    assert kg["n"] == 1 and kg["acc"] == 1.0 and kg["avg_files"] == 1.0
    md = _mod.render_markdown(report)
    for name in ("KG pool", "BM25 k=5", "Agentless"):
        assert name in md
    assert "78,166" in md  # paper-cited number rendered


# --- per-stage token accounting ---


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_load_agentless_stage_usage_reads_traj_usage(tmp_path):
    path = tmp_path / "loc_outputs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "instance_id": "i1",
                "file_traj": {
                    "prompt": "p",
                    "response": "r",
                    "usage": {"prompt_tokens": 2427, "completion_tokens": 122},
                },
            },
            {"instance_id": "i2", "file_traj": None},
        ],
    )
    usages = _mod.load_agentless_stage_usage(str(path), "file_traj")
    assert usages["i1"] == {
        "prompt_tokens": 2427,
        "completion_tokens": 122,
        "calls": 1,
    }
    assert usages["i2"] == {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def test_load_agentless_stage_usage_sums_list_traj_calls(tmp_path):
    # related_elements/loc_outputs.jsonl rows carry related_loc_traj as a
    # LIST of sequential calls (Agentless runs two prompts there).
    path = tmp_path / "loc_outputs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "instance_id": "i1",
                "related_loc_traj": [
                    {"usage": {"prompt_tokens": 5000, "completion_tokens": 300}},
                    {"usage": {"prompt_tokens": 7000, "completion_tokens": 400}},
                    {"usage": None},
                ],
            },
            {"instance_id": "i2", "related_loc_traj": []},
        ],
    )
    usages = _mod.load_agentless_stage_usage(str(path), "related_loc_traj")
    assert usages["i1"] == {
        "prompt_tokens": 12000,
        "completion_tokens": 700,
        "calls": 3,
    }
    assert usages["i2"] == {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}


def test_load_agentless_repair_usage_sums_traj_calls(tmp_path):
    path = tmp_path / "output.jsonl"
    calls = [
        {"usage": {"prompt_tokens": 1704, "completion_tokens": 718}},
        {"usage": {"prompt_tokens": 1000, "completion_tokens": 200}},
        {"usage": None},
    ]
    _write_jsonl(
        path,
        [
            {"instance_id": "i1", "traj": calls},
            {"instance_id": "i2", "traj": []},
            {"instance_id": "i3"},
        ],
    )
    usages = _mod.load_agentless_repair_usage(str(path))
    assert usages["i1"] == {
        "prompt_tokens": 2704,
        "completion_tokens": 918,
        "calls": 3,
    }
    assert usages["i2"] == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "calls": 0,
    }
    assert usages["i3"]["calls"] == 0


def test_usage_stage_summary_aggregates_prompt_and_completion():
    usages = {
        "i1": {"prompt_tokens": 100, "completion_tokens": 10, "calls": 1},
        "i2": {"prompt_tokens": 300, "completion_tokens": 30, "calls": 5},
    }
    summary = _mod.usage_stage_summary(usages)
    assert summary["n"] == 2
    assert summary["avg_calls"] == 3.0
    assert summary["prompt_mean_tokens"] == 200
    assert summary["completion_mean_tokens"] == 20
    assert summary["total_mean_tokens"] == 220
    assert summary["total_median_tokens"] == 220


def test_usage_stage_summary_empty():
    assert _mod.usage_stage_summary({}) == {"n": 0}


def test_collect_agentless_stages_reads_all_stage_files(tmp_path):
    base = tmp_path / "ag"
    _write_jsonl(
        base / "file_level" / "loc_outputs.jsonl",
        [
            {
                "instance_id": "i1",
                "file_traj": {"usage": {"prompt_tokens": 10, "completion_tokens": 2}},
            }
        ],
    )
    _write_jsonl(
        base / "related_elements" / "loc_outputs.jsonl",
        [
            {
                "instance_id": "i1",
                "related_loc_traj": {
                    "usage": {"prompt_tokens": 20, "completion_tokens": 3},
                },
            }
        ],
    )
    _write_jsonl(
        base / "edit_location_samples" / "loc_outputs.jsonl",
        [
            {
                "instance_id": "i1",
                "edit_loc_traj": {
                    "usage": {"prompt_tokens": 30, "completion_tokens": 4}
                },
            }
        ],
    )
    _write_jsonl(
        base / "repair" / "output.jsonl",
        [
            {
                "instance_id": "i1",
                "traj": [
                    {"usage": {"prompt_tokens": 40, "completion_tokens": 5}},
                    {"usage": {"prompt_tokens": 50, "completion_tokens": 6}},
                ],
            }
        ],
    )
    stages = _mod.collect_agentless_stages(str(base))
    assert set(stages) == {"file_loc", "related_loc", "edit_loc", "repair"}
    assert stages["repair"]["i1"]["calls"] == 2
    totals = _mod.agentless_per_instance_totals(stages)
    assert totals["i1"] == 10 + 2 + 20 + 3 + 30 + 4 + 90 + 11


def test_agentless_per_instance_totals_skips_missing_stage():
    totals = _mod.agentless_per_instance_totals(
        {"file_loc": {"i1": {"prompt_tokens": 5, "completion_tokens": 1, "calls": 1}}}
    )
    assert totals == {"i1": 6}


def test_solve_stage_from_predictions_counts_input_and_output():
    counter = _mod.CharsHeuristicCounter()
    predictions = {
        "i1": {"prompt": "x" * 40, "output": "y" * 8},
        "i2": {"prompt": "x" * 80, "output": ""},
    }
    stage = _mod.solve_stage_from_predictions(predictions, counter)
    assert stage["n"] == 2
    assert stage["input_mean_tokens"] == 15  # (10 + 20) / 2
    assert stage["output_n"] == 1
    assert stage["output_mean_tokens"] == 2
    assert stage["total_per_instance_mean_tokens"] == (12 + 20) / 2


def test_embed_query_chars_caps_issue_and_adds_prefix():
    chars = _mod.embed_query_chars("q" * 9000, doc_cap=8192, prefix_len=20)
    assert chars == 8192 + 20
    assert _mod.embed_query_chars("q" * 10, doc_cap=8192, prefix_len=20) == 30


def test_embed_cache_stats_counts_vectors_and_repos(tmp_path):
    import sqlite3

    cache = tmp_path / "chroma_db"
    cache.mkdir()
    con = sqlite3.connect(str(cache / "chroma.sqlite3"))
    con.execute(
        "CREATE TABLE embeddings (id INTEGER PRIMARY KEY, segment_id TEXT, embedding_id TEXT, seq_id INTEGER, created_at INTEGER)"
    )
    con.execute(
        "CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, int_value INTEGER, float_value REAL, bool_value INTEGER)"
    )
    for n in range(3):
        con.execute("INSERT INTO embeddings VALUES (?, 's', 'e', 0, 0)", (n + 1,))
        con.execute(
            "INSERT INTO embedding_metadata (id, key, string_value) VALUES (?, 'repo_key', ?)",
            (n + 1, "repoA__c1" if n < 2 else "repoB__c2"),
        )
    con.commit()
    con.close()
    stats = _mod.embed_cache_stats(str(cache))
    assert stats["vectors"] == 3
    assert stats["per_repo"] == {"repoA__c1": 2, "repoB__c2": 1}


def test_embed_cache_stats_missing_db_reports_zero(tmp_path):
    stats = _mod.embed_cache_stats(str(tmp_path / "nope"))
    assert stats["vectors"] == 0


def test_graphs_embed_chars_replays_composed_documents(tmp_path):
    graphs = tmp_path / "graphs" / "repoA__c1"
    graphs.mkdir(parents=True)
    file_node = {"id": "File-src/a.py", "label": "a.py"}
    function_node = {"id": "Function-src/a.py-f", "label": "f"}
    (graphs / "data.json").write_text(
        json.dumps(
            {
                "nodes": [file_node, function_node],
                "edges": [
                    {
                        "label": "CONTAINS",
                        "source": "File-src/a.py",
                        "target": "Function-src/a.py-f",
                    }
                ],
                "allValues": {
                    "File-src/a.py": {
                        "meta_type": "File",
                        "relative_path": "src/a.py",
                    },
                    "Function-src/a.py-f": {
                        "meta_type": "Function",
                        "name": "f",
                        "docstring": "doc",
                        "source_code": "code",
                    },
                },
            }
        )
    )
    result = _mod.graphs_embed_chars(str(tmp_path / "graphs"))
    assert result["graphs"] == 1
    assert result["nodes"] == 2
    # File: "search_code_document: " (22) + "File src/a.py" (13) = 35
    # Function: 22 + "Function f in src/a.py\ndoc\ncode" (31) = 53
    assert result["total_chars"] == 35 + 53


def test_render_markdown_includes_stage_table_and_embed_block():
    report = {
        "counter": "stub",
        "instance_count": 2,
        "methods": {
            "kg_pool": {
                "n": 2,
                "avg_files": 1.0,
                "acc": 0.5,
                "mean_tokens": 1,
                "median_tokens": 1,
                "p90_tokens": 1,
                "tokens_per_hit": 2,
            },
            "bm25_k5": {
                "n": 2,
                "avg_files": 1.0,
                "acc": 0.5,
                "mean_tokens": 1,
                "median_tokens": 1,
                "p90_tokens": 1,
                "tokens_per_hit": 2,
            },
            "agentless": {
                "n": 2,
                "avg_files": 1.0,
                "acc": 0.5,
                "mean_tokens": 1,
                "median_tokens": 1,
                "p90_tokens": 1,
                "tokens_per_hit": 2,
            },
        },
        "stages": {
            "bm25_solve": {
                "n": 2,
                "input_mean_tokens": 100768,
                "output_mean_tokens": 1200,
                "output_n": 2,
                "total_per_instance_mean_tokens": 101968,
            },
            "agentless": {
                "file_loc": {
                    "n": 2,
                    "avg_calls": 1.0,
                    "prompt_mean_tokens": 2427,
                    "completion_mean_tokens": 122,
                    "total_mean_tokens": 2549,
                },
                "repair": {
                    "n": 2,
                    "avg_calls": 5.0,
                    "prompt_mean_tokens": 8520,
                    "completion_mean_tokens": 3590,
                    "total_mean_tokens": 12110,
                },
            },
            "agentless_total": {
                "n": 2,
                "total_mean_tokens": 40000,
                "total_median_tokens": 38000,
                "total_p90_tokens": 45000,
            },
            "kg_solve": {
                "n": 2,
                "input_mean_tokens": 45000,
                "output_mean_tokens": 900,
                "output_n": 2,
                "total_per_instance_mean_tokens": 45900,
            },
        },
        "embed": {
            "model": "nomic-embed-code-Q8_0",
            "vectors": 2287225,
            "per_repo": {"django__django": 900000},
            "node_chars_exact": None,
            "node_chars_upper_bound": 2287225 * 8214,
            "query_chars_estimate": 300 * 8212,
            "query_calls_estimate": 300,
        },
    }
    md = _mod.render_markdown(report)
    assert "Per-stage token accounting" in md
    assert "file localization" in md
    assert "repair" in md
    assert "100,768" in md
    assert "Embedding cost (separate model" in md
    assert "nomic-embed-code-Q8_0" in md
    assert "2,287,225" in md


# --- sr sent-prompt reconstruction (what was actually sent to the model) ---


def _style3_text(issue: str, code: str) -> str:
    return (
        "You will be provided with a partial code base...\n"
        f"<issue>\n{issue}\n</issue>\n\n<code>\n{code}\n</code>\n\n"
        "Here is an example of a patch file...\n<patch>\n--- a/x.py\n</patch>\n"
    )


def test_load_sr_predictions_reconstructs_sent_prompt(tmp_path):
    code = "[start of lib/foo.py]\n1 def foo():\n2     return 1\n[end of lib/foo.py]\n"
    row = {
        "instance_id": "i1",
        "text": _style3_text("crash on foo", code),
        "full_output": "out",
    }
    path = tmp_path / "sr.jsonl"
    path.write_text(json.dumps(row) + "\n")
    got = _mod.load_sr_predictions(str(path), sr_reconstruct=True)["i1"]
    assert "crash on foo" in got["prompt"]
    assert "1 def foo():" not in got["prompt"]
    assert "<<<<<<< SEARCH" in got["prompt"]
    assert "Here is an example of a patch file" not in got["prompt"]
    assert got["prompt_dataset_text"] == row["text"]
    assert got["output"] == "out"


def test_load_sr_predictions_truncates_only_listed_ids(tmp_path):
    big = "[start of lib/big.py]\n" + ("y" * 70_000) + "\n[end of lib/big.py]\n"
    rows = [
        {"instance_id": "trunc1", "text": _style3_text("i", big), "full_output": ""},
        {"instance_id": "keep1", "text": _style3_text("i", big), "full_output": ""},
    ]
    path = tmp_path / "sr.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = _mod.load_sr_predictions(
        str(path), sr_reconstruct=True, truncated_ids={"trunc1"}
    )
    assert "characters truncated" in got["trunc1"]["prompt"]
    assert "characters truncated" not in got["keep1"]["prompt"]
    assert len(got["trunc1"]["prompt"]) < len(got["keep1"]["prompt"])


def test_load_sr_predictions_falls_back_to_raw_text_without_markers(tmp_path):
    row = {"instance_id": "i2", "text": "no markers here", "full_output": ""}
    path = tmp_path / "sr.jsonl"
    path.write_text(json.dumps(row) + "\n")
    got = _mod.load_sr_predictions(str(path), sr_reconstruct=True)["i2"]
    assert got["prompt"] == "no markers here"
    assert got["prompt_dataset_text"] == "no markers here"


def test_load_sr_predictions_default_keeps_raw_text_for_bm25(tmp_path):
    code = "[start of lib/foo.py]\n1 def foo():\n[end of lib/foo.py]\n"
    row = {"instance_id": "b1", "text": _style3_text("i", code), "full_output": ""}
    path = tmp_path / "sr.jsonl"
    path.write_text(json.dumps(row) + "\n")
    got = _mod.load_sr_predictions(str(path))["b1"]
    assert got["prompt"] == row["text"]
    assert "1 def foo():" in got["prompt"]
