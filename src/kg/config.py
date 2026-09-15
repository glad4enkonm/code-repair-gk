import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class KGConfig:
    # LLM
    api_base: str = "http://localhost:8081/v1"
    model: str = "qwen3-coder-next-Q8_0"
    api_key: str = "any"
    temperature: float = 0.0
    max_tokens: int = 1024
    # Exploration budget
    max_rounds: int = 20
    max_tool_calls: int = 50
    # Anomaly escalation (explore loop): each consecutive rejected round
    # (tool calls failing extraction/validation) raises the request
    # temperature by escalation_temp_step, capped at escalation_temp_max;
    # after rejection_restart_after consecutive rejections the
    # conversation resets to the initial prompt (once) and grants
    # restart_round_grant extra rounds
    escalation_temp_step: float = 0.15
    escalation_temp_max: float = 0.9
    rejection_restart_after: int = 3
    restart_round_grant: int = 6
    # Graph build
    meta_schema_dir: str = "swe_bench_kg"
    graph_cache_dir: str = "outputs/kg_graphs"
    repo_cache_dir: str = "outputs/kg_repos"
    # Context
    max_context_tokens: int = 128_000
    context_mode: str = "file_level"
    node_context_padding: int = 5
    # Dataset
    dataset_path: str = "data/swe-bench_lite"
    output_dir: str = "outputs"
    prompt_style: str = "style-3"
    split: str = "test"
    # Per-request thinking disable for reasoning models (llama.cpp chat
    # template kwarg; verified on gemma-4-26B-A4B-it-Q8_0): reasoning
    # otherwise spends the completion budget before the tool-call JSON.
    # Default off keeps non-reasoning models (qwen) request bodies intact.
    disable_thinking: bool = False
    # Inference
    inference_temp: float = 0.0
    inference_max_tokens: int = 8192
    # Repair agent
    repair_max_rounds: int = 20
    repair_max_tool_calls: int = 50
    # Mirror
    mirror_url: str = "https://github.com/swe-bench-repos/{repo}.git"
    # Embedding-augmented exploration (opt-in; needs llama-server with
    # an embedding model serving /v1/embeddings on embedding_api_base)
    use_embeddings: bool = False
    # Quantization-pinned: also the Chroma cache key — change it when
    # switching GGUF quants so the index rebuilds instead of reusing
    # vectors from the previous quant
    embedding_model: str = "nomic-embed-code-Q8_0"
    embedding_api_base: str = "http://localhost:8082/v1"
    embedding_dim: int = 3584
    embedding_batch_size: int = 32
    embedding_max_source_chars: int = 2048
    # Hard cap on any single composed document/query text (chars) —
    # guards against unbounded docstrings and Variable values
    embedding_max_doc_chars: int = 8192
    # Estimated-token budget per embeddings request; must stay below the
    # embedding server's context (nomic-embed-code: 32768)
    embedding_ctx_tokens: int = 30000
    embedding_top_k: int = 15
    embedding_search_k: int = 10
    embedding_skip_node_types: Optional[list] = None
    # Shared cross-graph vector cache (git blob-SHA keyed, see
    # kg.embeddings). Points at a directory; the Chroma store lives in
    # <dir>/chroma_db. None disables the cache entirely.
    embedding_cache_dir: Optional[str] = None

    @classmethod
    def from_file(cls, path: str) -> "KGConfig":
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @property
    def meta_path(self) -> str:
        return os.path.join(self.meta_schema_dir, "meta.json")

    @property
    def origin_path(self) -> str:
        return os.path.join(self.meta_schema_dir, "origin.json")
