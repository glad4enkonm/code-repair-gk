#!/usr/bin/env python3
"""
Repair inference: SEARCH/REPLACE or node-level repair.

Replaces standard diff-based inference with structured edit formats
that produce clean diffs via difflib. Two modes:

- ``sr``:   pure file-level SEARCH/REPLACE (Agentless-style)
- ``node``: graph-guided node replacement + file-level S&R fallback

Usage::

    python src/entrypoints/run_repair_inference.py         \\
        --dataset_name_or_path outputs/data__...__node_source \\
        --split dev                                          \\
        --mode node                                          \\
        --model_name_or_path qwen3-coder-next-Q8_0             \\
        --output_dir outputs                                 \\
        --max_tokens 8192 --temperature 0.0
"""

import argparse
import json
import os
import re
import sys
import traceback
from pathlib import Path

# Add src/ to path for direct script execution
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from entrypoints.swebench_override import apply_swebench_overrides

apply_swebench_overrides()

import openai
from datasets import load_from_disk
from tqdm.auto import tqdm

from kg.build_graphs import _checkout, _graph_key, ensure_repo_cloned
from kg.config import KGConfig
from kg.repair import (
    build_node_index,
    build_node_prompt,
    build_sr_prompt,
    load_graph_data,
    parse_repair_output,
    parse_search_replace_blocks,
    synthesize_node_patch,
    synthesize_sr_patch,
    truncate_file_sections,
)

_ISSUE_RE = re.compile(r"<issue>\n(.*?)\n</issue>", re.DOTALL)
_CODE_RE = re.compile(r"<code>\n(.*?)\n</code>", re.DOTALL)


def extract_issue_and_code(text: str) -> tuple:
    """Extract ``<issue>`` and ``<code>`` content from a style-3 prompt."""
    issue_match = _ISSUE_RE.search(text)
    code_match = _CODE_RE.search(text)
    issue = issue_match.group(1) if issue_match else ""
    code = code_match.group(1) if code_match else ""
    return issue, code


def call_llm(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool = False,
) -> str:
    """Call LLM via OpenAI-compatible API.

    Splits the prompt on the first newline into system/user messages,
    matching the swebench ``call_chat`` convention. ``disable_thinking``
    attaches the llama.cpp chat-template kwarg for reasoning models
    (verified on gemma-4-26B-A4B-it-Q8_0): reasoning otherwise spends the
    completion budget before the SEARCH/REPLACE block.
    """
    system_msg = prompt.split("\n", 1)[0]
    user_msg = prompt.split("\n", 1)[1] if "\n" in prompt else ""
    create_kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        create_kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
    response = openai.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content


def generate_with_thinking_fallback(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    disable_thinking: bool = False,
) -> tuple[str, bool]:
    """Generate, retrying once without thinking when content is empty.

    llama.cpp serves gemma-style reasoning in ``reasoning_content``; a
    thinking pass that hits the token cap returns empty ``content``
    (finish_reason=length), blanking the instance. The retry bounds that
    failure to one extra call; only divergent instances pay it.
    """
    response = call_llm(
        model, prompt, max_tokens, temperature, disable_thinking=disable_thinking
    )
    if response.strip() or disable_thinking:
        return response, False
    print(
        "  empty content (thinking truncated?) — retrying with thinking" " disabled",
        flush=True,
    )
    response = call_llm(model, prompt, max_tokens, temperature, disable_thinking=True)
    return response, True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        required=True,
        help="HuggingFace dataset path (local)",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sr", "node", "tool"],
        default="sr",
        help="Repair mode: sr, node, or tool (agent-based)",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="qwen3-coder-next-Q8_0",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Attach chat_template_kwargs.enable_thinking=False for "
        "reasoning models (gemma); reasoning otherwise eats the "
        "completion budget before the SEARCH/REPLACE block",
    )
    parser.add_argument(
        "--max_context_files",
        type=int,
        default=None,
        help="Drop file sections beyond the first N (guard for rerank "
        "fallbacks that kept the whole pool; first sections are the "
        "top-ranked files)",
    )
    return parser


def main():
    args = build_parser().parse_args()
    config = KGConfig.from_file("config/kg_config.json")

    # Load dataset
    dataset = load_from_disk(args.dataset_name_or_path)[args.split]
    print(
        f"Dataset: {args.dataset_name_or_path}/{args.split} "
        f"— {len(dataset)} instances"
    )
    print(f"Mode: {args.mode}")

    # Output file: {model}__{dataset_basename}__{mode}__{split}.jsonl
    dataset_name = os.path.basename(args.dataset_name_or_path.rstrip("/"))
    output_file = (
        Path(args.output_dir) / f"{args.model_name_or_path}__{dataset_name}"
        f"__{args.mode}__{args.split}.jsonl"
    )
    print(f"Output: {output_file}")

    # Resume support
    existing_ids: set = set()
    results: list = []
    if output_file.exists():
        with open(output_file) as f:
            for line_no, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    results.append(entry)
                    existing_ids.add(entry["instance_id"])
                except json.JSONDecodeError as exc:
                    print(
                        f"WARNING: skipping malformed line {line_no} in "
                        f"{output_file}: {exc}",
                        flush=True,
                    )
    if existing_ids:
        print(f"Resuming: {len(existing_ids)} already processed")

    # Sort by text length (shortest first) for efficiency
    indexed = sorted(range(len(dataset)), key=lambda i: len(dataset[i]["text"]))

    for idx in tqdm(indexed, desc=f"Repair inference ({args.mode})"):
        instance = dataset[idx]
        instance_id = instance["instance_id"]

        if instance_id in existing_ids:
            continue

        repo = instance["repo"]
        base_commit = instance["base_commit"]
        text = instance["text"]

        print(f"\n[{instance_id}] {repo}", flush=True)

        try:
            # Extract issue and code from existing prompt
            issue, code = extract_issue_and_code(text)
            truncated_code = truncate_file_sections(
                code, max_files=args.max_context_files
            )
            if len(truncated_code) < len(code):
                print(
                    f"  context truncated: {len(code)} -> "
                    f"{len(truncated_code)} chars",
                    flush=True,
                )
            code = truncated_code

            if args.mode == "tool":
                # Tool mode: multi-turn agent loop
                from kg.repair_agent import repair_agent

                repo_dir = ensure_repo_cloned(repo, config)
                model_patch = repair_agent(
                    repo_dir=str(repo_dir),
                    base_commit=base_commit,
                    context_text=code,
                    problem_statement=issue,
                    config=config,
                )
                response = "(tool-based repair — see git diff)"
                thinking_retry = False

            else:
                # Single-shot modes: sr or node
                if args.mode == "sr":
                    prompt = build_sr_prompt(issue, code)
                else:
                    prompt = build_node_prompt(issue, code)

                response, thinking_retry = generate_with_thinking_fallback(
                    args.model_name_or_path,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                    disable_thinking=args.disable_thinking,
                )

                repo_dir = ensure_repo_cloned(repo, config)
                _checkout(repo_dir, base_commit)

                if args.mode == "sr":
                    sr_edits = parse_search_replace_blocks(response)
                    model_patch = synthesize_sr_patch(str(repo_dir), sr_edits)
                else:
                    graph_dir = os.path.join(
                        config.graph_cache_dir,
                        _graph_key(repo, base_commit),
                    )
                    if not os.path.isdir(graph_dir):
                        print(f"  WARNING: graph not found at {graph_dir}")
                        model_patch = ""
                    else:
                        graph_data = load_graph_data(graph_dir)
                        node_index = build_node_index(graph_data)
                        node_edits, sr_edits = parse_repair_output(response)
                        model_patch = synthesize_node_patch(
                            str(repo_dir),
                            node_index,
                            node_edits,
                            sr_edits,
                        )

            entry = {
                "instance_id": instance_id,
                "model_name_or_path": args.model_name_or_path,
                "text": text,
                "full_output": response,
                "model_patch": model_patch,
                "thinking_retry": thinking_retry,
            }
            results.append(entry)
            existing_ids.add(instance_id)

            # Write incrementally
            with open(output_file, "w") as f:
                for r in results:
                    print(json.dumps(r, ensure_ascii=False), file=f)

            print(f"  patch: {len(model_patch)} chars", flush=True)

        except Exception:
            print(f"  FAILED: {instance_id}", flush=True)
            traceback.print_exc()

    print(f"\nDone. Output: {output_file}")


if __name__ == "__main__":
    main()
