import json
import os
import traceback
from typing import Dict, List

from .config import KGConfig
from .prompts import build_fault_loc_system_prompt

_CALL_TEMPLATE = (
    '[\n  {"function": {"name": "search_elements", '
    '"arguments": {"graph": "data", "query": "..."}}}\n]'
)

# Rotated phrasings for rejected rounds — a deterministic model repeats
# itself when told the same thing the same way; varying the framing (and
# the temperature) breaks the loop
_REJECTION_FEEDBACKS = (
    "Your tool calls had {n} issue(s):\n{issues}\n\n"
    "Please respond with tool calls as a JSON array of objects:\n" + _CALL_TEMPLATE,
    "Those calls still failed validation:\n{issues}\n\n"
    "Answer ONLY with a JSON array in this exact shape "
    "(no prose, no code fences):\n" + _CALL_TEMPLATE,
    "Still invalid:\n{issues}\n\n"
    "Copy this template literally and replace only the query:\n"
    '[\n  {"function": {"name": "search_elements", '
    '"arguments": {"graph": "data", "query": "your keywords"}}}\n]',
)

_RESTART_NOTE = (
    "Restarting exploration from scratch — your previous tool calls kept "
    "failing validation. Begin again with ONE simple call:\n" + _CALL_TEMPLATE
)


def _import_graph_ops():
    from tool.functions.graph_file_ops import get_function_definitions
    from tool.functions.graph_file_ops.read import (
        find_connected,
        get_element,
        get_neighbors,
        search_elements,
        traverse,
    )
    from tool.utils.json_tools import extract_function_calls

    from .prompts import _customize_for_code_graph

    def _find_connected(**kwargs):
        kwargs.setdefault("edge_label", "CONTAINS")
        return find_connected(**kwargs)

    # Build code-specific function definitions for get_function_info queries
    read_names = {
        "search_elements",
        "get_neighbors",
        "traverse",
        "get_element",
        "find_connected",
    }
    all_defs = get_function_definitions()
    fn_defs = {}
    for d in all_defs:
        name = d.get("function", {}).get("name", "")
        if name in read_names:
            fn_defs[name] = d
    fn_defs = _customize_for_code_graph(fn_defs)

    def _get_function_info(**kwargs):
        name = kwargs.get("function_name", "")
        if name in fn_defs:
            return {"success": True, "info": fn_defs[name]}
        return {
            "success": False,
            "error": (
                f"Function '{name}' not found. "
                f"Available: {', '.join(sorted(fn_defs))}"
            ),
        }

    return {
        "search_elements": search_elements,
        "get_neighbors": get_neighbors,
        "traverse": traverse,
        "get_element": get_element,
        "find_connected": _find_connected,
        "get_function_info": _get_function_info,
    }, extract_function_calls


def _extract_node_ids(result: dict) -> List[dict]:
    """Extract {id, label, properties?} entries from any graph read result."""
    nodes: List[dict] = []
    if not result.get("success"):
        return nodes
    if "items" in result:  # search_elements — filter to nodes only
        nodes = [item for item in result["items"] if item.get("kind") != "edge"]
    elif "nodes" in result:  # get_neighbors
        nodes = result["nodes"]
    elif "paths" in result:  # traverse
        for path in result["paths"]:
            nodes.extend(path.get("nodes", []))
    elif "kind" in result:  # get_element
        if result["kind"] == "node":
            entry = {"id": result["id"], "label": result["label"]}
            if result.get("properties"):
                entry["properties"] = result["properties"]
            nodes = [entry]
    elif "node" in result:  # find_connected
        nodes = [result["node"]]
    return nodes


def _collect_node(nodes: List[dict], touched: dict):
    for n in nodes:
        nid = n.get("id")
        if not nid:
            continue
        if nid not in touched:
            touched[nid] = {}
        if n.get("label"):
            touched[nid]["label"] = n["label"]
        if n.get("properties"):
            touched[nid].update(n["properties"])


def _format_results(results: List[tuple]) -> str:
    """Format graph read results for the LLM — includes full data, not just counts."""
    blocks = []
    for fn_name, fn_args, result in results:
        success = result.get("success", False)
        if not success:
            blocks.append(f"ERROR ({fn_name}): {result.get('error', 'unknown')}")
            continue

        # Build a compact but informative summary
        parts = [f"SUCCESS ({fn_name})"]
        if "total" in result:
            parts.append(f"{result['total']} items")
            for item in result.get("items", [])[:20]:
                nid = item.get("id", "?")
                label = item.get("label", "")
                parts.append(f"  {nid}" + (f" ({label})" if label else ""))
        elif "nodes" in result:
            parts.append(f"{len(result['nodes'])} neighbors")
            for node in result.get("nodes", [])[:20]:
                nid = node.get("id", "?")
                label = node.get("label", "")
                parts.append(f"  {nid}" + (f" ({label})" if label else ""))
        elif "paths" in result:
            parts.append(f"{len(result['paths'])} paths")
            for i, path in enumerate(result.get("paths", [])[:5]):
                node_ids = [n.get("id", "?") for n in path.get("nodes", [])]
                parts.append(f"  path {i}: {' -> '.join(node_ids)}")
        elif "kind" in result:
            nid = result.get("id", "?")
            label = result.get("label", "")
            parts.append(f"{nid}" + (f" ({label})" if label else ""))
            props = result.get("properties", {})
            if "source_code" in props:
                code = props["source_code"]
                if len(code) > 500:
                    code = code[:500] + "..."
                parts.append(f"  source_code:\n{code}")
            elif props:
                for k, v in props.items():
                    parts.append(f"  {k}: {v}")
        elif "node" in result:  # find_connected
            node = result["node"]
            nid = node.get("id", "?")
            label = node.get("label", "")
            parts.append(f"{nid}" + (f" ({label})" if label else ""))
            props = node.get("properties", {})
            for k, v in props.items():
                parts.append(f"  {k}: {v}")
        elif "info" in result:  # get_function_info
            fn = result["info"].get("function", {})
            parts.append(fn.get("name", "?"))
            params = fn.get("parameters", {}).get("properties", {})
            required = fn.get("parameters", {}).get("required", [])
            for pname, pdef in params.items():
                ptype = pdef.get("type", "?")
                pdesc = pdef.get("description", "")[:80]
                req = " (required)" if pname in required else ""
                default = pdef.get("default")
                dflt = f" = {default!r}" if default is not None else ""
                parts.append(f"  {pname} [{ptype}]{req}{dflt}: {pdesc}")

        blocks.append("\n".join(parts))

    return "\n\n".join(blocks)


def _parse_tool_calls(calls: list, graph_read_funcs: dict) -> tuple[list, list]:
    """Parse raw extracted calls into valid graph calls and issues.

    Returns ``(graph_calls, issues)`` where:
    - *graph_calls*: list of ``(name, args)`` tuples for valid calls
    - *issues*: list of human-readable error strings for malformed entries
    """
    graph_calls = []
    issues = []
    for c in calls:
        if not isinstance(c, dict):
            issues.append(f"Expected a JSON object but got: {repr(c)[:120]}")
            continue
        fn = c.get("function", {})
        if not isinstance(fn, dict):
            issues.append(f"'function' must be an object, got: {repr(fn)[:120]}")
            continue
        name = fn.get("name", "")
        args = fn.get("arguments", {})
        if name in graph_read_funcs:
            graph_calls.append((name, args))
        elif name:
            issues.append(
                f"Unknown function '{name}'. Available: "
                f"{', '.join(sorted(graph_read_funcs))}"
            )
        else:
            issues.append(f"Missing 'name' in function call: {repr(fn)[:120]}")
    return graph_calls, issues


def _format_head_start(seed_nodes: list) -> str:
    """Format semantic head-start nodes for the initial user message."""
    lines = [
        "Based on semantic similarity to the bug report, these graph "
        "nodes may be relevant:"
    ]
    for seed in seed_nodes:
        label = seed.get("label") or seed.get("name") or seed.get("id", "?")
        lines.append(
            f"- {seed.get('meta_type', 'Node')} {label} "
            f"in {seed.get('relative_path') or '?'} "
            f"(sim={seed.get('similarity', 0.0):.2f})"
        )
    lines.append(
        "Start by examining these, or use search_elements / "
        "search_by_embedding to find others."
    )
    return "\n".join(lines)


def _rejection_feedback(rejection_count: int, issues_text: str) -> str:
    template = _REJECTION_FEEDBACKS[(rejection_count - 1) % len(_REJECTION_FEEDBACKS)]
    # plain replace, not str.format — the templates embed JSON braces
    return template.replace("{n}", str(rejection_count)).replace(
        "{issues}", issues_text
    )


def _handle_rejection(
    messages: list, text: str, rejection_count: int, issues_text: str
):
    messages.append({"role": "assistant", "content": text})
    messages.append(
        {
            "role": "user",
            "content": _rejection_feedback(rejection_count, issues_text),
        }
    )


def _maybe_reset(
    messages: list, config: KGConfig, consecutive_rejections: int, reset_used: bool
):
    """Reset the conversation to the initial prompt (at most once).

    The accumulated error history itself pushes a deterministic model
    into repeating the same invalid call; a clean slate plus extra
    rounds gives the retry a real chance.  Returns
    ``(reset_used, rounds_granted)``.
    """
    if not reset_used and consecutive_rejections >= config.rejection_restart_after:
        del messages[2:]
        messages.append({"role": "user", "content": _RESTART_NOTE})
        return True, config.restart_round_grant
    return reset_used, 0


def explore(
    graph_dir: str,
    problem_statement: str,
    config: KGConfig,
) -> Dict[str, dict]:
    """
    Run the KG exploration loop.

    Returns {node_id: {label?, ...properties}} of all nodes touched during exploration.
    """
    GRAPH_READ_FUNCS, extract_function_calls = _import_graph_ops()

    # Absolute path up front: we chdir into graph_dir below, and the
    # embedding index path must stay valid regardless of CWD.
    graph_dir_abs = os.path.abspath(graph_dir)

    saved_cwd = os.getcwd()
    os.chdir(graph_dir)

    try:
        from openai import OpenAI

        client = OpenAI(base_url=config.api_base, api_key=config.api_key)

        head_start_section = ""
        embeddings_enabled = False
        if getattr(config, "use_embeddings", False):
            try:
                from .embeddings import get_top_k_for_issue, make_search_tool

                seed_nodes = get_top_k_for_issue(
                    problem_statement, graph_dir_abs, config, k=config.embedding_top_k
                )
                if seed_nodes:
                    head_start_section = _format_head_start(seed_nodes)
                GRAPH_READ_FUNCS["search_by_embedding"] = make_search_tool(
                    graph_dir_abs, config
                )
                embeddings_enabled = True
                print(
                    f"  embedding head start: {len(seed_nodes)} nodes, "
                    "search_by_embedding enabled",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"  [warn] embeddings unavailable ({exc}); " "continuing without",
                    flush=True,
                )

        system_prompt = build_fault_loc_system_prompt(use_embeddings=embeddings_enabled)

        if head_start_section:
            initial_content = (
                f"Bug report:\n{problem_statement}\n\n"
                f"{head_start_section}\n\n"
                "Find the relevant code locations in the graph."
            )
        else:
            initial_content = (
                f"Bug report:\n{problem_statement}\n\n"
                "Find the relevant code locations in the graph."
            )

        messages: list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_content},
        ]

        touched: Dict[str, dict] = {}
        total_calls = 0
        nudged = False
        reviewed = False
        call_log: list = []
        consecutive_rejections = 0
        reset_used = False
        rounds_left = config.max_rounds
        round_num = 0

        while rounds_left > 0:
            rounds_left -= 1
            round_num += 1
            request_temperature = min(
                config.temperature
                + config.escalation_temp_step * consecutive_rejections,
                config.escalation_temp_max,
            )
            if config.disable_thinking:
                resp = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=request_temperature,
                    max_tokens=config.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
            else:
                resp = client.chat.completions.create(
                    model=config.model,
                    messages=messages,
                    temperature=request_temperature,
                    max_tokens=config.max_tokens,
                )
            text = resp.choices[0].message.content
            if text is None:
                text = ""

            try:
                calls = extract_function_calls(text, max_json_length=10000)
            except Exception:
                print(
                    "  [warn] failed to parse function calls from response",
                    flush=True,
                )
                consecutive_rejections += 1
                _handle_rejection(
                    messages,
                    text,
                    consecutive_rejections,
                    "Response could not be parsed as JSON function calls.",
                )
                reset_used, granted = _maybe_reset(
                    messages, config, consecutive_rejections, reset_used
                )
                rounds_left += granted
                print(
                    f"  [round {round_num}] rejected "
                    f"({consecutive_rejections} in a row, "
                    f"temp={request_temperature:.2f})",
                    flush=True,
                )
                continue

            graph_calls, issues = _parse_tool_calls(calls, GRAPH_READ_FUNCS)

            if not graph_calls and issues:
                consecutive_rejections += 1
                issues_text = "\n".join(f"  - {s}" for s in issues)
                _handle_rejection(messages, text, consecutive_rejections, issues_text)
                reset_used, granted = _maybe_reset(
                    messages, config, consecutive_rejections, reset_used
                )
                rounds_left += granted
                print(
                    f"  [round {round_num}] rejected "
                    f"({consecutive_rejections} in a row, "
                    f"temp={request_temperature:.2f})",
                    flush=True,
                )
                continue
            consecutive_rejections = 0

            if not graph_calls:
                if not touched and not nudged:
                    nudged = True
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please use the graph tools to explore the repository. "
                                "Start with `search_elements` to find relevant code "
                                "locations based on keywords from the bug report."
                            ),
                        }
                    )
                    print(
                        f"  [round {round_num}] no calls, no nodes"
                        " touched — nudging to explore",
                        flush=True,
                    )
                    continue
                if not reviewed and touched:
                    reviewed = True
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Before finishing, critically review the code locations "
                                "you have found. Have you identified the root cause of "
                                "the bug, or is there a related function, caller, or "
                                "dependency you should still check? "
                                "If you need more information, use the graph tools. "
                                "If you are confident in your findings, respond with "
                                "no tool calls."
                            ),
                        }
                    )
                    print(
                        f"  [round {round_num}] no calls — "
                        "asking for critical review",
                        flush=True,
                    )
                    continue
                print(
                    f"  [round {round_num}] model finished" " (no graph calls)",
                    flush=True,
                )
                break

            call_descs = []
            for fn_name, fn_args in graph_calls:
                desc_parts = [fn_name]
                if "label" in fn_args:
                    desc_parts.append(f'label="{fn_args["label"]}"')
                if "query" in fn_args:
                    desc_parts.append(f'query="{str(fn_args["query"])[:40]}"')
                if "id" in fn_args:
                    desc_parts.append(f'id={fn_args["id"]}')
                if "edge_label" in fn_args:
                    desc_parts.append(f'edge={fn_args["edge_label"]}')
                if "type" in fn_args:
                    desc_parts.append(f'type={fn_args["type"]}')
                if "max_depth" in fn_args:
                    desc_parts.append(f'depth={fn_args["max_depth"]}')
                if "start_id" in fn_args:
                    desc_parts.append(f'start={str(fn_args["start_id"])[:60]}')
                if "pattern" in fn_args:
                    desc_parts.append(f'pattern={fn_args["pattern"]}')
                call_descs.append("(".join([desc_parts[0], ", ".join(desc_parts[1:])]))

            print(
                f"  [round {round_num}] {len(graph_calls)} call(s): "
                f"{', '.join(call_descs)}",
                flush=True,
            )

            round_results = []
            round_call_log = []
            for fn_name, fn_args in graph_calls:
                try:
                    result = GRAPH_READ_FUNCS[fn_name](**fn_args)
                except Exception:
                    print(f"    {fn_name} FAILED: {traceback.format_exc()}", flush=True)
                    info = GRAPH_READ_FUNCS.get(
                        "get_function_info", lambda **kw: {"success": False}
                    )(function_name=fn_name)
                    schema = (
                        json.dumps(info.get("info", {}), indent=2)
                        if info.get("success")
                        else ""
                    )
                    result = {
                        "success": False,
                        "error": (
                            f"Unexpected error calling {fn_name}.\n"
                            f"{schema}\n"
                            f"{traceback.format_exc()}"
                        ),
                    }

                round_results.append((fn_name, fn_args, result))
                nodes = _extract_node_ids(result)
                _collect_node(nodes, touched)

                hit_count = len(nodes) if result.get("success") else 0
                status = f"{hit_count} hits" if result.get("success") else "FAILED"
                arg_hint = ""
                if "label" in fn_args:
                    arg_hint = f' "{fn_args["label"]}"'
                elif "id" in fn_args:
                    arg_hint = f" {fn_args['id']}"
                print(f"    {fn_name}{arg_hint} -> {status}", flush=True)

                round_call_log.append((fn_name, fn_args, result.get("success", False)))

                total_calls += 1
                if total_calls >= config.max_tool_calls:
                    break

            call_log.append(round_call_log)

            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": f"--- TOOL RESULTS ---\n{_format_results(round_results)}",
                }
            )

            if total_calls >= config.max_tool_calls:
                print(f"  budget exhausted ({total_calls} calls)", flush=True)
                break

        from collections import Counter

        fn_counts: Counter = Counter()
        failures = 0
        for entries in call_log:
            for fn_name, _, success in entries:
                fn_counts[fn_name] += 1
                if not success:
                    failures += 1
        fn_summary = ", ".join(f"{name}×{cnt}" for name, cnt in fn_counts.most_common())
        print(
            f"  touched {len(touched)} nodes in {total_calls} calls "
            f"({fn_summary}), {failures} failure(s)",
            flush=True,
        )
        return touched

    finally:
        os.chdir(saved_cwd)
