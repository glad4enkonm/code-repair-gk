import copy
import logging

logger = logging.getLogger(__name__)


def extract_schema() -> str:
    """Extract graph schema from the meta graph using existing read API.

    Must be called while CWD is the graph directory.  Queries the
    ``"meta"`` graph to discover node types, their properties (via
    ``HAS_PROPERTY`` edges), and edge types.
    """
    from tool.functions.graph_file_ops.read import get_neighbors, search_elements

    result = search_elements("meta", kind="node", include=["properties"])
    if not result.get("success"):
        return ""

    node_types = []
    edge_types = []
    for item in result.get("items", []):
        mt = item.get("properties", {}).get("meta_type", "")
        if mt == "NodeType":
            node_types.append(item["id"])
        elif mt == "PropertyType":
            continue
        else:
            edge_types.append(item["id"])

    if not node_types:
        return ""

    # Get properties per type via HAS_PROPERTY edges
    type_props = {}
    for nt in node_types:
        nb = get_neighbors(
            "meta", id=nt, edge_label="HAS_PROPERTY", direction="outgoing"
        )
        props = sorted(n["label"] for n in nb.get("nodes", []))
        if props:
            type_props[nt] = props

    lines = ["GRAPH SCHEMA"]
    for nt in sorted(type_props.keys()):
        lines.append(f"  {nt}: {', '.join(type_props[nt])}")
    lines.append("")
    lines.append("Edge types: " + ", ".join(sorted(edge_types)))
    return "\n".join(lines)


FAULT_LOCALIZATION_MESSAGE = """You are a code fault localization assistant. You have access to a knowledge graph of a Python repository that represents its structure — files, classes, functions, methods, variables, and the relationships between them.

Your task: given a bug report (issue statement), explore the knowledge graph to find the code locations relevant to fixing this bug.

WORKFLOW:
1. Start with `search_elements` to find candidate functions/classes/methods whose names match keywords from the bug report.
2. Use `get_neighbors` to explore connections from a found node (callers, callees, imports).
3. Use `traverse` for multi-hop path finding (e.g. follow CALLS edges to trace execution flow).
4. Use `get_element(id, include=["properties"])` to read the actual `source_code` of a node.
5. Use `find_connected(id, type="File")` to find which file a node belongs to.
6. If unsure about a function's parameters, call `get_function_info(function_name="...")` to see its schema.
7. When you believe you have found the relevant code locations, respond with TEXT ONLY (no tool calls) summarizing your findings.

STRATEGY:
- Search across the "data" graph level.
- The GRAPH SCHEMA above lists available node types and edge types — use this to guide your exploration.
- Use `label_contains` for substring matching on function/class names.
- Only read source_code for nodes that seem genuinely relevant — don't read everything.
- Be efficient: budget is limited to ~15 tool calls total. Stop when you're confident you have the right files/functions.

RESPOND TEXT-ONLY (no function calls) when you are done. Your text response should list the node IDs or file paths you found relevant.
"""

FUNCTION_CALL_FORMAT = """

**Function Call Format (JSON array):**
```json
[
  {
    "function": {
      "name": "search_elements",
      "arguments": {"graph": "data", "meta_type": "Function", "label_contains": "keyword"}
    }
  }
]
```

Key Guidelines:
- Include JSON anywhere in your response.
- Use maximum one tool-call array per reply.
- RESPOND TEXT-ONLY when done — this signals you have finished exploring.
"""


SEARCH_BY_EMBEDDING_DEF = {
    "type": "function",
    "function": {
        "name": "search_by_embedding",
        "description": (
            "Semantic search over graph nodes by natural-language query. "
            "Use when keyword search (search_elements) misses because the "
            "bug report wording differs from the code naming. "
            "Returns {id, label, properties: {meta_type, name, "
            "relative_path, similarity}}."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to find, in natural language "
                        "(e.g. 'function that validates user credentials')"
                    ),
                },
                "meta_type": {
                    "type": "string",
                    "description": (
                        "Optional node type filter: 'Function', 'Method', "
                        "'Class', 'File', 'Import', 'Variable', 'Decorator'"
                    ),
                },
                "n_results": {
                    "type": "integer",
                    "description": "Max results to return (default 10)",
                },
            },
            "required": ["query"],
        },
    },
}

EMBEDDING_SEARCH_NOTE = """
SEMANTIC SEARCH:
- `search_by_embedding(query="...")` finds nodes by meaning, not by name.
  Prefer it when the bug report's wording does not match code identifiers.
- If the initial message lists semantically similar seed nodes, start
  from them before searching further.
"""


def _customize_for_code_graph(read_tools: dict) -> dict:
    """Override generic graph function definitions with code-specific defaults.

    The underlying ``find_connected`` in ``graph_file_ops`` is generic —
    no assumptions about edge labels or node types.  Here we specialize
    it for the code knowledge graph by setting ``edge_label`` default
    to ``"CONTAINS"`` and enriching the descriptions with code-specific
    examples (File, Class, Method, etc.).
    """
    result = dict(read_tools)
    if "find_connected" in result:
        fc = copy.deepcopy(result["find_connected"])
        fn = fc["function"]
        fn["description"] = (
            "Find the nearest node of a given type reachable by "
            "following edges in the specified direction. "
            "Useful for finding the File that contains a Method, "
            "the Class that defines a Variable, etc."
        )
        props = fn["parameters"]["properties"]
        props["edge_label"]["default"] = "CONTAINS"
        props["type"]["description"] = (
            "Target node meta_type "
            "(e.g. 'File', 'Class', 'Directory', 'Method', "
            "'Function', 'Variable', 'Import')."
        )
        result["find_connected"] = fc
    return result


def build_fault_loc_system_prompt(use_embeddings: bool = False) -> str:
    try:
        from tool.commands.ui import build_system_prompt_with_tools
        from tool.functions.graph_file_ops import get_function_definitions

        all_defs = get_function_definitions()
        # get_function_definitions returns a list of {"type":"function","function":{...}}
        # build_system_prompt_with_tools expects a dict keyed by function name
        read_names = {
            "search_elements",
            "get_neighbors",
            "traverse",
            "get_element",
            "find_connected",
            "get_function_info",
        }
        read_tools = {}
        for d in all_defs:
            fn = d.get("function", {})
            name = fn.get("name", "")
            if name in read_names:
                read_tools[name] = d
        read_tools = _customize_for_code_graph(read_tools)
        if use_embeddings:
            read_tools["search_by_embedding"] = copy.deepcopy(SEARCH_BY_EMBEDDING_DEF)

        schema = extract_schema()
        system_text = FAULT_LOCALIZATION_MESSAGE
        if use_embeddings:
            system_text = system_text + EMBEDDING_SEARCH_NOTE
        if schema:
            system_text = schema + "\n\n" + system_text

        return build_system_prompt_with_tools(
            system_text,
            "",
            FUNCTION_CALL_FORMAT,
            read_tools,
        )
    except ImportError as exc:
        logger.warning(
            "tool package unavailable (%s) — building fallback prompt "
            "without read tools",
            exc,
        )
        system_text = FAULT_LOCALIZATION_MESSAGE
        if use_embeddings:
            system_text = system_text + EMBEDDING_SEARCH_NOTE
        return system_text + FUNCTION_CALL_FORMAT
