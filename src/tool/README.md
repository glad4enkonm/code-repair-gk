# Vendored subset of `tool`

This directory is a trimmed vendored copy of the author's `tool`
package. It exists so that this repository is self-contained: the
knowledge-graph pipeline (`kg.build_graphs`, `kg.context`, `kg.explore`,
`kg.prompts`, `code_kg_builder.graph_writer`) imports a handful of
`tool` modules for graph queries, graph writes, and prompt assembly.
Only those modules are vendored here; everything else was left out.
The Z3-based graph verification used by graph writes ships as the
sibling `src/verification/` package.

## What is vendored

| Module | Status | Used by |
|---|---|---|
| `functions/graph_file_ops/read.py`, `_common.py`, `write.py`, `__init__.py` | verbatim | `kg.context` (pool assembly), `kg.explore`, `kg.prompts`, `code_kg_builder.graph_writer` (graph commits) |
| `utils/json_tools.py` | trimmed | `kg.explore` (parses LLM function calls) |
| `commands/ui.py` | trimmed | `kg.prompts` (system prompt builder) |
| `src/verification/` (sibling package) | verbatim | graph write verification (Z3); the pipeline's Stage-1 commits run with verification off by default |

## What was trimmed and why

- `tool/__init__.py`, `functions/__init__.py`, `utils/__init__.py`:
  replaced with minimal stubs. Upstream `__init__` files import the
  full CLI and function registry.
- `commands/ui.py`: only `build_system_prompt_with_tools` is kept; the
  rich-based terminal display helpers are removed.
- `utils/json_tools.py`: the `copy_invalid_json` clipboard round-trip
  (an interactive-CLI convenience) and rich console output are removed;
  parsing behavior is unchanged and logging uses the standard library.
- Not vendored at all: `file_operations`, `registry`,
  `get_function_info`, `logger`, `clipboard`, `truncation`,
  `exceptions`, the CLI, and everything else in upstream `tool`.

The trimming keeps every import used by the pipeline resolvable with
identical runtime behavior for pipeline call patterns.
