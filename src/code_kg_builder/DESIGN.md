# code_kg_builder — Design Document

> Complete specification for the knowledge graph builder that creates Data-level
> graphs from source code repositories, driven by the Meta-level schema.

## 1. Context: The 3-Level Knowledge Graph

The project uses a three-level graph architecture (see `KG-SWE-bench-Design.md`
for the full vision):

| Level | File | Role | Mutable? |
|---|---|---|---|
| **Origin** (L2) | `origin.json` | Axioms — defines what `NodeType`, `EdgeType`, `PropertyType` *are*. Meta-relationships: `HAS_SOURCE`, `HAS_TARGET`, `HAS_PROPERTY`. | Fixed |
| **Meta** (L1) | `meta.json` | Schema — domain-specific types for Python repos: `Directory`, `File`, `Class`, `Function`, `Method`, `Variable`, `Decorator`, `Import`, `Dependency`, `Exception`, `TypeAnnotation`. Edge types: `CONTAINS`, `IMPORTS`, `DEPENDS_ON`, `DECORATES`, `CALLS`, `INHERITS`, `RAISES`, `CATCHES`, `HAS_TYPE`. | Extensible |
| **Data** (L0) | `data.json` | Instances — concrete entities from the repository (`File-src/cli.py`, `Class-.../QuerySet`, etc.) | Built by this package |

**Rules-as-data**: Origin validates Meta; Meta validates Data. Every write goes
through `update_graph()` which applies atomic commits with structural + Z3
verification (rollback on failure).

**Live example**: `swe_bench_kg/` contains the origin.json, meta.json, and an
empty data.json that this builder will populate.

## 2. Purpose of `code_kg_builder`

Given a Meta graph (schema) and a source code repository, produce a Data graph
(populated instances) that conforms to the schema.

### Why the `executor` property?

Every NodeType and EdgeType in Meta carries an `executor` property — a Python
`"module:Class"` path pointing to the code that builds instances of that type.
This serves two purposes:

1. **Auto-loading & extensibility**: The orchestrator reads Meta, resolves
   executor paths via `importlib`, and dispatches them automatically. Adding a
   new type to Meta = add the executor path + write the class. Zero orchestrator
   changes.

2. **Provenance & debugging**: If a Data node is wrong, look at its type's
   `executor` in Meta → go straight to the code that created it. No guessing.

Multiple Meta types can share the same executor. For example, `Class`,
`Function`, `Method`, `Variable`, `Decorator`, and `Import` are all produced by
one `AstStructureExecutor` in a single AST pass per file.

## 3. Changes to Existing Graphs

### 3.1 Origin — add `executor` PropertyType

Add `executor` as a new PropertyType, available to `NodeType` and `EdgeType`:

**New node:**
```json
{"id": "executor", "label": "executor"}
```

**New edges:**
```json
{"source": "NodeType", "target": "executor", "label": "HAS_PROPERTY", "id": "NodeType-HAS_PROPERTY-executor"}
{"source": "EdgeType", "target": "executor", "label": "HAS_PROPERTY", "id": "EdgeType-HAS_PROPERTY-executor"}
```

**New allValues entry:**
```json
"executor": {
    "description": "Python module:Class path for the builder executor. Enables auto-loading and provenance tracking.",
    "dataType": "string",
    "origin_type": "MetaProperty"
}
```

This does NOT create `HAS_PROPERTY executor` edges in Meta — `executor` is a
property of the Meta node itself (like `description`), not a property that Data
instances must have. The DataValidator only checks properties defined by
Meta-level `HAS_PROPERTY` edges (e.g., `Directory-HAS_PROPERTY-relative_path`),
so Data instances are never checked for `executor`.

### 3.2 Meta — add `executor` to every NodeType and EdgeType

Every type in Meta's `allValues` gets an `executor` field:

```json
"Directory":  {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.filesystem:FilesystemExecutor"}
"File":       {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.filesystem:FilesystemExecutor"}
"Class":      {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"Function":   {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"Method":     {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"Variable":   {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"Decorator":  {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"Import":     {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"Dependency": {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.dependency:DependencyExecutor"}
"Exception":  {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.raises_edge:RaisesEdgeExecutor"}
"TypeAnnotation": {"meta_type": "NodeType", "description": "...", "executor": "code_kg_builder.executors.python.has_type_edge:HasTypeEdgeExecutor"}
"CONTAINS":   {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"IMPORTS":    {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.imports_edge:ImportsEdgeExecutor"}
"DEPENDS_ON": {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.dependency:DependencyExecutor"}
"DECORATES":  {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.ast_structure:AstStructureExecutor"}
"CALLS":      {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.calls_edge:CallsEdgeExecutor"}
"INHERITS":   {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.inherits_edge:InheritsEdgeExecutor"}
"RAISES":     {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.raises_edge:RaisesEdgeExecutor"}
"CATCHES":    {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.catches_edge:CatchesEdgeExecutor"}
"HAS_TYPE":   {"meta_type": "EdgeType", "description": "...", "executor": "code_kg_builder.executors.python.has_type_edge:HasTypeEdgeExecutor"}
```

## 4. Package Structure

Follows the File & Folder Placement Rules in `AGENTS.md`.

```
src/code_kg_builder/
├── __init__.py
├── DESIGN.md                       # This file
├── cli.py                          # Click CLI entry point
├── orchestrator.py                 # Pipeline: read meta → load executors → run → verify
├── meta_schema.py                  # Parse meta.json → MetaSchema (reuses verification logic)
├── build_state.py                  # Node index + batch accumulator
├── graph_writer.py                 # Wraps update_graph(graph="data") for batched commits
├── build_verifier.py               # Wraps verify_graph_update("data")
├── executor_base.py                # ABC: Executor.run(ctx) -> Iterator[BatchResult]
├── executor_resolver.py            # importlib: "module:Class" → instance (fail-fast)
├── context.py                      # BuildContext dataclass
└── executors/
    ├── __init__.py
    └── python/
        ├── __init__.py
        ├── filesystem.py           # FilesystemExecutor — Directory, File, CONTAINS (Dir→File)
        ├── ast_structure.py        # AstStructureExecutor — Class, Function, Method, Variable,
        │                           #   Decorator, Import + CONTAINS, DECORATES
        ├── dependency.py           # DependencyExecutor — Dependency, DEPENDS_ON
        ├── calls_edge.py           # CallsEdgeExecutor — CALLS
        ├── imports_edge.py         # ImportsEdgeExecutor — IMPORTS
        ├── inherits_edge.py        # InheritsEdgeExecutor — INHERITS
        ├── raises_edge.py          # RaisesEdgeExecutor — RAISES + Exception nodes
        ├── catches_edge.py         # CatchesEdgeExecutor — CATCHES
        ├── has_type_edge.py        # HasTypeEdgeExecutor — HAS_TYPE + TypeAnnotation nodes
        └── utils/
            ├── __init__.py
            ├── id_convention.py    # ID patterns per entity type
            ├── source_extract.py   # Extract source_code by line range
            ├── complexity.py       # McCabe cyclomatic complexity
            └── scope_resolver.py   # Scope maps for call/import resolution
```

### Why this layout

- **Top level** = language-agnostic infrastructure only (Rule 5)
- **`executors/python/`** = executors are the core concept; language is the
  subdivision (Rule 2). Adding `executors/javascript/` later touches nothing in
  `python/` (Rule 4)
- **`executors/python/utils/`** = utils are used only by Python executors, so
  they're co-located (Rule 1). No top-level `utils/` (Rule 7)
- **Filenames** drop the `_executor` suffix — the folder provides context
  (Rule 3). Class names keep it (`FilesystemExecutor`) for code clarity
- **Tests** mirror this structure with `_test.py` suffix (Rule 6)

## 5. Core Interfaces

### 5.1 BatchResult

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class BatchResult:
    """A batch of graph elements to be committed atomically via update_graph()."""
    nodes_data: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {"id": str, "label": str}

    edges_data: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {"id": str, "source": str, "target": str, "label": str}

    all_values_data: List[Dict[str, Any]] = field(default_factory=list)
    # Each: {"id": str, <property_name>: <primitive_value>, ...}

    errors: List[str] = field(default_factory=list)
```

### 5.2 BuildContext

```python
from pathlib import Path
from typing import Any, Dict, List, Optional

@dataclass
class NodeInfo:
    """Cached info about a committed node, for edge resolution."""
    id: str
    label: str
    meta_type: str  # "Class", "Function", etc.
    properties: Dict[str, Any]

@dataclass
class BuildContext:
    """Shared state passed to every executor during the build."""
    repo_path: Path
    meta_schema: "MetaSchema"
    node_index: Dict[str, NodeInfo]       # id → NodeInfo, updated on each commit
    python_files: List[Path]              # populated by FilesystemExecutor
    ast_cache: Dict[Path, Any]            # file → ast.Module, populated lazily
    graph_dir: str                        # directory containing origin/meta/data JSON
    errors: List[Dict[str, Any]]         # accumulated errors: {file, executor, error}

    def log_error(self, source: str, error: str) -> None: ...
    def update_index(self, batch: BatchResult) -> None: ...
```

### 5.3 Executor

```python
from abc import ABC, abstractmethod
from typing import Iterator

class Executor(ABC):
    """Base class for all executors.

    An executor processes repository source code and yields BatchResult objects.
    Each batch is committed atomically via update_graph(). If a batch fails
    (UNSAT or exception), it is logged and skipped — previously committed
    batches are preserved.
    """

    phase: int = 0
    # 0 = filesystem/structural (Directory, File, Dependency)
    # 1 = AST nodes (Class, Function, Method, Variable, Decorator, Import)
    # 2 = edge resolution (CALLS, IMPORTS, INHERITS, RAISES, CATCHES, HAS_TYPE)

    @abstractmethod
    def run(self, ctx: BuildContext) -> Iterator[BatchResult]:
        """Process the repository and yield batches for atomic commits.

        Typically yields one batch per file for resilience: if one file
        fails, all previously committed files are preserved.
        """
        ...
```

### 5.4 MetaSchema

Parses `meta.json` into a structured representation. Reuses the extraction
logic from `verification/data_validator.py:DataValidator._extract_meta_schema()`.

```python
@dataclass
class EdgeSignature:
    source_type: str  # e.g., "Directory"
    target_type: str  # e.g., "File"

@dataclass
class MetaSchema:
    node_types: Dict[str, Dict[str, Any]]      # type_name → allValues (incl. executor)
    edge_types: Dict[str, Dict[str, Any]]
    property_types: Dict[str, Dict[str, Any]]
    edge_signatures: Dict[str, List[EdgeSignature]]
    # Edge type → list of (source_type, target_type) pairs
    type_properties: Dict[str, List[str]]      # node/edge type → property names
    property_constraints: Dict[str, Dict]      # "{Type}-HAS_PROPERTY-{Prop}" → constraints

    @classmethod
    def from_file(cls, path: str) -> "MetaSchema": ...

    def get_executors(self) -> Dict[str, str]:
        """Return {executor_path: [type_names]} — deduplicated."""
        ...
```

## 6. Executor → Type Mapping

| Executor class | Phase | Produces NodeTypes | Produces EdgeTypes |
|---|---|---|---|
| `FilesystemExecutor` | 0 | Directory, File | CONTAINS (Directory→File) |
| `AstStructureExecutor` | 1 | Class, Function, Method, Variable, Decorator, Import | CONTAINS (File→Class, Class→Method, Method→Variable), DECORATES |
| `DependencyExecutor` | 1 | Dependency | DEPENDS_ON |
| `CallsEdgeExecutor` | 2 | — | CALLS |
| `ImportsEdgeExecutor` | 2 | — | IMPORTS |
| `InheritsEdgeExecutor` | 2 | — | INHERITS |
| `RaisesEdgeExecutor` | 2 | Exception | RAISES |
| `CatchesEdgeExecutor` | 2 | — | CATCHES |
| `HasTypeEdgeExecutor` | 2 | TypeAnnotation | HAS_TYPE |

**Derived nodes**: `Exception` and `TypeAnnotation` are created as side-effects
by edge executors (RaisesEdgeExecutor, HasTypeEdgeExecutor). Their `executor`
property in Meta points to their "primary" creator for provenance. Other
executors that also create them do idempotent upserts (node already exists →
update properties, don't duplicate).

## 7. Orchestrator Pipeline

```
CLI (--repo PATH --meta FILE --data FILE --origin FILE)
  │
  ├─ 1. Parse MetaSchema from meta.json
  │     Extract: node_types, edge_types, edge_signatures, executor paths
  │
  ├─ 2. Resolve executors
  │     Deduplicate executor paths from all types
  │     For each: importlib resolve "module:Class" → instance
  │     FAIL FAST: if any path is broken, report and abort
  │
  ├─ 3. Sort executors by phase (0 → 1 → 2)
  │
  ├─ 4. Run executors (batched, resilient)
  │     for executor in sorted_executors:
  │         try:
  │             for batch in executor.run(ctx):
  │                 result = graph_writer.commit(batch)
  │                 if result["status"] == "sat":
  │                     ctx.update_index(batch)
  │                 else:
  │                     ctx.log_warning(batch, result["errors"])
  │         except Exception as e:
  │             ctx.log_error(str(executor), str(e))
  │             continue   ← skip, preserve committed state
  │
  └─ 5. Verify
        build_verifier.verify()  → verify_graph_update("data", graph_dir)
        Report: node count, edge count, violations, skipped files
```

### Write strategy: batched & resilient

- Each executor yields batches (typically one per file)
- Each batch → `update_graph(graph="data", ...)` → atomic commit with Z3
- If a file fails: exception caught, logged with file path + traceback, skipped
- Previously committed data is preserved (no rollback of prior batches)
- Problematic files are reflected in the log for later investigation

## 8. ID Convention

| Entity | ID pattern | Example |
|---|---|---|
| Directory | `Directory-{rel_path}` | `Directory-django/db/models` |
| File | `File-{rel_path}` | `File-django/db/models/query.py` |
| Class | `Class-{rel_path}-{name}` | `Class-django/db/models/query.py-QuerySet` |
| Function | `Function-{rel_path}-{name}` | `Function-django/db/models/query.py-annotate` |
| Method | `Method-{rel_path}-{ClassName}-{name}` | `Method-django/.../query.py-QuerySet-_filter_or_exclude` |
| Variable | `Variable-{parent_id}-{name}` | `Variable-Method-...-args` |
| Decorator | `Decorator-{target_id}-{name}` | `Decorator-Method-...-cached_property` |
| Import | `Import-{file_id}-{idx}` | `Import-File-.../query.py-2` |
| Dependency | `Dependency-{name}` | `Dependency-django` |
| Exception | `Exception-{name}` | `Exception-FieldError` |
| TypeAnnotation | `TypeAnnotation-{name}` | `TypeAnnotation-QuerySet` |

**Edge IDs**: `Source-RELATION-Target` (e.g.,
`File-.../query.py-CONTAINS-Class-.../QuerySet`).

## 9. Reuse of Existing Code

| Existing code | Location | Reused as |
|---|---|---|
| `update_graph()` | `src/tool/functions/graph_file_ops/write.py` | `GraphWriter.commit()` wrapper — atomic batch write with Z3 validation |
| `verify_graph_update()` | `src/verification/graph_verifier.py:15` | `BuildVerifier.verify()` — post-build verification |
| `GraphParser` | `src/verification/graph_parser.py:47` | Parse origin/meta/data JSON files |
| `_extract_meta_schema()` | `src/verification/data_validator.py:64` | `MetaSchema._extract()` — same logic for extracting node_types, edge_signatures, constraints |
| AST patterns | `src/tool/functions/reduce_python_code.py` | Reference for `ast_structure.py` — imports, classes, functions, decorators, docstrings |

### Import paths

```python
from tool.functions.graph_file_ops.write import update_graph
from verification.graph_verifier import verify_graph_update
from verification.graph_parser import GraphParser
```

The package `code_kg_builder` is configured in `pyproject.toml` alongside
`tool` and `verification` (all under `src/` with
`[tool.setuptools.packages.find] where = ["src"]`).

### Graph file resolution

`update_graph()` resolves graph files from the current working directory or
a `graph_store/` subdirectory (see `graph_file_ops/read.py`). The CLI must be
run from the directory containing `origin.json`, `meta.json`, `data.json`, or
the `--graph-dir` option must point to it.

## 10. CLI

```bash
python -m code_kg_builder \
    --repo /path/to/repo \
    --meta swe_bench_kg/meta.json \
    --data swe_bench_kg/data.json \
    --origin swe_bench_kg/origin.json \
    --graph-dir swe_bench_kg
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--repo` | (required) | Path to the repository to analyze |
| `--meta` | `swe_bench_kg/meta.json` | Path to meta.json |
| `--data` | `swe_bench_kg/data.json` | Path to data.json (output) |
| `--origin` | `swe_bench_kg/origin.json` | Path to origin.json |
| `--graph-dir` | (auto from --meta) | Directory containing the 3 JSON files |
| `--max-files` | `None` | Limit number of files (debugging) |
| `--no-verify` | `False` | Skip final Z3 verification |

## 11. TDD Implementation Order

Follow RED → GREEN → REFACTOR for each step (see `AGENTS.md` TDD section).

| Step | Component | Test file | Test asserts |
|---|---|---|---|
| 1 | `utils/id_convention.py` | `id_convention_test.py` | ID patterns match table in section 8 for each entity type |
| 2 | `utils/source_extract.py` | `source_extract_test.py` | Given file content + line range → correct source snippet |
| 3 | `utils/complexity.py` | `complexity_test.py` | Given AST → correct McCabe complexity (if/elif/for/while/and/or/except/with/assert/comprehension) |
| 4 | `meta_schema.py` | `meta_schema_test.py` | Parse `swe_bench_kg/meta.json` → correct node_types, edge_signatures, executor paths, property constraints |
| 5 | `executor_resolver.py` | `executor_resolver_test.py` | Resolve `"module:Class"` → instance; fail-fast on bad path with clear error |
| 6 | `build_state.py` | `build_state_test.py` | Accumulate nodes/edges/allValues; idempotent upsert; node index query by id and by meta_type |
| 7 | `graph_writer.py` | `graph_writer_test.py` | Wrap `update_graph`; batch commit returns SAT/UNSAT; correct file written |
| 8 | `context.py` | (covered by orchestrator test) | BuildContext populated correctly; error accumulation |
| 9 | `executors/python/filesystem.py` | `filesystem_test.py` | Walk temp test repo → Directory + File nodes + CONTAINS edges; properties: relative_path, line_count |
| 10 | `executors/python/ast_structure.py` | `ast_structure_test.py` | Parse temp test .py → Class, Function, Method, Variable, Decorator, Import nodes + CONTAINS, DECORATES edges; properties: name, line_start, line_end, source_code, docstring, complexity, is_async, is_staticmethod, is_classmethod, return_type, value |
| 11 | `executors/python/dependency.py` | `dependency_test.py` | Parse temp requirements.txt → Dependency nodes + DEPENDS_ON edges; properties: name, version |
| 12 | `executors/python/calls_edge.py` | `calls_edge_test.py` | Resolve calls in test file → CALLS edges between existing Function/Method nodes |
| 13 | `executors/python/imports_edge.py` | `imports_edge_test.py` | Resolve imports → IMPORTS edges |
| 14 | `executors/python/inherits_edge.py` | `inherits_edge_test.py` | Resolve class bases → INHERITS edges |
| 15 | `executors/python/raises_edge.py` | `raises_edge_test.py` | Resolve ast.Raise → RAISES edges + Exception nodes created |
| 16 | `executors/python/catches_edge.py` | `catches_edge_test.py` | Resolve ast.Try.handlers → CATCHES edges |
| 17 | `executors/python/has_type_edge.py` | `has_type_edge_test.py` | Resolve AnnAssign.annotation → HAS_TYPE edges + TypeAnnotation nodes |
| 18 | `executors/python/utils/scope_resolver.py` | `scope_resolver_test.py` | Build scope map for test file; resolve self.method(), cls.method(), bare func() |
| 19 | `orchestrator.py` | `orchestrator_test.py` | End-to-end: load meta → resolve executors → run → verify output on small temp repo |
| 20 | `cli.py` + `build_verifier.py` | `cli_test.py` | CLI produces data.json, passes verification |

Each step: write failing test → implement minimum to pass → refactor → next.

### Test fixtures

`tests/code_kg_builder/conftest.py` should provide:

- `tmp_repo` fixture: creates a small Python project in a temp directory with
  known structure (2-3 directories, 5-10 .py files, a requirements.txt)
- `meta_schema_fixture`: loads `swe_bench_kg/meta.json` as MetaSchema
- `build_context_fixture`: BuildContext with tmp_repo and meta_schema
- `graph_dir_fixture`: copies swe_bench_kg/*.json to temp dir for isolated
  write/verify tests

## 12. Data Node Properties (per Meta schema)

Reference for which properties each node type should have when built:

| NodeType | Mandatory properties | Optional properties |
|---|---|---|
| Directory | relative_path | — |
| File | relative_path | line_count |
| Class | name, source_code | line_start, line_end, docstring, complexity |
| Function | name, source_code | line_start, line_end, return_type, is_async, docstring, complexity |
| Method | name, source_code | line_start, line_end, return_type, is_async, is_staticmethod, is_classmethod, docstring, complexity |
| Variable | name | line_start, value |
| Decorator | name | line_start |
| Import | — | module_path, imported_name, line_number |
| Dependency | name | version |
| Exception | name | — |
| TypeAnnotation | name | — |

(From `meta.json` allValues cardinality constraints — `isMandatory: true` =
mandatory, `isMandatory: false` = optional.)

## 13. Edge Signatures (per Meta schema)

Edge types support multiple source→target signatures (paired by index in Meta):

| EdgeType | Signatures |
|---|---|
| CONTAINS | Directory→File, File→Class, Class→Method, Method→Variable |
| IMPORTS | Function→Function, File→Dependency, (File→Import) |
| DEPENDS_ON | File→Dependency |
| DECORATES | Decorator→Method, Decorator→Class |
| CALLS | Function→Function, Method→Method |
| INHERITS | Class→Class |
| RAISES | Function→Exception |
| CATCHES | Function→Exception |
| HAS_TYPE | Variable→TypeAnnotation |

(From `meta.json` edges — paired `HAS_SOURCE`/`HAS_TARGET` with matching index
suffixes.)

## 14. What We Do NOT Build (v1)

- P2 layers (ChangeRecord, TestCase, DebuggingPattern, Route, Middleware)
- `embedding` properties
- Graph caching by (repo, commit_hash)
- Layout generation
- Multi-language support (only Python for now — but the architecture supports
  adding `executors/javascript/` later without touching existing code)
- Cross-repository dependency resolution (only local project imports resolved)

## 15. Extensibility Demonstration

**Adding `TestCase` NodeType (future P2):**

1. Add `TestCase` NodeType + properties to `meta.json` via `update_graph(graph="meta")`,
   including `"executor": "code_kg_builder.executors.python.test_case:TestCaseExecutor"`
2. Create `src/code_kg_builder/executors/python/test_case.py` with
   `TestCaseExecutor(Executor)` class
3. **Zero changes** to: orchestrator, build_state, graph_writer, verifier,
   meta_schema parser, or any existing executor

The orchestrator reads Meta at startup, sees `TestCase` with its executor path,
resolves it via importlib, and calls `run()` automatically.
