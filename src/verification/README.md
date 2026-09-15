# Graph Verification System using Z3

This project implements a formal verification system for the three-tier graph architecture using the Z3 SMT solver.

## Overview

The system implements a three-tier verification architecture:

1. **Origin → Meta Validation**: Verifies that the Meta Graph (schema layer) conforms to the Origin Graph axioms
2. **Meta → Data Validation**: Verifies that the Data Graph (instances) conforms to the Meta Graph schema
3. Ensures data validation rules are consistent and satisfiable
4. Type constraints and cardinalities are properly enforced at all levels

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Verify Meta Graph (Schema Validation)

```bash
python verify_meta.py
```

This will:
1. Load the origin.json axiomatization
2. Generate Z3 constraints from the origin rules
3. Verify that meta.json conforms to these constraints
4. Report any violations with detailed diagnostics

### Verify Data Graph (Instance Validation)

```bash
python verify_data.py
```

This will:
1. Load the meta.json schema definition
2. Extract type definitions, property constraints, and cardinality rules
3. Verify that data.json instances conform to the schema
4. Report any violations with detailed diagnostics

### Generate Constraints

```bash
python generate_constraints.py
```

This will output the SMT-LIB representation of the constraints derived from origin.json.

## Architecture

### Core Components

- `graph_parser.py`: Parses JSON graph representations
- `origin_axioms.py`: Encodes the Origin Graph axioms as Z3 constraints
- `meta_validator.py`: Validates Meta Graph against Origin axioms
- `data_validator.py`: Validates Data Graph against Meta schema

### Verification Scripts

- `verify_meta.py`: Validates meta.json against origin.json
- `verify_data.py`: Validates data.json against meta.json
- `generate_constraints.py`: Generates SMT constraints from origin.json
- `test_verification.py`: Test suite for verification system

## Verification Rules

### Meta Graph Validation (Origin → Meta)

The system checks that meta.json conforms to origin.json:

1. **Type Conformance**: All Meta nodes/edges use types defined in Origin
2. **Property Typing**: Properties have correct data types as per Origin
3. **Edge Signatures**: Edge source/target types match Origin definitions
4. **Cardinality Constraints**: min/max counts are properly defined
5. **Mandatory Properties**: Required properties are present
6. **Z3 Constraint Satisfaction**: All formal constraints are satisfiable

### Data Graph Validation (Meta → Data)

The system checks that data.json conforms to meta.json:

1. **Node Type Conformance**: All data nodes use types defined in Meta
2. **Edge Type Conformance**: All data edges use types defined in Meta
3. **Edge Connectivity**: Edges connect to valid source/target node types per schema
4. **Required Properties**: Nodes have all mandatory properties
5. **Property Data Types**: Property values match expected types (string, integer, etc.)
6. **Cardinality Constraints**: Relationship counts satisfy min/max constraints
7. **Validation Patterns**: Values match regex patterns when specified
