"""
Utility for handling JSON extraction and validation from text.

Vendored subset of tool.utils.json_tools: the clipboard round-trip for
invalid JSON (copy_invalid_json=True, an interactive-CLI convenience the
pipeline never uses) and rich console output are removed; parsing
behavior is unchanged. Logging uses the standard library.
"""

import json
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extracts valid JSON objects from a string.

    This function first attempts to parse the entire string as a JSON
    object or array. If that fails, it searches for JSON within
    markdown-style code blocks.

    Args:
        text: The string to extract JSON from.

    Returns:
        A list of valid JSON objects found in the text.
    """
    json_objects: List[Any] = []

    # Attempt to parse the entire text first
    try:
        data = json.loads(text)
        if isinstance(data, list):
            json_objects.extend(d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            json_objects.append(data)

        if json_objects:
            return json_objects

    except json.JSONDecodeError:
        logger.debug("Text is not a single JSON object. Checking for code blocks.")

    # If parsing the whole text fails, look for JSON in code blocks
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)

    if not code_blocks:
        stripped_text = text.strip()
        # Check if the text looks like a JSON object but is invalid
        if stripped_text.startswith(("{", "[")) and stripped_text.endswith(("}", "]")):
            try:
                json.loads(stripped_text)
            except json.JSONDecodeError as e:
                logger.warning(
                    "Invalid JSON-like string: %s. Text: %s...",
                    e,
                    stripped_text[:100],
                )
        return []

    for block in code_blocks:
        block = block.strip()
        if not block:
            continue
        try:
            data = json.loads(block)
            if isinstance(data, list):
                json_objects.extend(d for d in data if isinstance(d, dict))
            elif isinstance(data, dict):
                json_objects.append(data)
        except json.JSONDecodeError as e:
            logger.warning(
                "Invalid JSON in code block: %s. Content: %s...",
                e,
                block[:80],
            )
            continue

    return json_objects


def extract_function_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extracts valid JSON function calls from the LLM response text.
    A function call is a JSON object with a 'function' key.

    Args:
        text: The LLM response string to extract function calls from.

    Returns:
        A list of valid function call dictionaries.
    """
    json_objects = extract_json_from_text(text)
    function_calls = [
        obj for obj in json_objects if isinstance(obj, dict) and "function" in obj
    ]
    return function_calls
