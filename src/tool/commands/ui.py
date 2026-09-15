"""
Prompt-building utility shared by the pipeline's system prompts.

Vendored subset of tool.commands.ui: only build_system_prompt_with_tools
is included. The upstream module additionally contains rich-based
terminal display helpers, which the pipeline does not use.
"""

from typing import Any, Dict


def build_system_prompt_with_tools(
    system_message: str,
    tools_header: str,
    call_format: str,
    tools: Dict[str, Any],
) -> str:
    """Build a full system prompt by appending tool descriptions to a base message.

    Args:
        system_message: The command-specific system message.
        tools_header: The AVAILABLE_TOOLS_HEADER constant for this command.
        call_format: The FUNCTION_CALL_FORMAT_ARRAY constant for this command.
        tools: Mapping of tool name to tool definition dict.

    Returns:
        The complete system prompt string.
    """
    prompt = system_message + "\n\n" + tools_header
    for tool_name, tool_def in tools.items():
        function_info = tool_def.get("function", {})
        description = function_info.get("description", "No description")
        parameters = function_info.get("parameters", {})
        required = parameters.get("required", [])
        properties = parameters.get("properties", {})
        optional_params = [
            param for param in properties.keys() if param not in required
        ]
        prompt += f"\n- **{tool_name}**: {description}"
        if required:
            prompt += f" (Required: {', '.join(required)})"
        if optional_params:
            prompt += f" (Optional: {', '.join(optional_params)})"
    prompt += call_format
    return prompt
