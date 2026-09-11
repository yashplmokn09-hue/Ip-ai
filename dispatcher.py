"""
IP-AIv1 Tool Dispatcher
Parses tool call syntax from model output and executes tools.

Model outputs tool calls like:
  <tool>web_search {"query": "latest news"}</tool>
"""

import re
import json
from web_tools import TOOLS, TOOL_SCHEMAS


TOOL_CALL_RE = re.compile(r"<tool>(.*?)\{(.*?)\}</tool>", re.DOTALL)
SYSTEM_TOOLS = "\n".join([
    f"- {t['name']}: {t['description']}" for t in TOOL_SCHEMAS
])

SYSTEM_PROMPT = f"""You are IP-AIv1, a helpful AI assistant with real-time internet access.

You have access to these tools:
{SYSTEM_TOOLS}

To use a tool, output exactly:
<tool>tool_name {{"param": "value"}}</tool>

After getting the tool result, use it to answer the user's question.
Always be helpful, accurate, and concise."""


def parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract tool name and args from model output."""
    m = TOOL_CALL_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    try:
        args = json.loads("{" + m.group(2) + "}")
    except Exception:
        return None
    return name, args


def execute_tool(name: str, args: dict) -> str:
    """Run a tool and return result as string."""
    fn = TOOLS.get(name)
    if not fn:
        return f"Unknown tool: {name}"
    try:
        result = fn(**args)
        return json.dumps(result, ensure_ascii=False, indent=2)[:2000]
    except Exception as e:
        return f"Tool error: {e}"


def maybe_run_tool(model_output: str) -> tuple[str, bool]:
    """
    If model output contains a tool call, execute it and return
    (tool_result_string, True). Otherwise return ("", False).
    """
    call = parse_tool_call(model_output)
    if call is None:
        return "", False
    name, args = call
    result = execute_tool(name, args)
    return result, True
