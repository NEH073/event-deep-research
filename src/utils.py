import os

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool


@tool(
    description="Mandatory reflection tool. Analyze results and plan the next search query."
)
def think_tool(reflection: str) -> str:
    """Mandatory reflection step. Use this to analyze the last result, identify gaps, and formulate the EXACT query for the next search.

    You MUST use this tool immediately after every ResearchEventsTool call.

    Analyze if an additional call to the ResearchEventsTool is needed to fill the gaps or the research is completed. When is completed, you must call the FinishResearchTool.

    The `reflection` argument must follow the structure defined in the system prompt, culminating in the precise search query you will use next.

    Args:
        reflection: Structured analysis of the last result, current gaps, and the PLANNED QUERY for the next step.

    Returns:
        Confirmation and instruction to proceed to the next step.
    """
    # The return value is crucial. It becomes the ToolMessage the LLM sees next.
    # By explicitly telling it what to do, we break the loop.
    return f"Reflection recorded. {reflection}"


def get_api_key_for_model(model_name: str, config: RunnableConfig):
    """Get API key for a specific model from environment or config."""
    model_name = model_name.lower()

    if model_name.startswith("openai:"):
        return os.getenv("OPENAI_API_KEY")
    elif model_name.startswith("anthropic:"):
        return os.getenv("ANTHROPIC_API_KEY")
    elif model_name.startswith("google"):

        return os.getenv("GOOGLE_API_KEY")
    elif model_name.startswith("ollama:"):
        # Ollama doesn't need API key
        return None
    return None


def get_buffer_string_with_tools(messages: list[BaseMessage]) -> str:
    """Return a readable transcript showing roles, including tool names for ToolMessages."""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"Human: {m.content}")
        elif isinstance(m, AIMessage):
            ai_content = f"AI: {m.content}"
            # Include tool calls if present
            if hasattr(m, "tool_calls") and m.tool_calls:
                tool_calls_str = ", ".join(
                    [
                        f"{tc.get('name', 'unknown')}({tc.get('args', {})})"
                        for tc in m.tool_calls
                    ]
                )
                ai_content += f" [Tool calls: {tool_calls_str}]"
            lines.append(ai_content)
        elif isinstance(m, SystemMessage):
            lines.append(f"System: {m.content}")
        elif isinstance(m, ToolMessage):
            # Include tool name if available
            tool_name = (
                getattr(m, "name", None) or getattr(m, "tool", None) or "unknown_tool"
            )
            lines.append(f"Tool[{tool_name}]: {m.content}")
        else:
            # fallback for unknown or custom message types
            lines.append(f"{m.__class__.__name__}: {m.content}")
    return "\n".join(lines)


def get_langfuse_handler():
    try:
        from langfuse.langchain import CallbackHandler

        return CallbackHandler()
    except ImportError:
        return None
