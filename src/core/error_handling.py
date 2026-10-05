from functools import wraps
from typing import Any, Dict

from langgraph.types import Command


class GraphError(Exception):
    def __init__(self, message: str, node: str, state: dict):
        self.message = message
        self.node = node
        self.state = state
        super().__init__(f"Error in {node}: {message}")


def with_error_handling(func):
    @wraps(func)
    async def wrapper(state: Dict[str, Any], config) -> Command:
        try:
            return await func(state, config)
        except Exception as e:
            error_info = {
                "error": str(e),
                "node": func.__name__,
                "state_snapshot": state,
            }
            return Command(goto="error_handler", update=error_info)

    return wrapper
