from langchain_core.messages import ToolMessage

message = ToolMessage(content="Test", tool_call_id="123")


print(message)
