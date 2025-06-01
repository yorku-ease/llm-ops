from dataclasses import dataclass
from .tool import ToolCall, Tool, get_tool_outputs


class Message:
    type = "text"
    def __init__(self, role: str, content: str):
        self.role = role # ['assistant', 'user', 'system']
        self.content = content

class ToolCallMessage(Message):
    type = "tool_call"
    def __init__(self, tool_calls: list[ToolCall], model_original_message: dict):
        super().__init__(None, None)
        self.tool_calls = tool_calls
        self._orig_msg = model_original_message # needed to make sure format is consistent, handled by model class 

    def to_tool_output(self, tools: list[Tool]):
        return ToolOutputMessage(
            self, 
            get_tool_outputs(tools, self.tool_calls)
        )

class ToolOutputMessage(Message):
    type = "tool_output"
    def __init__(self, tool_call_msg: ToolCallMessage, tool_outputs: dict):
        super().__init__(None, None)
        self.tool_call_msg = tool_call_msg # tool call associated with these outputs
        self.tool_outputs = tool_outputs
    
@dataclass
class MessageHistory:
    messages: list[Message]


