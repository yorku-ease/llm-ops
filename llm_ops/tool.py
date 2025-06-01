from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from langchain_core.tools import tool, BaseTool


@dataclass
class Tool:
    """Fn must have type annotations for any arguments"""
    name: str
    description: str
    fn: Callable
    _tool: BaseTool
    
    @property
    def arg_types(self):
        return {
            arg_name: type 
            for arg_name, type in self.fn.__annotations__.items() 
            if arg_name != "return"
        }

    @property
    def args_schema(self):
        """Returns dict containing description and type information about function arguments"""
        return self._tool.args
    
    def __call__(self, **kwargs): 
        return self.fn(**kwargs)
    
    @classmethod
    def from_fn(cls, fn: Callable) -> Tool:
        """Creates a Tool from a function with a google-style docstring.

        A google style docstring looks like this, containing an args and 
        results section exactly as below. These sections are used to generate
        the descriptions and types of the arguments, which are passed to models
        so they can understand how to use tools

        Args:
            fn (Callable): The function to use as a Tool.
        
        Returns:
            Tool: A tool that can be used by LLMs 
        """
        return tool(parse_docstring=True)(fn)

@dataclass
class ToolCall:
    name: str
    params: dict[str, str]

def handle_tool_call(tool, tool_call):
    # cast arguments to correct types
    tool_arg_types = tool.arg_types
    fn_args = {}
    for param, value in tool_call.params.items():
        fn_args[param] = tool_arg_types[param](value)
    
    return tool(**fn_args)


def get_tool_outputs(tools: list[Tool], tool_calls: list[ToolCall]):
    name_to_tool = {t.name: t for t in tools}
    outputs = {}
    for tc in tool_calls:
        appropriate_tool = name_to_tool.get(tc.name)
        if appropriate_tool is None:
            raise ValueError(f"Tool call {tc} attempting to call tool that does not exist: {tc.name}")
        outputs[appropriate_tool.name] = handle_tool_call(appropriate_tool, tc)
    
    return outputs