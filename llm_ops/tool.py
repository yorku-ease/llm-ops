from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any
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
        return dict(self._tool.args)
    
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
        _tool = tool(parse_docstring=True)(fn)
        return cls(
            _tool.name,
            _tool.description,
            fn,
            _tool
        )

@dataclass
class ToolCall:
    name: str
    params: dict[str, str]

def cast_value_to_str_arg_type(str_arg_type: str, value: Any):
    """Cast value to the type in the str e.g. str_arg_type="<class 'int'>" """
    return eval(f"{str_arg_type}({value})")

def handle_tool_call(tool, tool_call):
    # cast arguments to correct types
    tool_arg_types = tool.arg_types
    fn_args = {}
    for param, value in tool_call.params.items():
        val_type = tool_arg_types[param]
        casted_value = None
        if type(val_type) is str: # TODO: strange behaviour with arg types move logic to Tool.arg_types
            casted_value = cast_value_to_str_arg_type(tool_arg_types[param], value)
        else:
            casted_value = val_type(value)
        fn_args[param] = casted_value
    
    # call tool
    return tool(**fn_args)


def get_tool_outputs(tools: list[Tool], tool_calls: list[ToolCall]) -> list[Any]: # We need to return one output for every tool call because there could be multiple calls to the same tool
    name_to_tool = {t.name: t for t in tools}
    outputs = []
    for tc in tool_calls:
        appropriate_tool = name_to_tool.get(tc.name)
        if appropriate_tool is None:
            raise ValueError(f"Tool call {tc} attempting to call tool that does not exist: {tc.name}")
        outputs.append(handle_tool_call(appropriate_tool, tc))
    
    return outputs

if __name__ == "__main__":

    @Tool.from_fn
    def test_fn(a: int, b: int = None) -> int:
        """This is a test fn.
        
        This is more description.

        Args:
            a (int): Input to the function

        Returns:
            int: The thing returned by the function
        """
        return a
    
    print(test_fn)
    print(test_fn.args_schema)
    print(test_fn.arg_types)

    assert(
        get_tool_outputs(
            [test_fn],
            [
                ToolCall("test_fn", {"a": "1"}), # make sure arg casting works
                ToolCall("test_fn", {"a": 3})
            ] 
        ) == [1, 3]
    )