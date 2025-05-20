from .prompt import Prompt
from .model import Model
from .message import Message, ToolOutputMessage
from .tool import Tool, get_tool_outputs


class LLMFunction:
    """Allows to use an llm like a function
    
    Define a Prompt which defines the task for the LLM to accomplish and the inputs (see Prompt class 
    to understand how inputs work). Object can then be used like a normal python function, which takes 
    as input the inputs to the prompt as keywords and outputs the output of the LLM. If tools are provided
    then the answer from the llm is returned after it makes as many tool calls as desired.
    """
    def __init__(self, prompt: Prompt, model: Model, tools: list[Tool] = None, system_prompt: str =None):
        self.prompt = prompt
        self.model = model
        self.tools = tools
        self.messages = []
        if system_prompt is not None:
            self.messages.append(Message("system", system_prompt))

    def call(self, **inputs):
        return self(**inputs)

    def __call__(self, **inputs):
        model_input = self.prompt.make(**inputs)
        input_messages = self.messages + [Message("user", model_input)]

        # handle tool calls until model provides answer
        while True:
            model_output = self.model.generate(input_messages)
            
            if model_output.type == "text":
                break
            # output is tool call
            tool_calls = model_output.tool_calls
            tool_outputs = get_tool_outputs(self.tools, tool_calls)
            input_messages += [ToolOutputMessage(model_output, tool_outputs)]

        return model_output.content


