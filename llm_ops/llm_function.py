from llm_ops.prompt import Prompt
from llm_ops.model import Model
from llm_ops.message import Message, ToolOutputMessage
from llm_ops.tool import Tool, get_tool_outputs


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

    def __call__(self, **inputs) -> str:
        model_input = self.prompt.make(**inputs)
        input_messages = self.messages + [Message("user", model_input)]

        # handle tool calls until model provides answer
        while True:
            model_output = self.model.generate(input_messages, tools=self.tools)
            
            if model_output.type == "text":
                break
            # output is tool call
            tool_calls = model_output.tool_calls
            tool_outputs = get_tool_outputs(self.tools, tool_calls)
            input_messages += [ToolOutputMessage(model_output, tool_outputs)]

        return model_output.content


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

    p = Prompt("Call the function test_fn and return the result")
    from llm_ops.model import OpenAIModel
    llm_fn = LLMFunction(p, OpenAIModel(), tools = [test_fn])
    print(llm_fn())