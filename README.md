# Philosophy

Allow us just enough flexibility to build LLM-powered solutions for our research while
taking care of as much of the boilerplate as possible and abstracting away the choice of model.

Why not Langchain or similar? Langchain does not support all the models we want to use and I believe it is more complicated than necessary for our purposes. This library allows us to iterate more quickly on the
solutions to our problems.

# Use

## Prompt

```python
from llm_ops import Prompt

instruction = "Please write a poem about someone called {name} while speaking like a {thing}"
print(Prompt(instruction).make(name="Marios", thing="Pirate"))
# prints:
# Please write a poem about someone called Marios while speaking like a Pirate
```

## Environment Variables

See `.env.template` for a list of environment variables that may need to be set to use the models.

## Message

The object passed to and returned by LLMs while generating results. See model section
for more information

## Model

All models are used in the same way. They accept a list of Message objects as input
and return a Message. If the response from the model is text, a Message is returned.
If the model attempts to make a tool call, a ToolCallMessage is returned. The type of
the message can be determined by the Message's "type" attribute.

If environment variables are required to use the model (for example, Openai's models - see .env.template), they should be set 

```python
from llm_ops.model import OpenAIModel
from llm_ops import Message

model = OpenAIModel("gpt-3.5")

query = Message(
    role="user", 
    content="Please write a poem about someone called Marios while speaking like a Pirate"
)

output = model.generate([query])

print(output.role) # assistant
print(output.content) # (A poem about marios)
```

## Tool Use

The best way to define a tool is to decorate a function defined with a google-style docstring
(see below) with `Tool.from_fn`. This results in a `Tool` object that can be called just like the original function, but allows it to be used by LLMs to help them answer queries. 

The docstring is used to determine the function description and argument descriptions passed to the model, so that it knows how to use the function

```python
from llm_ops import Tool

Tool.from_fn
def google_search(query: str, k: int) -> list[str]:
    """Returns the first k urls obtained by searching 'query' on Google

    Args:
        query (str): The query to search on Google
        k (int): The number of urls to return
    
    Returns:
        list[str]: A list of urls
    """
    # pretend it performs a google search
    return urls

# Now we can pass it to the model defined in the previous section
# to help it answer queries
query = Message(
    role="user",
    content="Please return the url of the linkedin account of Marios Fokaefs"
)

tools = [google_search]

output = model([query], tools = tools)
print(type(output)) # ToolCallMessage

# To return the output of a tool call to a model, we create a ToolOutputMessage
# this is done most simply by calling ToolCallMessage.to_tool_output(tools)

tool_outputs = output.to_tool_output(tools)

# Now we can call the model again with the tool outputs so that it can generate the final answer

output = model([query, tool_outputs], tools = tools)

# It could be that the model wants to perform another tool call, but for this demo
# we will assume that it does not:
assert output.type == "text", "The model wants to perform another tool call"

print(output.content)
```

## LLMFunction

Fortunately, for the majority of our research, we will not have to worry about many of the details I have described above. If no user interaction is required with the model, then the workflow is as follows: 

1. Pass instruction containing inputs and desired tools to model
2. Get response from model

The only thing that changes for one problem is the inputs to the model, so it would be nice if
we could turn this functionality into a normal function that accepts the inputs to our problem and returns the result. 

For example, lets take the problem in the previous section, but allow the name to vary. In other words, given a name, we want the model to return the url of their linkedin.

```python
from llm_ops import LLMFunction

prompt = Prompt("Pleae return the url of the linkedin account of {name}")
model = OpenAIModel("gpt-3.5")
# assuming that google_search is defined as above
get_linkedin = LLMFunction(prompt, model, tools=[google_search])

# now we can use this like a function
print(get_linkedin(name="Mackenzie van Zanden")) # Mackenzie's linkedin
print(get_linkedin(name="Marios Fokaefs")) # Marios' Linkedin

```

The LLMFunction handles tool calls internally. 

Of course, these functions can have as many inputs as we want, and we can use as many as we want. 

Having solved quite a few problems with LLMs, the most effective way to use them is to break the problem down into as many steps as possible, so that the LLM solves an easier task each time. 

# TODO

1. Support for all models we want to use
2. Specifying an output format for an LLMFunction (e.g. make sure it returns an int, or a dict with specific keys, etc)
3. Print tool calls in LLMFunctions to aid in debugging our solutions
4. Setup virtualenv instructions
5. Test everything except Prompt
6. .env.template