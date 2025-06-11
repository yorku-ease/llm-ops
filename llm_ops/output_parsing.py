from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel


def llm_output_to_pydantic(llm_output: str, pydantic_model: type[BaseModel]) -> BaseModel:
    """Converts llm_output into an instance of pydantic_model"""
    return PydanticOutputParser(pydantic_object=pydantic_model).parse(llm_output)

def pydantic_format_instructions(pydantic_model: type[BaseModel]) -> str:
    """Returns instructions for formatting output such that it can be converted into an instance of pydantic_model"""
    return PydanticOutputParser(pydantic_object=pydantic_model).get_format_instructions()
# if we want to make the output some pydantic class, we need to add instructions to the prompt 
# so we can offer two functions: generate pydantic format prompt, convert output to pydantic

if __name__ == "__main__":
    class TestModel(BaseModel):
        test_property: str
    
    print(
        pydantic_format_instructions(TestModel)
    )
    model_instance = llm_output_to_pydantic('{"test_property": "test"}', TestModel)
    print(model_instance)
    assert(
        isinstance(model_instance, TestModel)
    )