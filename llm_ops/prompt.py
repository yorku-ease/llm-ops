# The ast module parses python code into a tree 
import ast

def extract_code_expressions(f_string: str):
    # ast magic, extracts the python code in the { brackets in f strings like: f"The db password is {db_password}" 
    # where db_password is a variable defined in whatever scope the f string is declared in
    values = ast.parse(f'f"""{f_string}"""').body[0].value.values
    f_exprs = [v for v in values if isinstance(v, ast.FormattedValue)]
    return f_exprs


class Prompt:
    def __init__(self, template):
        if not Prompt._validate_template(template):
            raise ValueError("Template cannot contain python code other than variable names")
        self.template = template
        self.input_vars = self._extract_input_var_names(template)

    def _extract_input_var_names(self, f_string: str):
        input_vars = []
        for e in extract_code_expressions(f_string):
            input_vars.append(e.value.id)
        
        return input_vars

    @staticmethod
    def _validate_template(template: str) -> bool:
        for e in extract_code_expressions(template):
            if not isinstance(e.value, ast.Name):
                return False
        return True

    def make(self, **kwargs):
        missing_vars = []
        for v in self.input_vars:
            if v not in kwargs:
                missing_vars.append(v)
        if missing_vars:
            raise ValueError(f"Missing input variables: {missing_vars}")
        return self.template.format(**kwargs)

if __name__ == "__main__":
    def test_validate_template():
        good_template = "{good}"
        bad_template = "{bad**2}"
        
        assert Prompt._validate_template(good_template)
        assert not Prompt._validate_template(bad_template) 
        assert Prompt(good_template).make(good="Test") == "Test"
        return True
    
    test_validate_template()

