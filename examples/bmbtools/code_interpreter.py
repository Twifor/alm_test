from agent.tools import Tool
from typing import Union, Dict


class CodeInterpreter:
    def __init__(self, timeout=300):
        self.globals = {}
        self.locals = {}
        self.timeout = timeout

    def execute_code(self, code):
        try:
            # Wrap the code in an eval() call to return the result
            wrapped_code = f"__result__ = eval({repr(code)}, globals(), locals())"
            exec(wrapped_code, self.globals, self.locals)
            return self.locals.get("__result__", None)
        except Exception as e:
            try:
                # If eval fails, attempt to exec the code without returning a result
                exec(code, self.globals, self.locals)
                return "Code executed successfully. Using ExecuteCode('expression') to calculate the value of one expression."
            except Exception as e:
                return f"Error: {str(e)}"

    def reset_session(self):
        self.globals = {}
        self.locals = {}


class ExecuteCodeTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "ExecuteCode"
        self.interpreter = CodeInterpreter()

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        code = invoke_data.strip().strip("```")
        if code.startswith('"') or code.startswith("'"):
            code = code[1:-1]
        code = code.replace("\\n", "\n")
        code = code.replace('\\"', '"')
        res = self.interpreter.execute_code(code)
        return str(res), 0, False, {}

    def description(self) -> str:
        return 'ExecuteCode(code), execute Python expressions with Python Interpreter, can be used as a simple calculator e.g.,"(123 + 234) / 23 * 19.".'

    def reset(self) -> None:
        self.interpreter = CodeInterpreter()
