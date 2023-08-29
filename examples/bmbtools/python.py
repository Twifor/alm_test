import requests
import json
import os
import sys
from io import StringIO
from typing import Dict, Optional
from agent.tools import Tool
from typing import Union, Dict
import eventlet
# from wrapt_timeout_decorator import *


class PythonREPL:
    """Simulates a standalone Python REPL."""

    def __init__(self) -> None:
        self.globals: Optional[Dict] = globals()
        self.locals: Optional[Dict] = None

    def run(self, command: str) -> str:
        print(command)
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            with eventlet.Timeout(3, False):
                exec(command, self.globals, self.locals)
                sys.stdout = old_stdout
                output = mystdout.getvalue()
        except TimeoutError:
            sys.stdout = old_stdout
            return "Timeout. You need to simplify your code."
        except Exception as e:
            sys.stdout = old_stdout
            output = "You code is invalid or incomplete: " + repr(e)
        if output == "":
            return "Code executed successfully. But nothing output (You should use 'print()', or your 'print' is not reached.)."
        if len(output) >= 100:
            output =  output[0:100] + "..."
        return output


class RunPythonTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "RunPython"
        self.python_repl = PythonREPL()
        self.globals = globals()

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        code = invoke_data.strip().strip("```")
        if code.startswith('"') or code.startswith("'"):
            code = code[1:-1]
        code = code.replace("\\n", "\n")
        code = code.replace('\\"', '"')

        codes = code.split("\n")
        if len(codes) > 1:
            if codes[-1][0] != "\t" and codes[-1][0] != " ":
                codes[-1] = f"try:\n   print({codes[-1]})\nexcept:\n    pass\n"
        code = "\n".join(codes)

        return self.python_repl.run(code), 0, False, {}

    def description(self) -> str:
        return "RunPython(your_python_code), A Python interpreter. Use this to execute python codes."

    def reset(self) -> None:
        self.python_repl = PythonREPL()
        self.python_repl.globals = self.globals
        self.python_repl.locals = None
