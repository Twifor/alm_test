import requests
import json
import os
import sys
from io import StringIO
from typing import Dict, Optional
from agent.tools import Tool
from typing import Union, Dict


class PythonREPL:
    """Simulates a standalone Python REPL."""

    def __init__(self) -> None:
        self.globals: Optional[Dict] = globals()
        self.locals: Optional[Dict] = None

    def run(self, command: str) -> str:
        """Run command with own globals/locals and returns anything printed."""
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            exec(command, self.globals, self.locals)
            sys.stdout = old_stdout
            output = mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            output = repr(e)
        if output == "":
            return "Code executed successfully. You should use print in your code to get the value of your variable."
        return output


class RunPythonTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "RunPython"
        self.python_repl = PythonREPL()

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        code = invoke_data.strip().strip("```")
        if code.startswith('"') or code.startswith("'"):
            code = code[1:-1]
        code = code.replace("\\n", "\n")
        code = code.replace('\\"', '"')
        codes = code.split("\n")
        if len(codes) > 1:
            codes[-1] = "print(" + codes[-1] + ")"
        code = "\n".join(codes)
        return self.python_repl.run(code), 0, False, {}

    def description(self) -> str:
        return "RunPython(code), A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."

    def reset(self) -> None:
        self.python_repl = PythonREPL()
