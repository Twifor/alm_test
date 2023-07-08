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
        return output


class RunPythonTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "RunPython"
        self.python_repl = PythonREPL()

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        invoke_data = invoke_data.strip().strip("```")
        invoke_data = invoke_data.replace("\\n", "\n")
        return self.python_repl.run(invoke_data), 0, False, {}

    def description(self) -> str:
        return "RunPython(command), A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`."
