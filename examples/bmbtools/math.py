from agent.tools import Tool
from typing import Union, Dict


class RoundTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "Round"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        number = float(eval(invoke_data))
        return round(number), 0, False, {}

    def description(self) -> str:
        return "Round(number), transform a number (float) to an integer."
