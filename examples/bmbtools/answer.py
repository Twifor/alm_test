from agent.tools import Tool
from typing import Union, Dict


class AnswerTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "Answer"

    def invoke(self, invoke_data) -> Union[str, int, bool, Dict]:
        return invoke_data, 0, True, {}

    def description(self) -> str:
        return "Answer(answer), you can submit your final answer by using this tool and then exit the process. You should call this function with caution so as not to submit wrong answers."
