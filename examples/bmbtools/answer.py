from agent.tools import Tool
from typing import Union, Dict


class AnswerTool(Tool):
    def __init__(self, verification_func):
        super().__init__()
        self.invoke_label = "Answer"
        self.func = verification_func

    def invoke(self, invoke_data) -> Union[str, int, bool, Dict]:
        res = invoke_data
        is_correct = self.func(res)
        return "You are CORRECT." if is_correct else "You are INCORRECT.", 1 if is_correct else 0, True, {}

    def description(self) -> str:
        return "Answer(answer), you can submit your final answer by using this tool and then exit the process. You should call this function with caution so as not to submit wrong answers."
