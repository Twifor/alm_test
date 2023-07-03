from agent.tools import Tool
from typing import Union, Dict


class AnswerTool(Tool):
    def __init__(self, verification_func, gt=None):
        super().__init__()
        self.invoke_label = "Answer"
        self.func = verification_func
        self.gt = gt

    def invoke(self, invoke_data) -> Union[str, int, bool, Dict]:
        res = invoke_data
        is_correct = self.func(res)
        return (
            "You are CORRECT." if is_correct else "You are INCORRECT.",
            1 if is_correct else 0,
            True,
            {"gt_answer": self.gt} if self.gt is not None else {},
        )

    def description(self) -> str:
        return "Answer(answer), you can submit your final answer by using this tool and then exit the process. You should call this function with caution so as not to submit wrong answers."


class BeginTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "Begin"

    def invoke(self, invoke_data) -> Union[str, int, bool, Dict]:
        return "You can not directly invoke this tool", 0, False, {}

    def description(self) -> str:
        return "Begin(), this tool don not have any function. You do not need to invoke this tool."
