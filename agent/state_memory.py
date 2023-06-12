class RawState:
    def __init__(self):
        pass

    def updateState(self):
        raise NotImplementedError("You need to implement this function.")

    def description(self):
        raise NotImplementedError("You need to implement this function.")


class HistoryState(RawState):
    def __init__(self):
        super().__init__()
        self.PromptHead = "You can read the history of the current procedure, which will help you to take next action.\n"
        self.PromptHead += "For each step, the format of history is like the following one:\n"
        self.PromptHead += "Thought 1: Your thought at step.\n"
        self.PromptHead += "Action 1: Your action at this step.\n"
        self.PromptHead += "Observation 1: The external observation you received at this step.\n\n"
        self.PromptHead += "The history of your procedure is described as:\n"
        self.PromptEnd = "\nNow, accoring to the history of the procedure, take your next action and provide your thought.\n"
        self.steps = []

    def updateState(self, thought, action, observation):
        id = len(self.steps) + 1
        self.steps.append(
            {"Thought %d: " % id: thought, "Action %d: " %
                id: action, "Observation %d: " % id: observation}
        )

    def description(self):
        res = self.PromptHead
        for step in self.steps:
            for k, v in step.items():
                res += k + v + "\n"
        return res + self.PromptEnd
