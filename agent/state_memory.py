class ReActRawHistoryState:
    def __init__(self):
        super().__init__()
        self.steps = []
        self.request = ""

    def updateState(self, thought, action, observation):
        id = len(self.steps) + 1
        self.steps.append(
            {"Thought %d: " % id: thought, "Action %d: " %
                id: action, "Observation %d: " % id: observation}
        )

    def num(self):
        return len(self.steps)

    def description(self):
        res = ""
        for step in self.steps:
            for k, v in step.items():
                res += k + v + "\n"
        return res

    def reset(self):
        self.steps = []
