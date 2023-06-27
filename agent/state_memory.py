class ReActRawHistoryState:
    def __init__(self):
        super().__init__()
        self.steps = []
        self.request = ""

    def updateState(self, thought, action, observation, rk=True):
        id = len(self.steps) + 1
        if rk:
            self.steps.append(
                {
                    "Thought %d: " % id: thought.strip(),
                    "Action %d: " % id: action,
                    "Observation %d: " % id: observation,
                }
            )
        else:
            self.steps.append(
                {
                    "Thought: ": thought.strip(),
                    "Action: ": action,
                    "Observation: ": observation,
                }
            )

    def num(self):
        return len(self.steps)

    def description(self):
        res = ""
        for step in self.steps:
            for k, v in step.items():
                res += k + str(v) + "\n"
        return res

    def reset(self):
        self.steps = []
