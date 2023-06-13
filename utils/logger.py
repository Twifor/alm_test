import os
import json


class ALMLogger:
    def __init__(self, event_id):
        self.event_id = event_id
        self.log_path = "./logs/"
        self.event_path = self.log_path + event_id + ".log"
        if os.path.exists(self.log_path) == False:
            os.mkdir(self.log_path)
        if os.path.exists(self.event_path) == True:
            raise ValueError(self.event_path +
                             " already exists. Please modify your event_id.")
        self.chains = []
        self._answer = ""
        self._prompt = ""
        self._gt_answer = ""
        self._query = ""

    def close(self):
        self.simple_log.close()

    def finish(self):
        self.simple_log = open(self.event_path, "w")
        obj = {
            "prompt": self._prompt,
            "query": self._query,
            "chains": self.chains,
            "answer": self._answer,
            "gt_answer": self._gt_answer
        }
        json_str = json.dumps(obj, indent=4, separators=(',', ':'))
        self.simple_log.write(json_str)
        self.simple_log.close()

    def prompt(self, prompt):
        self._prompt = prompt

    def query(self, query):
        self._query = query

    def step(self, thought, action,  observation):
        self.chains.append(
            {"thought": thought, "action": action, "observation": observation}
        )

    def answer(self, answer):
        self._answer = answer

    def gt_answer(self, gt_answer):
        self._gt_answer = gt_answer
