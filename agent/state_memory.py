from memory.vectorstores.chroma import Chroma
from memory.vectorstores.vectorstore import VectorStoreRetrieverMemory
from memory.embeddings import OpenAIEmbeddings
from abc import ABC, abstractmethod
from agent.llm import LLM

class ReActBaseHistoryState(ABC):
     
    @abstractmethod
    def updateState(self, thought, action, observation, rk=True):
        pass
     
    @abstractmethod
    def num(self):
        pass

    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def reset(self):
        pass

class ReActRawHistoryState(ReActBaseHistoryState):
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


class ReActSummaryState(ReActRawHistoryState):
    def __init__(self, llm: LLM):
        super().__init__()
        self.llm = llm
        self.history = ""

    def description(self):
        if self.steps == []:
            return ""
        self.history += "\nTheir next step:\n"
        for step in self.steps:
            for k, v in step.items():
                self.history += k + str(v) + "\n"
        prompt = "You are a smart assistant who helps to generate a summary of the given context.\n"
        prompt += "Now I will provide the history of conversation between another assistant and some tools or users. You need to summarize them to a brief passage.\n"
        prompt += "You summary has an significant impact on another assistant. You need to guarantee that your summary does not lack any important information.\n"        
        prompt += "The history will be showed here:\n"
        prompt += self.history
        prompt += "Summary:"
        summary = self.llm.response(prompt, "\n")
        res = "You can recover from last steps from the summary below:\n"
        res += summary + "\n"
        self.history = summary
        self.steps = []
        return res
