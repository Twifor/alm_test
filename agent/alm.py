from agent.state_memory import ReActRawHistoryState
from agent.tools import ToolList, Tool
from agent.llm import GPT3_5LLM
from utils.logger import ALMLogger


class ReActAgent:
    def __init__(self, llm: GPT3_5LLM, logger: ALMLogger, toolList: ToolList = None, max_steps=7):
        self.llm = llm
        self.state_memory = ReActRawHistoryState()
        self.toolList = toolList
        if self.toolList == None:
            self.toolList = ToolList()
        self.PromptHead = "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation.\n"
        self.current_steps = 0
        self.max_steps = max_steps
        self.logger = logger

    def setRequest(self, request):
        self.state_memory.init_request(request)
        self.logger.query(request)

    def registerTool(self, tool: Tool) -> bool:
        return self.toolList.registerTool(tool)

    def step(self):
        prompt = self.PromptHead
        prompt += self.toolList.description()
        self.logger.prompt(prompt)
        prompt += self.state_memory.description()
        prompt += f"Thought {self.current_steps + 1}:"
        llm_response = self.llm.response(
            prompt, stop=f"\nObservation {self.current_steps + 1}:")
        thought, action = llm_response.strip().split(
            f"\nAction {self.current_steps + 1}: ")
        obs, reward, isDone = self.toolList.invoke(action, self.logger)
        self.state_memory.updateState(thought, action, obs)
        self.logger.step(thought, action, obs)
        self.current_steps += 1
        if self.current_steps > self.max_steps:
            return True, 0
        return isDone, reward
