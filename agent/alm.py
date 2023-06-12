from state_memory import RawState, HistoryState
from tools import ToolList, Tool
from llm import LLM


class ReActAgent:
    def __init__(self, llm: LLM, toolList: ToolList = None):
        self.state_memory = HistoryState()
        self.toolList = toolList
        self.PromptHead = ""

    def registerTool(self, tool: Tool) -> bool:
        return self.toolList.registerTool(tool)
