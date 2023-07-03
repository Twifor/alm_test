from agent.state_memory import *
from agent.tools import ToolList, Tool
from agent.llm import GPT3_5LLM
from utils.logger import logger
import warnings


class ReActToolAgent:
    def __init__(
        self,
        llm: GPT3_5LLM,
        tool: Tool,
        state:ReActBaseHistoryState = None,
    ):
        self.llm = llm
        self.toolList = ToolList()
        self.toolList.registerTool(tool)
        self.PromptHead = "Solve a task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation.\n"
        self.PromptHead += 'You must output your answer in the following format:\nFirstly, output your thought and then output "Action:" followed by your action. Finally, output "Observation:" to finish your output.\n'
        self.PromptHead += 'The action you provide should be the format like "LookUp(Kam Heskin)". Except this, do not output anything else. You need to provide the tool_label like "LookUp" and then provide your parameters in parentheses.\n'
        self.PromptHead += "You can only use the tools we provide to you. The list of tools you can use will be showed later.\n"
        self.PromptHead += "You can ONLY invoke ONE tool at each step. You CANNOT invoke two or more tools at one step.\n"
        self.PromptTail = (
            "Now based on the following information and the task, take the next step.\n"
        )
        self.ExamplePrompt = "There are some examples, which describes the output format (including the corresponding observation) you need to follow:\n"
        self.request = ""
        self.state = state
        if self.state == None:
            self.state = ReActRawHistoryState()

    def setRequest(self, request):
        self.request = request

    def registerTool(self, tool: Tool) -> bool:
        return self.toolList.registerTool(tool)

    def step(self):
        prompt = self.PromptHead + "\n"
        prompt += self.ExamplePrompt + EXAMPLES + "\n\n"
        prompt += self.toolList.description(use_examples=False) + "\n"
        prompt += self.PromptTail + f"Task: {self.request}\n"
        prompt += self.state.description()
        if self.request == "":
            warnings.warn("Request is empty.")
        prompt += f"Then take your next step.\nThought:"
        llm_response = self.llm.response(prompt, stop=f"\nObservation:")
        try:
            thought, action = llm_response.strip().split(f"Action:")
        except:
            warnings.warn("LLM fail recover.")
            action = self.llm.response(
                prompt + llm_response + "Action: ", stop=f"\nObservation:"
            )
            thought = llm_response
        obs, reward, isDone = self.toolList.invoke(action)
        return prompt, thought, action, obs, reward, isDone


class AgentNetWork:
    def __init__(self, llm:LLM=None):
        self.tool_label2tool = {}
        self.tool_label2agent = {}
        self.links = set()
        self.history_state:ReActBaseHistoryState = ReActRawHistoryState()
        self.now = None
        self.current_tool_label = ""
        self.task = ""
        self.isFinished = False
        self.current_steps = 0
        self.log_history = {}
        self.tool_info = {}

    def output_TAO(self, now_tool_label, next_tools, thought, action, observation):
        print(f"At step \033[31m{self.current_steps + 1}\033[0m:")
        print(f"\033[35mToolAgent\033[0m: {now_tool_label}")
        print(f"\033[35mNext Tools\033[0m: {next_tools}")
        print(f"\033[32mThought\033[0m: {thought.strip()}")
        print(f"\033[33mAction\033[0m: {action.strip()}")
        print(f"\033[34mObservation\033[0m: {str(observation).strip()}")
        print("==" * 20)
        self.log_history["chains"].append(
            {
                "now": now_tool_label,
                "next_tools": next_tools,
                "Thought": thought,
                "Action": action,
                "Observation": observation,
            }
        )

    def addToolAgent(self, toolAgent: ReActToolAgent):
        tool_label = list(toolAgent.toolList.tool_map.items())[0][0]
        tool = list(toolAgent.toolList.tool_map.items())[0][1]
        if tool_label in self.tool_label2tool.keys():
            raise ValueError("Redundant tool_label!")
        self.tool_label2tool[tool_label] = tool
        self.tool_label2agent[tool_label] = toolAgent

    def link(self, src_tool, dest_tool):
        src_tool_label = src_tool.invoke_label
        dest_tool_label = dest_tool.invoke_label
        if src_tool_label not in self.tool_label2agent.keys():
            raise ValueError(f"src_tool_label {src_tool_label} not found.")
        if dest_tool_label not in self.tool_label2agent.keys():
            raise ValueError(f"dest_tool_label {dest_tool_label} not found.")
        if (src_tool_label, dest_tool_label) in self.links:
            raise ValueError(
                f"Redundant link from {src_tool_label} to {dest_tool_label}"
            )
        self.links.add((src_tool_label, dest_tool_label))
        src_agent = self.tool_label2agent[src_tool_label]
        src_agent.toolList.registerTool(self.tool_label2tool[dest_tool_label])

    def allLink(self, dest_tool):
        cnt = 0
        for i in self.tool_label2tool.values():
            if i.invoke_label != dest_tool.invoke_label:
                cnt += 1
                self.link(i, dest_tool)
        # print(f"Link {cnt} edges successfully.")

    def init(self, begin_tool, task):
        begin_tool_label = begin_tool.invoke_label
        if begin_tool_label not in self.tool_label2agent.keys():
            raise ValueError(f"tool_label {begin_tool_label} not found!")
        begin_agent = self.tool_label2agent[begin_tool_label]
        self.now = begin_agent
        self.task = task
        self.current_tool_label = begin_tool_label
        self.log_history["query"] = task
        self.log_history["begin_tool"] = begin_tool.invoke_label
        self.log_history["chains"] = []

    def step(self):
        if self.isFinished:
            return
        current_agent = self.now
        current_agent.state = self.history_state  # update history
        current_agent.setRequest(self.task)  # submit task/request
        (
            prompt,
            thought,
            action,
            obs,
            reward,
            isDone,
        ) = current_agent.step()  # take next step
        self.history_state.updateState(
            thought, action, obs, rk=False
        )  # update history state
        next_tool_label = action.strip()[
            : action.strip().find("(")
        ].strip()  # split the tool_label
        if reward != -500:
            next_agent = self.tool_label2agent[next_tool_label]
        else:
            next_agent = self.now
        self.tool_info.update(self.now.toolList.info)
        self.output_TAO(
            self.current_tool_label,
            list(self.now.toolList.tool_map.keys()),
            thought,
            action,
            obs,
        )
        self.now = next_agent
        if isDone:
            self.isFinished = True

        self.current_steps += 1
        if reward != -500:
            self.current_tool_label = next_tool_label

    def steps(self, max_steps=12):
        while self.isFinished == False:
            self.step()
            if self.current_steps > max_steps:
                break

    def saveLog(self, event_id):
        self.log_history["tool_info"] = self.tool_info
        logger(event_id, self.log_history)

    def addExternalLog(self, external_log):
        self.log_history["external_log"] = external_log


EXAMPLES = """
1. Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search(Colorado orogeny)
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
2. Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup(eastern sector)
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
3. Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: Search(High Plains)
Observation: High Plains refers to one of two distinct land regions:
4. Thought: I need to instead search High Plains (United States).
Action: Search(High Plains (United States))
Observation: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]
5. Thought: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.
Action: Finish(1,800 to 7,000 ft)
Observation: Your are correct.
End of the examples.
"""
