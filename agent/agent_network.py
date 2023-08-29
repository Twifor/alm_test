from agent.state_memory import *
from agent.tools import ToolList, Tool
from agent.llm import GPT3_5LLM
from utils.logger import logger
import numpy as np
from operator import itemgetter
import json
from agent.prompts import *
import warnings


class ReActToolAgent:
    def __init__(
        self,
        llm: GPT3_5LLM,
        tool: Tool,
        state: ReActBaseHistoryState = None,
    ):
        self.llm = llm
        self.toolList = ToolList()
        self.toolList.registerTool(tool)
        self.request = ""
        self.state = state
        if self.state == None:
            self.state = ReActRawHistoryState()

    def setRequest(self, request):
        self.request = request

    def registerTool(self, tool: Tool) -> bool:
        return self.toolList.registerTool(tool)

    def step(self, is_last_trial):
        external_prompt = ""
        prompt = AGENT_NETWORK_PROMPT.format(
            prompt=REACT_INSTRUCTION,
            examples=REACT_EXAMPLES,
            tool_description=self.toolList.description(
                use_examples=False, use_conf=True
            ),
            task=self.request,
            history=self.state.description(),
            external_prompt=external_prompt,
        )
        if is_last_trial:
            prompt = REACT_LAST_TRIAL.format(
                task=self.request,
                history=self.state.description(),
                format=REACT_LAST_TRIAL_FORMAT,
            )
        if self.request == "":
            warnings.warn("Request is empty.")
        llm_response = self.llm.response(prompt, stop="END")
        try:
            llm_response = llm_response.replace("\\", "\\\\")
            obj = json.loads(llm_response.strip())
            action = str(obj["Action"])
            parameter = str(obj["Parameter"])
            thought = str(obj["Thought"])
        except:
            print("xxxxxxxxxxxxx")
            print(llm_response)
            print("xxxxxxxxxxxxx")
            action = "Please output a string of JSON."
            parameter = "Please output a string of JSON."
            thought = "Please output a string of JSON."
        obs, reward, isDone = self.toolList.invoke(action, parameter)
        obs = str(obs)
        return prompt, thought, action, parameter, obs, reward, isDone


class AgentNetWork:
    def __init__(self, llm: LLM = None):
        self.tool_label2tool = {}
        self.tool_label2agent = {}
        self.links = set()
        self.llm = llm
        self.clear()

    def clear(self):
        self.history_state = ReActRawHistoryState()
        self.now = None
        self.current_tool_label = ""
        self.task = ""
        self.isFinished = False
        self.current_steps = 0
        self.log_history = {}
        self.tool_info = {}

    def output_TAO(self, now_tool_label, next_tools, thought, action, observation):
        next_tools = [
            (tool.invoke_label, float("%.3f" % tool.conf)) for tool in next_tools
        ]
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

    def link_tool_label(self, src_tool_label: str, dest_tool_label: str):
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

    def link(self, src_tool: Tool, dest_tool: Tool):
        self.link_tool_label(src_tool.invoke_label, dest_tool.invoke_label)

    def allLink(self, dest_tool):
        cnt = 0
        for i in self.tool_label2tool.values():
            if i.invoke_label != dest_tool.invoke_label:
                cnt += 1
                self.link(i, dest_tool)
        # print(f"Link {cnt} edges successfully.")

    def init(self, begin_tool, task):
        self.clear()
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
        self.tool_chain = []

    def __with_llm_reward(self, reward):
        print("Updating confidence...")
        tool_set = set()
        tool_prompt = ""
        for tool in self.tool_chain:
            tool_set.add(tool)
        idx = 1
        for tool in tool_set:
            tool_prompt += f"{idx}. {tool.description()}\n"
            idx += 1
        prompt = AGENT_NETWORK_REWARD_PROMPT.format(
            tool_description=tool_prompt,
            task=self.task,
            history=self.history_state.description(),
        )
        response = self.llm.response(prompt, stop="END")
        tool_scores = response.split("Tool: ")
        tool_score_delta = {}
        for t_s in tool_scores[1:]:
            try:
                tool_name, s_thought = t_s.split("Score: ")
                score, thought = s_thought.split("Thought: ")
                print(thought)
                tool_name = tool_name.strip()
                if tool_name.find("(") != -1:
                    tool_name = tool_name[: tool_name.find("(")]
                score = int(score.strip())
                tool_score_delta[tool_name] = score
            except:
                continue
        for tool_name in tool_score_delta.keys():
            delta = tool_score_delta[tool_name] * 0.456
            if tool_name not in [tool.invoke_label for tool in tool_set]:
                continue
            try:
                self.tool_label2tool[tool_name].perf_confidence += delta

                if delta < 0:
                    print(f"\033[31m{tool_name} -= {-delta}\033[0m")
                elif delta > 0:
                    print(f"\033[32m{tool_name} += {delta}\033[0m")
            except:
                print(f"Error. Omit {tool_name}")

    def backward(self, reward):
        self.__with_llm_reward(reward)
        # self.__update_conf(reward)
        tool_score_record = {}
        tool_score_list = []
        for tool in self.tool_label2tool.values():
            tool_score_list.append((tool.invoke_label, tool.perf_confidence))
        tool_score_list = sorted(tool_score_list, key=lambda x: x[1], reverse=True)
        for i in tool_score_list:
            tool_score_record[i[0]] = i[1]
        tool_score_file = open("tool_score_record.json", "w")
        tool_score_file.write(
            json.dumps(tool_score_record, indent=4, separators=(",", ":"))
        )
        tool_score_file.close()

    def recover_tool_score(self, file_name="tool_score_record.json"):
        tool_score_file = open(file_name, "r")
        tool_score_record = json.loads(tool_score_file.read())
        for tool in self.tool_label2tool.values():
            tool.perf_confidence = tool_score_record[tool.invoke_label]
        tool_score_file.close()

    def recover_edges(self, file_name="edges.json"):
        tool_edges = open(file_name, "r")
        tool_edges_record = json.loads(tool_edges.read())
        for edge in tool_edges_record:
            if edge["src"] != edge["tgt"]:
                self.link_tool_label(edge["src"], edge["tgt"])

    def step(self, is_last_trial=False):
        if self.isFinished:
            return
        current_agent = self.now
        current_agent.state = self.history_state  # update history
        current_agent.setRequest(self.task)  # submit task/request
        (prompt, thought, action, parameter, obs, reward, isDone) = current_agent.step(
            is_last_trial
        )  # take next step
        state_Res = self.history_state.updateState(
            thought, f"{action}({parameter})", obs
        )  # update history state
        if state_Res == False:
            return  # invalid
        next_tool_label = action
        if reward != -500:
            next_agent = self.tool_label2agent[next_tool_label]
        else:
            next_agent = self.now
        self.tool_info.update(self.now.toolList.info)
        self.output_TAO(
            self.current_tool_label,
            list(self.now.toolList.toolListWithConf()),
            thought,
            f"{action}({parameter})",
            obs,
        )
        self.now = next_agent
        if isDone:
            self.isFinished = True
            # self.backward(reward)

        self.current_steps += 1
        if reward > -500:
            self.current_tool_label = next_tool_label
            self.tool_chain.append(self.tool_label2tool[self.current_tool_label])

    def steps(self, max_steps=12):
        while self.isFinished == False:
            self.step(self.current_steps == max_steps)
            if self.current_steps > max_steps:
                # self.backward(-1.0)
                break

    def saveLog(self, event_id):
        self.log_history["tool_info"] = self.tool_info
        logger(event_id, self.log_history)

    def addExternalLog(self, external_log):
        self.log_history["external_log"] = external_log
