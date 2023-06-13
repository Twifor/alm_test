from examples.hotpotQA.hotpotqa_tools import *
from agent.tools import ToolList
from agent.alm import ReActAgent, ReActReflexionAgent
from utils.loadenv import Env
from agent.llm import GPT3_5LLM
import time

query = "Sadok Sassi played for a national team that made its first World Cup in what year?"
gt_answer = "1978"

hotpotQA_tool_list = ToolList()

wiliEnv = WikiEnv(gt_answer)
hotpotQA_tool_list.registerTool(SearchTool(wiliEnv))
hotpotQA_tool_list.registerTool(LookupTool(wiliEnv))
hotpotQA_tool_list.registerTool(FinishTool(wiliEnv))
hotpotQA_tool_list.addExampleFromFile(
    "./examples/hotpotQA/hotpotqa_tool_examples.txt")
env = Env()
llm = GPT3_5LLM(env.openai_key())

react_agent = ReActReflexionAgent(llm, llm, hotpotQA_tool_list)
react_agent.readReflecionExampleFromFile(
    "./examples/hotpotQA/hotpotqa_reflection_examples.txt")
react_agent.setRequest(query)
while not react_agent.isFinished():
    reward = react_agent.step()
    time.sleep(2)
react_agent.saveLog("test0")
