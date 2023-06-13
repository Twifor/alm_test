from examples.hotpotQA.hotpotqa_tools import *
from agent.tools import ToolList
from agent.alm import ReActAgent
from utils.loadenv import Env
from agent.llm import GPT3_5LLM
from utils.logger import ALMLogger
import time

query = "If You Ever Get Lonely was covered by what Lyric Street Records-affiliated band?"
gt_answer = "Love and Theft"
logger = ALMLogger("test1")
logger.gt_answer("Love and Theft")

hotpotQA_tool_list = ToolList()

wiliEnv = WikiEnv(gt_answer)
hotpotQA_tool_list.registerTool(SearchTool(wiliEnv))
hotpotQA_tool_list.registerTool(LookupTool(wiliEnv))
hotpotQA_tool_list.registerTool(FinishTool(wiliEnv))
hotpotQA_tool_list.addExampleFromFile(
    "./examples/hotpotQA/hotpotqa_tool_examples.txt")
env = Env()
llm = GPT3_5LLM(env.openai_key())

react_agent = ReActAgent(llm, logger, hotpotQA_tool_list)
react_agent.setRequest(query)
while True:
    is_done, reward = react_agent.step()
    if is_done:
        break
    time.sleep(2)

logger.finish()
#

# print(react_agent.step())
# print(llm.response("You are a smart assistant.", "Good evening!"))
