from tkinter import E
from turtle import begin_fill
from examples.bmbtools.douban import (
    PrintDetailTool,
    NowPlayingOutFilterTool,
    ComingOutFilterTool,
)
from examples.hotpotQA.hotpotqa_tools import EM
import json
import random
from examples.bmbtools.weather import (
    ForcastWeatherTool,
    GetWeatherTool,
)
from examples.bmbtools.file_operation import WriteFileTool, ReadFileTool
from examples.bmbtools.wikipedia import (
    WikiPediaSearchTool,
    WikiLookUpTool,
    WikiPediaDisambiguationTool,
)
from examples.bmbtools.answer import AnswerTool, BeginTool
from examples.bmbtools.python import RunPythonTool
from examples.bmbtools.google_search import GoogleSearchTool
from examples.bmbtools.code_interpreter import ExecuteCodeTool
from agent.agent_network import ReActToolAgent, AgentNetWork
from agent.llm import GPT3_5LLM, Davinci003LLM
from utils.loadenv import Env


env = Env()
llm = GPT3_5LLM(env.openai_key())

google_search_tool = GoogleSearchTool(env.searper_key())
execute_python_tool = ExecuteCodeTool()
answer_tool = AnswerTool(lambda x: int(x) == 55, 55)
begin_tool = BeginTool()

agent0 = ReActToolAgent(llm, begin_tool)
agent1 = ReActToolAgent(llm, google_search_tool)
agent2 = ReActToolAgent(llm, execute_python_tool)
agent3 = ReActToolAgent(llm, answer_tool)

network = AgentNetWork(llm)
network.addToolAgent(agent0)
network.addToolAgent(agent1)
network.addToolAgent(agent2)
network.addToolAgent(agent3)
 
# network.link(begin_tool, google_search_tool)
network.link(begin_tool, execute_python_tool)
network.link(execute_python_tool, answer_tool)

# network.init(begin_tool, "Define x = difference of ages between Obama and Biden. Calculate sqrt(x) and round it to an integer and the result is defined by y. Finally calaulate y^2 Fibonacci number.")
network.init(begin_tool, "Tell me 10th Fibonacci number.")
network.steps()