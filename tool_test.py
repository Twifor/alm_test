from cProfile import run
from examples.bmbtools.python import RunPythonTool
# from examples.bmbtools.shell import RunShellTool
from examples.bmbtools.wikipedia import WikiLookUpTool, WikiPediaDisambiguationTool, WikiPediaSearchTool
from examples.bmbtools.arxiv import SearchArxivTool
from examples.bmbtools.file_operation import ReadFileTool, WriteFileTool
from examples.bmbtools.weather import GetWeatherTool, ForcastWeatherTool
from examples.bmbtools.google_search import GoogleSearchTool
from examples.bmbtools.douban import ComingOutFilterTool, NowPlayingOutFilterTool, PrintDetailTool
from examples.bmbtools.chemical import GetIdTool, GetPropTool, GetIdByStructTool, GetAllNameTool, GetNameTool
from examples.bmbtools.tutorial import TutorialTool
from examples.bmbtools.code_interpreter import ExecuteCodeTool
from examples.bmbtools.wolframalpha import GetWolframAlphaResultsTool
from examples.bmbtools.bing_map import SearchNearbyTool, GetCoordinatesTool, GetRouteTool, GetDistanceTool
from examples.bmbtools.answer import AnswerTool
from examples.bmbtools.math import RoundTool
from agent.agent_network import ReActToolAgent, AgentNetWork
from agent.tools import ToolList
from agent.llm import GPT3_5LLM
from utils.loadenv import Env
from agent.state_memory import ReActRawHistoryState
import math

env = Env()
llm = GPT3_5LLM(env.openai_key())

read_file_tool = ReadFileTool()
distance_tool = GetDistanceTool(env.bing_map_key())
round_tool = RoundTool()
python_tool = ExecuteCodeTool()
answer_tool = AnswerTool(lambda x: abs(float(x) - 1347.0) < 1)

agent1 = ReActToolAgent(llm, read_file_tool)
agent2 = ReActToolAgent(llm, distance_tool)
agent3 = ReActToolAgent(llm, python_tool)
agent4 = ReActToolAgent(llm, answer_tool)
agent5 = ReActToolAgent(llm, round_tool)

network = AgentNetWork()
network.addToolAgent(agent1)
network.addToolAgent(agent2)
network.addToolAgent(agent3)
network.addToolAgent(agent4)
network.addToolAgent(agent5)

network.link(read_file_tool.invoke_label, distance_tool.invoke_label)
network.link(distance_tool.invoke_label, round_tool.invoke_label)
network.link(round_tool.invoke_label, python_tool.invoke_label)
network.link(python_tool.invoke_label, answer_tool.invoke_label)

network.init(read_file_tool.invoke_label,
             "Read two locations from \"location.txt\" and get the distance in kilometers between these two locations. The round this number into an integer x. Finally tell me the the result of sqrt(x)*(x+1)/2 and submit your result.")

network.steps()
# llm = GPT3_5LLM(env.openai_key())
# react_tool_agent = ReActToolAgent(llm, tool, history)
# react_tool_agent.setRequest(
#     "How to travel from Tianjin to Beijing?")
# network = AgentNetWork()
# network.addToolAgent(react_tool_agent)
# _, _, _, obs, _, _ = react_tool_agent.step()
# print(obs)
# env = Env()
# tool = TutorialTool(env.openai_key())
# print(tool.invoke("Clean up the dormitory."))

# tool = GetNameTool()
# print(tool.invoke("967"))

# tool = PrintDetailTool()
# print(tool.invoke("suzume no tojimari")[0])


# env = Env()
# write_file = WriteFileTool()
# read_file = ReadFileTool()
# run_python = RunPythonTool()

# tool_list = ToolList()
# tool_list.registerTool(write_file)
# tool_list.registerTool(read_file)
# tool_list.registerTool(run_python)

# llm = GPT3_5LLM(env.openai_key())

# react_tool_agent = ReActToolAgent(llm, tool_list)
# react_tool_agent.setRequest("Tell me the 10th Fibonacci number.")
# obs, _, _ = react_tool_agent.step()
# print(obs)
# google_search = GoogleSearchTool(env.searper_key())
# print(google_search.invoke("Tianjin Univerisity"))
# weather_tool = GetWeatherTool(env.weather_key())
# forecast_tool = ForcastWeatherTool(env.weather_key())
# print(weather_tool.invoke("Tianjin"))
# print(forecast_tool.invoke("Tianjin, 3"))
# read_file = ReadFileTool()
# write_file = WriteFileTool()
# print(read_file.invoke(".env"))
# print(write_file.invoke("./sls, slslslsslsls"))

# arxiv = SearchArxivTool()
# print(arxiv.invoke(
#     "Attention, Compilation, and Solver-based Symbolic Analysis are All You Need"))

# run_python = RunPython()
# print(run_python.invoke(
#     '''
# print(1+2+3)
# '''
# ))

# run_shell = RunShellTool()
# print(run_shell.invoke("ls -al"))

# search = WikiPediaSearchTool()
# print(search.invoke("blue archive"))
# lookup = WikiLookUpTool()
# print(lookup.invoke("China"))
# disambiguation = WikiPediaDisambiguationTool()
# print(disambiguation.invoke("Blue Archive"))
