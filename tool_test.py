from cProfile import run
from examples.bmbtools.python import RunPythonTool
from examples.bmbtools.shell import RunShellTool
from examples.bmbtools.wikipedia import WikiLookUpTool, WikiPediaDisambiguationTool, WikiPediaSearchTool
from examples.bmbtools.arxiv import SearchArxivTool
from examples.bmbtools.file_operation import ReadFileTool, WriteFileTool
from examples.bmbtools.weather import GetWeatherTool, ForcastWeatherTool
from examples.bmbtools.google_search import GoogleSearchTool
from agent.alm import ReActToolAgent
from agent.tools import ToolList
from agent.llm import GPT3_5LLM
from utils.loadenv import Env


env = Env()
write_file = WriteFileTool()
read_file = ReadFileTool()
run_python = RunPythonTool()

tool_list = ToolList()
tool_list.registerTool(write_file)
tool_list.registerTool(read_file)
tool_list.registerTool(run_python)

llm = GPT3_5LLM(env.openai_key())

react_tool_agent = ReActToolAgent(llm, tool_list)
react_tool_agent.setRequest("Tell me the 10th Fibonacci number.")
obs, _, _ = react_tool_agent.step()
print(obs)
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
