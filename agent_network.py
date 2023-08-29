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
from examples.bmbtools.google_search import GoogleSearchTool, GoogleSearch2Tool
from examples.bmbtools.code_interpreter import ExecuteCodeTool
from examples.bmbtools.gradio import ImageCaptionTool, ImageToPromptTool, OCRTool
from examples.scienceQA.read_lecture import ReadLectureTool
from examples.scienceQA.rubbish_tools import *
from agent.agent_network import ReActToolAgent, AgentNetWork
from agent.llm import GPT3_5LLM, Davinci003LLM
from utils.loadenv import Env
import openai
import json

env = Env()
# openai.api_key = "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"
llm = GPT3_5LLM(env.openai_key())
llm_judge = GPT3_5LLM(env.openai_key())
tools = [
    BeginTool(),
    AnswerTool(lambda x: True),
    WikiPediaSearchTool(),
    WikiLookUpTool(),
    WikiPediaDisambiguationTool(),
    GoogleSearch2Tool(),
    OCRTool(),
    # ImageCaptionTool(),
    GoogleSearchTool(env.searper_key()),
    ExecuteCodeTool(),
    RunPythonTool(),
    R_OCRTool(),
    R_SearchTool(),
    R_UnknownTool(),
    R_LoopUpTool(),
    # ReadLectureTool(""),
    R_CalculatorTool(),
    R_ExecuteCodeTool(),
]
agents = [ReActToolAgent(llm, tool) for tool in tools]
network = AgentNetWork(llm_judge)
for agent in agents:
    network.addToolAgent(agent)
# network.recover_edges()
for i in range(1, len(tools)):
    network.link(tools[0], tools[i])
# network.allLink(tools[1])
for i in range(1, len(tools)):
    for j in range(1, len(tools)):
        if i != j:
            network.link(tools[i], tools[j])
network.recover_tool_score()

i = 173
while i < 1000:
    try:
        react_agent = AgentNetWork(llm)
        f = open(f"./dataset/math/{i}.json", "r")
        d = json.loads(f.read())
        query = d["problem"] + "\n"
        query += "Use Answer tool to submit your final answer. The answer should be an integer instead of an expression or a variable.\n"

        def f(x):
            return EM(x, d["answer"])

        ans = d["answer"]
        tools[1].func = lambda x: EM(x, ans)
        tools[8].reset()
        tools[9].reset()
        network.init(tools[0], query)
        llm.tokens = 0
        network.steps(max_steps=8)
        network.addExternalLog({"ground_truth": ans, "token_use": llm.tokens})
        import os

        os.system(f"del ./logs/agentNet_math_{i}.log")
        network.saveLog(f"agentNet_math_{i}")

        i += 1
    except Exception as e:
        print(e)
        print("retry")


# file = open("dataset/tableMWP/problems_train.json", "r")
# obj:dict = json.loads(file.read())
# data = []
# for i in obj.values():
#     data.append(i)
# for i in range(98, 100):
#     d = data[i]
#     query = d["question"] + "\n"
#     query += "You need to read from this table to generate your answer:"
#     query += d["table"]
#     if d["ans_type"].endswith("number"):
#         f = lambda x: abs(eval(x)-eval(d["answer"].replace(',',''))) < 0.001
#     else:
#         f = lambda x: EM(x, d["answer"])
#     solve(query, f, d["answer"], None, i)
