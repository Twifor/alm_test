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
    ReadLectureTool(""),
]
agents = [ReActToolAgent(llm, tool) for tool in tools]
network = AgentNetWork(llm_judge)
for agent in agents:
    network.addToolAgent(agent)
# network.recover_edges()
for i in range(2, len(tools)):
    network.link(tools[0], tools[i])
network.allLink(tools[1])
for i in range(2, len(tools)):
    for j in range(2, len(tools)):
        if i != j:
            network.link(tools[i], tools[j])
# network.recover_tool_score()

for i in range(0, 100):
    file = open(f"dataset/scienceQA/train/{i}.json", "r")
    obj = json.loads(file.read())
    query = (
        obj["question"] + " You must choose one answer from the following choices:\n"
    )
    query += "Choices: " + str(obj["choices"]) + "\n"
    if "image" in obj.keys():
        query += "This question has a related image, you can use some tools to read from this image to help you to solve this problem.\n"
        query += f"The path of this image: dataset/scienceQA/train/{i}.jpg.\n"
    ans = obj["choices"][obj["answer"]]
    lecture = obj["lecture"]

    tools[1].func = lambda x: EM(x, ans)
    tools[-1].knowledge = lecture

    network.init(tools[0], query)
    llm.tokens = 0
    network.steps(max_steps=8)
    network.addExternalLog({"ground_truth": ans, "token_use": llm.tokens})
    network.saveLog(f"sciQA_{i}")

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
