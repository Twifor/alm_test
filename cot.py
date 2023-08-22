from tkinter import E
from turtle import begin_fill
from agent.tools import ToolList
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
from agent.alm import CoTAgent
from agent.llm import GPT3_5LLM, Davinci003LLM
from utils.loadenv import Env
import openai
import json

env = Env()
llm = GPT3_5LLM(env.openai_key())
file = open("dataset/tableMWP/problems_test1k.json", "r")
obj: dict = json.loads(file.read())
data = []
for i in obj.values():
    data.append(i)
for i in range(0, 1000):
    d = data[i]
    query = d["question"] + "\n"
    query += "You need to read from this table to generate your answer:\n"
    query += d["table"] + "\n"
    query += "If the final answer is a number, you should submit a number as your final answer. For example, you need to submit \"3\" instead of \"3 minutes\".\n"
    query += "Do not submit expression as your final answer. You need to calculate the final result.\n"
    ground_truth = ""
    if d["ans_type"].endswith("number"):
        def f(x): return abs(
            float(x)-eval(d["answer"].replace(',', ''))) < 0.001
        ground_truth = eval(d["answer"].replace(',', ''))
    else:
        def f(x): return EM(x, d["answer"])
        ground_truth = d["answer"]
    cot_agent = CoTAgent(llm, AnswerTool(f))
    cot_agent.setRequest(query)
    cot_agent.setContext("")
    llm.tokens = 0
    # print(ground_truth)
    cot_agent.steps()
    cot_agent.saveLog(
        f"cot_tableMWP_{i}", {"ground_truth": ground_truth, "token_use": llm.tokens})
# for i in range(1, 1000):
#     file = open(f"dataset/scienceQA/test/{i}.json", "r")
#     obj = json.loads(file.read())
#     query = (
#         obj["question"] + " You must choose one answer from the following choices:\n"
#     )
#     query += "Choices: " + str(obj["choices"]) + "\n"
#     query += "If you find it hard to solve, you need to choose the best answer by using Answer().\n"
#     ans = obj["choices"][obj["answer"]]
#     lecture = obj["lecture"]
#     cot_agent = CoTAgent(llm, AnswerTool(lambda x: EM(x, ans)))
#     cot_agent.setRequest(query)
#     cot_agent.setContext(lecture)
#     llm.tokens = 0
#     cot_agent.steps()
#     cot_agent.saveLog(
#         f"cot_sciQA_{i}", {"ground_truth": ans, "token_use": llm.tokens})
