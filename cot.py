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
for i in range(0, 100):

    file = open(f"dataset/scienceQA/train/{i}.json", "r")
    obj = json.loads(file.read())
    query = (
        obj["question"] + " You must choose one answer from the following choices:\n"
    )
    query += "Choices: " + str(obj["choices"]) + "\n"
    ans = obj["choices"][obj["answer"]]
    lecture = obj["lecture"]
    cot_agent = CoTAgent(llm, AnswerTool(lambda x: EM(x, ans)))
    cot_agent.setRequest(query)
    cot_agent.setContext(lecture)
    llm.tokens = 0
    cot_agent.steps()
    cot_agent.saveLog(
        f"cot_sciQA_{i}", {"ground_truth": ans, "token_use": llm.tokens})
