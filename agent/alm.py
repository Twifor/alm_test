from agent.state_memory import ReActRawHistoryState
from agent.tools import ToolList, Tool
from agent.llm import GPT3_5LLM
from examples.bmbtools.answer import AnswerTool
from utils.logger import logger
import warnings
from agent.prompts import *


class ReActAgent:
    def __init__(self, llm: GPT3_5LLM,  toolList: ToolList = None):
        self.llm = llm
        self.state_memory = ReActRawHistoryState()
        self.toolList = toolList
        if self.toolList == None:
            self.toolList = ToolList()
        self.current_steps = 0
        self.__is_correct = False
        self.__isFinished = False
        self.request = ""

    def output_TAO(self, thought, action, observation):
        print(f"At step \033[31m{self.current_steps + 1}\033[0m:")
        print(f"\033[32mThought\033[0m: {thought.strip()}")
        print(f"\033[33mAction\033[0m: {action}")
        print(f"\033[34mObservation\033[0m: {observation}")
        print("=="*20)

    def setRequest(self, request):
        self.request = request

    def registerTool(self, tool: Tool) -> bool:
        return self.toolList.registerTool(tool)

    def isFinished(self) -> bool:
        return self.__isFinished

    def isCorrect(self) -> bool:
        return self.__is_correct

    def step(self, is_output=True):
        if self.isFinished() or self.isCorrect():
            return self.isCorrect()
        prompt = REACT_PROMPT.format(prompt=REACT_INSTRUCTION,
                                     tool_description=self.toolList.description(
                                         use_examples=False),
                                     task=self.request, history=self.state_memory.description(),
                                     examples=REACT_EXAMPLES)
        if self.request == "":
            warnings.warn("Request is empty.")
        llm_response = self.llm.response(
            prompt, stop=f"\nObservation:")
        try:
            thought, action = llm_response.strip().split(f"Action:")
        except:
            warnings.warn("LLM fail recover.")
            action = self.llm.response(
                prompt + llm_response + "Action: ", stop=f"\nObservation:"
            )
            thought = llm_response
        obs, reward, isDone = self.toolList.invoke(action)
        if is_output:
            self.output_TAO(thought, action, obs)
        self.state_memory.updateState(thought, action, obs)
        self.current_steps += 1
        if isDone:
            self.__is_correct = reward == 1
            self.__isFinished = True
        return reward

    def steps(self, max_steps):
        while self.__isFinished == False:
            self.step()
            if self.current_steps > max_steps:
                self.__is_correct = False
                self.__isFinished = True
                break

    def saveLog(self, event_id, external_info={}):
        if not self.isFinished():
            raise ValueError(
                "You can save log only if this agent has finished.")
        obj = {}
        obj["question"] = self.request
        obj["chains"] = self.state_memory.steps
        obj["external_log"] = external_info
        logger(event_id, obj)


class ReActReflexionAgent:
    def __init__(self, llm: GPT3_5LLM, reflexion_llm: GPT3_5LLM, toolList: ToolList = None):
        self.reflexion_llm = reflexion_llm
        self.llm = llm

        self.current_trials = 0
        self.current_steps = 0

        self.__isPaused = False
        self.__isCorrect = False
        self.__isFinished = False

        self.reflecionExample = REFLEXION_EXAMPLE
        self.reflections = []
        self.trial_history = []
        self.state_memory = ReActRawHistoryState()
        self.toolList = toolList
        if self.toolList == None:
            self.toolList = ToolList()

        self.request = ""

    def isPaused(self):
        return self.__isPaused

    def __generateReflectPrompt(self):
        if self.reflecionExample == "":
            warnings.warn("Reflecion Example is empty.")
        reflect_prompt_head = REFLECT_INSTRUCTION
        reflection_prompt = reflect_prompt_head + "\n" + self.reflecionExample
        reflection_prompt += "\n<END of EXAMPLES>\nPrevious trial:\n\n"
        reflection_prompt += f"Task: {self.request}\n"
        reflection_prompt += self.state_memory.description() + "\nReflection:\n"
        return reflection_prompt

    def __reflecion(self):
        reflect_prompt = self.__generateReflectPrompt()
        llm_response = self.reflexion_llm.response(
            reflect_prompt, stop="\n")
        self.reflections.append(llm_response)

    def readReflecionExampleFromFile(self, file_path):
        file = open(file_path, "r", encoding='utf-8')
        self.reflecionExample = file.read()
        file.close()

    def __reflection_description(self):
        if len(self.reflections) == 0:
            return ""
        header = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n"
        return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in self.reflections]) + "\n"

    def __output_reflecion(self):
        print(
            f"At trial \033[31m{self.current_trials + 1}\033[0m, the reflections are:\n")
        for f in self.reflections:
            print(f"- \033[33m{f}\033[0m\n")

    def step(self, is_output=True, max_trials=3, max_steps=8):
        if self.isFinished():
            return self.isCorrect()
        if self.isPaused():
            if self.current_trials > max_trials:
                self.__isFinished = True
                return 0
            self.trial_history.append(
                {"reflections": self.reflections.copy(), "chains": self.state_memory.steps.copy()})
            self.__reflecion()
            if is_output:
                self.__output_reflecion()
            self.__isPaused = False
            self.state_memory.reset()
            self.current_trials += 1
            self.current_steps = 0
        prompt = REFLEXION_PROMPT.format(prompt=REACT_INSTRUCTION,
                                         tool_description=self.toolList.description(
                                             use_examples=False),
                                         task=self.request, history=self.state_memory.description(),
                                         examples=REACT_EXAMPLES,
                                         reflexion_prompt=self.__reflection_description())
        llm_response = self.llm.response(
            prompt, stop=f"\nObservation:")
        try:
            thought, action = llm_response.strip().split(f"Action:")
        except:
            warnings.warn("LLM fail recover.")
            action = self.llm.response(
                prompt + llm_response + "Action: ", stop=f"\nObservation:"
            )
            thought = llm_response
        obs, reward, isDone = self.toolList.invoke(action)
        if is_output:
            self.output_TAO(thought, action, obs)
        self.state_memory.updateState(thought, action, obs)
        self.current_steps += 1
        if isDone:
            self.__isCorrect = reward == 1
            self.__isPaused = True
            if self.isCorrect():
                self.__isFinished = True
        if self.current_steps > max_steps:
            self.__isCorrect = False
            self.__isPaused = True
            return 0
        return reward

    def steps(self, max_trials, max_steps):
        while self.isFinished() == False:
            self.step(max_trials=max_trials, max_steps=max_steps)

    def output_TAO(self, thought, action, observation):
        print(f"At step \033[31m{self.current_steps + 1}\033[0m:")
        print(f"\033[32mThought\033[0m: {thought.strip()}")
        print(f"\033[33mAction\033[0m: {action}")
        print(f"\033[34mObservation\033[0m: {observation}")
        print("=="*20)

    def saveLog(self, event_id, external_info={}):
        if not self.isFinished():
            raise ValueError(
                "You can save log only if this agent has finished.")
        obj = {}
        obj["question"] = self.request
        obj["trial_history"] = self.trial_history
        obj["last_trial"] = {"reflections": self.reflections,
                             "chains": self.state_memory.steps}
        for k, v in external_info.items():
            obj[k] = v
        logger(event_id, obj)

    def isFinished(self) -> bool:
        return self.__isFinished

    def isCorrect(self) -> bool:
        return self.__isCorrect

    def setRequest(self, request):
        self.request = request

    def registerTool(self, tool: Tool) -> bool:
        return self.toolList.registerTool(tool)


class CoTAgent:
    def __init__(self, llm: GPT3_5LLM, answer_tool):
        self.llm = llm
        self.toolList = ToolList()
        self.toolList.registerTool(answer_tool)
        self.request = ""
        self.state_memory = ReActRawHistoryState()
        self.__isFinished = False
        self.__is_correct = False

    def output_TAO(self, thought, action, observation):
        print(f"\033[32mThought\033[0m: {thought.strip()}")
        print(f"\033[33mAction\033[0m: {action}")
        print(f"\033[34mObservation\033[0m: {observation}")
        print("=="*20)

    def setRequest(self, request):
        self.request = request

    def setContext(self, context):
        self.context = context

    def isFinished(self) -> bool:
        return self.__isFinished

    def isCorrect(self) -> bool:
        return self.__is_correct

    def step(self, is_output=True):
        if self.isFinished() or self.isCorrect():
            return self.isCorrect()
        prompt = COT_PROMPT.format(
            examples=COT_EXAMPLES, context=self.context, task=self.request)

        if self.request == "":
            warnings.warn("Request is empty.")
        llm_response = self.llm.response(
            prompt, stop=f"\n:")
        try:
            thought, action = llm_response.strip().split(f"Action:")
        except:
            warnings.warn("LLM fail recover.")
            # print(prompt + llm_response + "Action: ")
            action = self.llm.response(
                prompt + llm_response + "\nAction: ", stop=f"\n:"
            )
            thought = llm_response
        obs, reward, isDone = self.toolList.invoke(action)
        isDone = True
        self.state_memory.updateState(thought, action, obs)
        if is_output:
            self.output_TAO(thought, action, obs)
        if isDone:
            self.__is_correct = reward == 1
            self.__isFinished = True
        return reward

    def steps(self):
        while not self.isFinished():
            self.step()

    def saveLog(self, event_id, external_info={}):
        if not self.isFinished():
            raise ValueError(
                "You can save log only if this agent has finished.")
        obj = {}
        obj["question"] = self.request
        obj["chains"] = self.state_memory.steps
        for k, v in self.toolList.toolInfo().items():
            obj[k] = v
        obj["external_log"] = external_info
        logger(event_id, obj)
