from agent.state_memory import ReActRawHistoryState
from agent.tools import ToolList, Tool
from agent.llm import GPT3_5LLM
from utils.logger import logger
import warnings


class ReActAgent:
    def __init__(self, llm: GPT3_5LLM,  toolList: ToolList = None, max_steps=7):
        self.llm = llm
        self.state_memory = ReActRawHistoryState()
        self.toolList = toolList
        if self.toolList == None:
            self.toolList = ToolList()
        self.PromptHead = "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation.\n"
        self.current_steps = 0
        self.max_steps = max_steps
        self._is_correct = False
        self._isFinished = False
        self.request = ""

    def output_TAO(self, thought, action, observation):
        print(f"At step \033[31m{self.current_steps + 1}\033[0m:")
        print(f"\033[32mThought\033[0m: {thought}")
        print(f"\033[33mAction\033[0m: {action}")
        print(f"\033[34mObservation\033[0m: {observation}")
        print("=="*20)

    def setRequest(self, request):
        self.request = request

    def registerTool(self, tool: Tool) -> bool:
        return self.toolList.registerTool(tool)

    def isFinished(self) -> bool:
        return self._isFinished

    def isCorrect(self) -> bool:
        return self._is_correct

    def step(self, is_output=True):
        if self.isFinished() or self.isCorrect():
            return self.isCorrect()
        prompt = self.PromptHead
        prompt += self.toolList.description()
        if self.request == "":
            warnings.warn("Request is empty.")
        prompt += f"Question: {self.request}\n"
        prompt += self.state_memory.description()
        prompt += f"Thought {self.current_steps + 1}:"
        llm_response = self.llm.response(
            prompt, stop=f"\nObservation:")
        thought, action = llm_response.strip().split(
            f"\nAction {self.current_steps + 1}: ")
        obs, reward, isDone = self.toolList.invoke(action)
        if is_output:
            self.output_TAO(thought, action, obs)
        self.state_memory.updateState(thought, action, obs)
        self.current_steps += 1
        if self.current_steps > self.max_steps:
            self._is_correct = False
            self._isFinished = True
            return 0
        if isDone:
            self._is_correct = reward == 1
            self._isFinished = True
        return reward

    def saveLog(self, event_id, external_info={}):
        if not self.isFinished():
            raise ValueError(
                "You can save log only if this agent has finished.")
        obj = {}
        obj["question"] = self.request
        obj["chains"] = self.state_memory.steps
        for k, v in self.toolList.toolInfo().items():
            obj[k] = v
        for k, v in external_info.items():
            obj[k] = v
        logger(event_id, obj)


class ReActReflexionAgent(ReActAgent):
    def __init__(self, llm: GPT3_5LLM, reflexion_llm: GPT3_5LLM, toolList: ToolList = None, max_trials=7,  max_steps=7):
        super().__init__(llm, toolList, max_steps)
        self.reflexion_llm = reflexion_llm
        self.max_trials = max_trials
        self.current_trials = 0
        self._isPaused = False
        self.reflecionExample = ""
        self.reflections = []
        self.trial_history = []

    def isPaused(self):
        return self._isPaused

    def __generateReflectPrompt(self):
        if self.reflecionExample == "":
            warnings.warn("Reflecion Example is empty.")
        reflect_prompt_head = "You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.\nHere are some examples:\n"
        reflection_prompt = reflect_prompt_head + self.reflecionExample
        reflection_prompt += "Previous trial:\n"
        reflection_prompt += f"Question: {self.request}\n"
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

    def step(self, is_output=True):
        if self.isFinished():
            return self.isCorrect()
        if self.isPaused():
            if self.current_trials > self.max_trials:
                self._isFinished = True
                return 0
            self.trial_history.append(
                {"reflections": self.reflections.copy(), "chains": self.state_memory.steps.copy()})
            self.__reflecion()
            if is_output:
                self.__output_reflecion()
            self._isPaused = False
            self.state_memory.reset()
            self.current_trials += 1
            self.current_steps = 0
        prompt = self.PromptHead
        prompt += self.toolList.description()
        if self.request == "":
            warnings.warn("Request is empty.")
        prompt += f"Question: {self.request}\n"
        prompt += self.__reflection_description()
        prompt += self.state_memory.description()
        prompt += f"Thought {self.current_steps + 1}:"
        llm_response = self.llm.response(
            prompt, stop=f"\nObservation {self.current_steps + 1}:")
        thought, action = llm_response.strip().split(
            f"\nAction {self.current_steps + 1}: ")
        obs, reward, isDone = self.toolList.invoke(action)
        if is_output:
            self.output_TAO(thought, action, obs)
        self.state_memory.updateState(thought, action, obs)
        self.current_steps += 1
        if self.current_steps > self.max_steps:
            self._is_correct = False
            self._isPaused = True
            return 0
        if isDone:
            self._is_correct = reward == 1
            self._isPaused = True
            if self.isCorrect():
                self._isFinished = True
        return reward

    def saveLog(self, event_id, external_info={}):
        if not self.isFinished():
            raise ValueError(
                "You can save log only if this agent has finished.")
        obj = {}
        obj["question"] = self.request
        obj["trial_history"] = self.trial_history
        obj["last_trial"] = {"reflections": self.reflections,
                             "chains": self.state_memory.steps}
        for k, v in self.toolList.toolInfo().items():
            obj[k] = v
        for k, v in external_info.items():
            obj[k] = v
        logger(event_id, obj)
