from typing import Union, Dict
import warnings
import numpy as np


class Tool:
    perf_confidence: float = 0.0

    def __init__(self, perf_confidence: float = 0.0, token_confidence: float = 0.0):
        self.invoke_label = "none"
        self.perf_confidence = perf_confidence
        self.token_confidence = token_confidence

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        raise NotImplementedError("You need to implement this tool.")

    def description(self) -> str:
        raise NotImplementedError("You need to implement this tool.")


class ToolList:
    def __init__(self, threshold: float = 0.02):
        self.tool_map = {}
        self.examples = []
        self.info = {}
        self.threshold = threshold

    def registerTool(self, tool: Tool) -> bool:
        tool_label = tool.invoke_label
        if tool_label in self.tool_map.keys():
            warnings.warn("The tool %s has already been registed." %
                          tool_label)
            return False
        self.tool_map[tool_label] = tool

    def addExampleFromFile(self, file_path):
        file = open(file_path, "r")
        self.examples.append(file.read())
        file.close()

    def addExample(self, example: str):
        self.examples.append(example)

    def toolListWithConf(self):
        tools = [tool for _, tool in self.tool_map.items()]
        for tool in tools:
            self.info[tool.invoke_label] = tool.perf_confidence
        confidence = [tool.perf_confidence for tool in tools]
        for i in range(len(confidence)):
            confidence[i] = np.exp(
                confidence[i]) if confidence[i] < 0 else confidence[i] + 1
        confidence = confidence / np.sum(confidence)
        for i in range(0, len(tools)):
            tools[i].conf = confidence[i]
        tools = filter(lambda x: x.conf >
                       self.threshold or x.invoke_label == "Answer", tools)
        tools = sorted(tools, key=lambda x: x.conf, reverse=True)
        return tools

    def description(self, use_examples=True, use_conf=False):
        tools = self.toolListWithConf()

        if self.num() == 0:
            warnings.warn(
                "This toollist has not registed any tool yet or all of tools are low-confident."
            )
        if len(tools) == 1:
            res = "There is one action or tool you can use:\n"
        else:
            res = f"There are {len(tools)} actions or tools you can use.\n"
            if use_conf:
                res += "For each tool, a score of confidence in [0, 1] will be provided. The tool with higher confidence may have better performance and it is recommended to invoke it.\n"
        cnt = 1
        for tool in tools:
            if use_conf:
                res += "%d. [Confidence: %.2f] %s\n" % (
                    cnt, tool.conf, tool.description())
            else:
                res += "%d. %s\n" % (cnt, tool.description())
            cnt += 1
        if use_examples:
            res += "Here are some examples.\n"
            for e in self.examples:
                res += e + "\n"
            res += "\n"
        return res

    def num(self):
        return len(self.tool_map)

    def invoke(self, invoke_cmd) -> Union[str, float, bool]:
        try:
            invoke_id = invoke_cmd[0: invoke_cmd.find("(")].strip()
            args = invoke_cmd[invoke_cmd.find("(") + 1: invoke_cmd.rfind(")")]
            if args.startswith("'") or args.startswith("\""):
                args = args[1: -1]
            obs, reward, isDone, info = self.tool_map[invoke_id].invoke(args)
            self.info.update(info)
            return obs, reward, isDone
        except KeyError:
            return (
                f'"{invoke_id}" not found. You can only invoke the tools we provide to you: {", ".join(self.tool_map.keys())}',
                -500,
                False,
            )
        except:
            return "Invalid invoke_cmd %s!" % invoke_cmd, -0.2, False

    def toolInfo(self):
        return self.info


def argsParser(arg_str):
    args = []
    tmp = ""
    status = 0
    for i in arg_str:
        if status == 0:
            if i == ",":
                args.append(tmp)
            elif i == '"':
                status = 1
                tmp = ""
        else:
            if i == '"':
                status -= 1
            else:
                tmp += i
    if tmp != "":
        args.append(tmp)
    return args
