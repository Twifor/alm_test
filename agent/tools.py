from typing import Union, Dict
import warnings


class Tool:
    def __init__(self):
        self.invoke_label = "none"

    def invoke(self, invoke_data) -> Union[str, int, bool, Dict]:
        raise NotImplementedError("You need to implement this tool.")

    def description(self) -> str:
        raise NotImplementedError("You need to implement this tool.")


class ToolList:
    def __init__(self):
        self.tool_map = {}
        self.examples = []
        self.info = {}

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

    def description(self, use_examples=True):
        if self.num() == 0:
            warnings.warn("This toollist has not registed any tool yet.")
        if self.num() == 1:
            res = "There is one action or tool you can use:\n"
        else:
            res = "There are %d actions or tools you can use:\n" % self.num()
        cnt = 1
        for _, v in self.tool_map.items():
            res += "%d. %s\n" % (cnt, v.description())
            cnt += 1
        if use_examples:
            res += "Here are some examples.\n"
            for e in self.examples:
                res += e + "\n"
            res += "\n"
        return res

    def num(self):
        return len(self.tool_map)

    def invoke(self, invoke_cmd) -> Union[str, int, bool]:
        try:
            invoke_id = invoke_cmd[0:invoke_cmd.find("(")]
            args = invoke_cmd[invoke_cmd.find("(") + 1: invoke_cmd.rfind(")")]
            obs, reward, isDone, info = self.tool_map[invoke_id].invoke(args)
            self.info.update(info)
            return obs, reward, isDone
        except:
            raise ValueError("Invalid invoke_cmd %s!" % invoke_cmd)

    def toolInfo(self):
        return self.info
