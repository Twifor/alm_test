class Tool:
    def __init__(self):
        self.invoke_label = "none"

    def invoke(self, invoke_data) -> str:
        raise NotImplementedError("You need to implement this tool.")

    def description(self) -> str:
        raise NotImplementedError("You need to implement this tool.")

    def example(self) -> str:
        raise NotImplementedError("You need to implement this tool.")


class ToolList:
    def __init__(self):
        self.tool_map = {}

    def registerTool(self, tool: Tool) -> bool:
        tool_label = tool.invoke_label
        if tool_label in self.tool_map.keys():
            Warning("The tool %s has already been registed." % tool_label)
            return False
        self.tool_map[tool_label] = tool

    def num(self):
        return len(self.tool_map)

    def invoke(self, invoke_cmd):
        try:
            invoke_id = invoke_cmd[0:invoke_cmd.find("(")]
            args = invoke_cmd[invoke_cmd.find("(") + 1: invoke_cmd.find(")")]
            self.tool_map[invoke_id].invoke(args)
        except:
            raise ValueError("Invalid invoke_cmd %s!" % invoke_cmd)
