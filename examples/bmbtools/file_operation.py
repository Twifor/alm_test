from pathlib import Path
from agent.tools import Tool
from typing import Union, Dict


class WriteFileTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "WriteFile"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        spliter = invoke_data.find(",")
        file_path = invoke_data[:spliter]
        text = invoke_data[spliter+1:]
        write_path = (
            Path(file_path)
        )
        try:
            write_path.parent.mkdir(exist_ok=True, parents=False)
            with write_path.open("w", encoding="utf-8") as f:
                f.write(text)
            return f"File written successfully to {file_path}.", 0, False, {}
        except Exception as e:
            return "Error: " + str(e), 0, False, {}

    def description(self) -> str:
        return "WriteFile(file_path, text), Write file to disk."


class ReadFileTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "ReadFile"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        file_path = invoke_data
        read_path = (
            Path(file_path)
        )
        try:
            with read_path.open("r", encoding="utf-8") as f:
                content = f.read()
            return content, 0, False, {}
        except Exception as e:
            return "Error: " + str(e), 0, False, {}

    def description(self) -> str:
        return "ReadFile(file_path), Read file from disk."
