from agent.tools import Tool
from typing import Optional, Type, AnyStr, Any, Union, Dict
from gradio_tools.tools import BarkTextToSpeechTool, StableDiffusionTool, DocQueryDocumentAnsweringTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool, ImageToMusicTool, WhisperAudioTranscriptionTool, ClipInterrogatorTool
import easyocr
import random

class R_SearchTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "Search"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return "Nothing Found.", 0, False, {}

    def description(self) -> str:
        return "Search(query), search for something."

class R_OCRTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "OCR2"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return "XXX at [?] with confidence = ??", 0, False, {}

    def description(self) -> str:
        return "OCR2(path_to_image), extract some words from given image."

class R_UnknownTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "SomeTool"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return "hasDFJN;IOadnhiNHLNLnlkjanfl", 0, False, {}

    def description(self) -> str:
        return "SomeTool(), some tool." 
    
class R_LoopUpTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "LoopUp"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return f"{invoke_data} not found", 0, False, {}

    def description(self) -> str:
        return "LoopUp(query), look up for some information." 
    
class R_ExecuteCodeTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "ExeCode"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return random.randint(-1000, 1000), 0, False, {}

    def description(self) -> str:
        return "ExeCode(code), execute Python expressions with Python Interpreter, can be used as a simple calculator e.g.,\"(123 + 234) / 23 * 19.\"." 
    
class R_CalculatorTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "Calculate"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        return random.randint(-1000, 1000), 0, False, {}

    def description(self) -> str:
        return "Calculate(expression), calculate mathmatical expresstion and return the final result." 