from agent.tools import Tool
from typing import Optional, Type, AnyStr, Any, Union, Dict
from gradio_tools.tools import BarkTextToSpeechTool, StableDiffusionTool, DocQueryDocumentAnsweringTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool, ImageToMusicTool, WhisperAudioTranscriptionTool, ClipInterrogatorTool
import easyocr

class PunishTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "Punish"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        raise ValueError("This tool should be processed by agent network.")

    def description(self) -> str:
        return "Punish(tool_name), if you find a tool is really noisy and unhelpful, you can invoke this tool to punish the tool you dislike." 