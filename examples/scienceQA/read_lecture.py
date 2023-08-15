from agent.tools import Tool
from typing import Optional, Type, AnyStr, Any, Union, Dict
from gradio_tools.tools import BarkTextToSpeechTool, StableDiffusionTool, DocQueryDocumentAnsweringTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool, ImageToMusicTool, WhisperAudioTranscriptionTool, ClipInterrogatorTool
import easyocr

class ReadLectureTool(Tool):
    def __init__(self, knowledge: str):
        super().__init__()
        self.invoke_label = "ReadLecture"
        self.knowledge = knowledge

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        tmp = self.knowledge
        self.knowledge = "You can only invoke it once."
        return tmp, 0, False, {}

    def description(self) -> str:
        return "ReadLecture(), read from lecture to acquire some related knowledge."
