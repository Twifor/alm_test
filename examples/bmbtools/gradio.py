from agent.tools import Tool
from typing import Optional, Type, AnyStr, Any, Union, Dict
from gradio_tools.tools import BarkTextToSpeechTool, StableDiffusionTool, DocQueryDocumentAnsweringTool, ImageCaptioningTool, StableDiffusionPromptGeneratorTool, TextToVideoTool, ImageToMusicTool, WhisperAudioTranscriptionTool, ClipInterrogatorTool
import easyocr

class ImageCaptionTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "ImageCaption"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        path_to_image = invoke_data
        ans = ImageCaptioningTool().run(f"{path_to_image}")
        return ans, 0, False, {}

    def description(self) -> str:
        return "ImageCaption(path_to_image), generating a caption for an image, the path of which is path_to_image."

class ImageToPromptTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "ImageToPrompt"

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        path_to_image = invoke_data
        ans = ClipInterrogatorTool().run(path_to_image)
        return ans, 0, False, {}

    def description(self) -> str:
        return "ImageToPrompt(path_to_image), creating a prompt for StableDiffusion that matches the input image, the path of which is path_to_image."
    
class OCRTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "OCR"
        self.reader = easyocr.Reader(['en'])
    
    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        path_to_image = invoke_data
        result = self.reader.readtext(path_to_image)
        final_ans = ""
        if len(result) == 0:
            final_ans = "Cannot extract any words from this image."
        else:
            for i in result:
                pos = i[0]
                words = i[1]
                cfd = i[2]
                final_ans += "%s at %s with confidence = %.0f%%.\n" % (words, str(pos), float(cfd)*100)
        return final_ans, 0, False, {}

    def description(self) -> str:
        return "OCR(path_to_image), extract some words from a given image, the path of which is path_to_image."
      