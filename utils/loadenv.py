import os
from dotenv import load_dotenv


class Env:
    def __init__(self):
        load_dotenv()

    def openai_key(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Environment variables are missing.")
        return api_key
