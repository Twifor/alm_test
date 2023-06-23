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

    def weather_key(self):
        weather_key = os.getenv("WEATHER_API_KEY")
        if not weather_key:
            raise ValueError("Environment variables are missing.")
        return weather_key

    def searper_key(self):
        searper_key = os.getenv("SERPER_API_KEY")
        if not searper_key:
            raise ValueError("Environment variables are missing.")
        return searper_key
