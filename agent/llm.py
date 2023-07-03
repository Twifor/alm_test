import openai
from openai.error import RateLimitError, ServiceUnavailableError
import time
import warnings
# from retry import retry


class LLM:
    def __init__(self):
        pass

    def response(self, prompt, stop):
        raise NotImplementedError("You need to implement your LLM.")


class Davinci003LLM(LLM):
    def __init__(
        self,
        openai_key,
        temperature=0,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        super().__init__()
        openai.api_key = openai_key
        self.model = "text-davinci-003"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.tokens = 0

    # @retry(Exception, tries=3, delay=1)
    def response(self, prompt, stop="\n"):
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=stop,
        )
        self.tokens += response["usage"]["total_tokens"]
        return response["choices"][0]["text"]


class GPT3_5LLM(LLM):
    def __init__(
        self,
        openai_key,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        super().__init__()
        openai.api_key = openai_key
        self.model = "gpt-3.5-turbo"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.tokens = 0

    # @retry(Exception, tries=8, delay=1)
    def response(
        self,
        user,
        system="You are ChatGPT, a large language model trained by OpenAI.\n"
        + "Knowledge cutoff: 2021-09\n"
        + "Current date: 2023-6-27",
        stop="\n",
    ):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=stop,
        )
        self.tokens += response["usage"]["total_tokens"]
        return response["choices"][0]["message"]["content"]
