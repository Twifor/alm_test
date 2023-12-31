import requests
import json
import os
from agent.tools import Tool
from typing import Union, Dict
from typing import Any, Dict, List, Optional
import aiohttp
from pydantic.main import BaseModel
from pydantic.class_validators import root_validator
from googlesearch import search


class GoogleSerperAPIWrapper:
    def __init__(self, subscription_key) -> None:
        self.k: int = 10
        self.gl: str = "us"
        self.hl: str = "en"
        self.type: str = "search"  # type: search, images, places, news
        self.tbs: Optional[str] = None
        self.serper_api_key: str = subscription_key
        self.aiosession: Optional[aiohttp.ClientSession] = None

    def results(self, query: str, **kwargs: Any) -> Dict:
        """Run query through GoogleSearch."""
        return self._google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            tbs=self.tbs,
            search_type=self.type,
            **kwargs,
        )

    def run(self, query: str, **kwargs: Any) -> str:
        """Run query through GoogleSearch and parse result."""
        results = self._google_serper_search_results(
            query,
            gl=self.gl,
            hl=self.hl,
            num=self.k,
            tbs=self.tbs,
            search_type=self.type,
            **kwargs,
        )
        return self._parse_results(results)

    def _parse_snippets(self, results: dict) -> List[str]:
        snippets = []

        if results.get("answerBox"):
            answer_box = results.get("answerBox", {})
            if answer_box.get("answer"):
                return [answer_box.get("answer")]
            elif answer_box.get("snippet"):
                return [answer_box.get("snippet").replace("\n", " ")]
            elif answer_box.get("snippetHighlighted"):
                return answer_box.get("snippetHighlighted")

        if results.get("knowledgeGraph"):
            kg = results.get("knowledgeGraph", {})
            title = kg.get("title")
            entity_type = kg.get("type")
            if entity_type:
                snippets.append(f"{title}: {entity_type}.")
            description = kg.get("description")
            if description:
                snippets.append(description)
            for attribute, value in kg.get("attributes", {}).items():
                snippets.append(f"{title} {attribute}: {value}.")

        for result in results["organic"][: self.k]:
            if "snippet" in result:
                snippets.append(result["snippet"])
            for attribute, value in result.get("attributes", {}).items():
                snippets.append(f"{attribute}: {value}.")

        if len(snippets) == 0:
            return ["No good Google Search Result was found"]
        return snippets

    def _parse_results(self, results: dict) -> str:
        return " ".join(self._parse_snippets(results))

    def _google_serper_search_results(
        self, search_term: str, search_type: str = "search", **kwargs: Any
    ) -> dict:
        headers = {
            "X-API-KEY": self.serper_api_key or "",
            "Content-Type": "application/json",
        }
        params = {
            "q": search_term,
            **{key: value for key, value in kwargs.items() if value is not None},
        }
        response = requests.post(
            f"https://google.serper.dev/{search_type}", headers=headers, params=params
        )
        response.raise_for_status()
        search_results = response.json()
        return search_results


class GoogleSearchTool(Tool):
    def __init__(self, key):
        super().__init__()
        self.invoke_label = "GoogleSearch"
        self.api_wrapper = GoogleSerperAPIWrapper(key)

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        query = invoke_data
        return str(self.api_wrapper.run(query)), 0, False, {}

    def description(self) -> str:
        return "GoogleSearch(query), Run query through GoogleSearch and parse result. Useful for when you need to answer questions about current events. Input should be a search query."

class GoogleSearch2Tool(Tool):
    def __init__(self, limit:int = 3):
        super().__init__()
        self.invoke_label = "GoogleSearch2"
        self.limit = limit

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        query = invoke_data
        res = ""
        cnt = 0 
        for item in search(query, advanced=True):
            res += item.title + ", " + item.description + "\n"
            cnt += 1
            if cnt > self.limit:
                break
        return res, 0, False, {}

    def description(self) -> str:
        return "GoogleSearch2(query), Run query through GoogleSearch and parse result. Useful for when you need to answer questions about current events. Input should be a search query. Output is a JSON object of the query results."
