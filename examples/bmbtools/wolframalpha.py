
import requests
from agent.tools import Tool
from typing import Union, Dict
from typing import Any
import os
import xmltodict


class GetWolframAlphaResultsTool(Tool):
    def __init__(self, appid):
        super().__init__()
        self.invoke_label = "GetWolframAlphaResults"
        self.appid = appid

    def invoke(self, invoke_data) -> Union[str, float, bool, Dict]:
        input = invoke_data
        URL = "https://api.wolframalpha.com/v2/query"
        APPID = self.appid

        params = {'appid': APPID, "input": input}

        response = requests.get(URL, params=params)

        json_data = xmltodict.parse(response.text)

        if 'pod' not in json_data["queryresult"]:
            return "WolframAlpha API cannot parse the input query.", 0, False, {}
        rets = json_data["queryresult"]['pod']

        cleaned_rets = []
        blacklist = ["@scanner", "@id", "@position", "@error", "@numsubpods", "@width",
                     "@height", "@type", "@themes", "@colorinvertable", "expressiontypes"]

        def filter_dict(d, blacklist):
            if isinstance(d, dict):
                return {k: filter_dict(v, blacklist) for k, v in d.items() if k not in blacklist}, 0, False, {}
            elif isinstance(d, list):
                return [filter_dict(i, blacklist) for i in d], 0, False, {}
            else:
                return d, 0, False, {}

        for ret in rets:
            ret = filter_dict(ret, blacklist=blacklist)
            # Do further cleaning to retain only the input and result pods
            if "@title" in ret:
                if ret["@title"] == "Input" or ret["@title"] == "Result":
                    cleaned_rets.append(ret)

        return cleaned_rets, 0, False, {}

    def description(self) -> str:
        return "GetWolframAlphaResults(input), Get Wolfram|Alpha results using natural query. Queries to getWolframAlphaResults must ALWAYS have this structure: {\"input\": query}. And please directly read the output json."
