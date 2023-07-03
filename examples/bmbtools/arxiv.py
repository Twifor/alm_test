from typing import Any
import arxiv
from agent.tools import Tool
from typing import Union, Dict


class SearchArxivTool(Tool):
    def __init__(self):
        super().__init__()
        self.invoke_label = "SearchArxiv"

    def invoke(self, invoke_data) -> Union[str, int, bool, Dict]:
        arxiv_exceptions: Any  # :meta private:
        top_k_results: int = 3
        ARXIV_MAX_QUERY_LENGTH = 300
        doc_content_chars_max: int = 4000
        query = invoke_data
        param = {"q": query}
        try:
            results = arxiv.Search(  # type: ignore
                query[:ARXIV_MAX_QUERY_LENGTH], max_results=top_k_results
            ).results()
        except arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}", 0, False, {}
        docs = [
            f"Published: {result.updated.date()}\nTitle: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}"
            for result in results
        ]
        res = ""
        if docs:
            res = "\n\n".join(docs)[:doc_content_chars_max]
        else:
            res = "No good Arxiv Result was found"
        return res, 0, False, {}

    def description(self) -> str:
        return (
            "SearchArxiv(query), "
            + "Search information from Arxiv.org "
            + "Useful for when you need to answer questions about Physics, Mathematics, "
            + "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
            + "Electrical Engineering, and Economics "
            + "from scientific articles on arxiv.org. "
            + "Input should be a search query."
        )
