from agent.state_memory import HistoryState
from agent.llm import GPT3_5LLM

llm = GPT3_5LLM("<>")
print(llm.response("You are a smart assistant.", "Good evening!"))
