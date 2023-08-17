from langchain.prompts import PromptTemplate
REACT_EXAMPLES = """1.
Thought: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.
Action: Search(Colorado orogeny)
Observation: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.
2. 
Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup(eastern sector)
Observation: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.
3.
Thought: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.
Action: Search(High Plains)
Observation: High Plains refers to one of two distinct land regions.
End of the examples."""

REACT_INSTRUCTION = """Solve a task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation. There are some rules your should follow:
1. You must output your answer in the following format:\nFirstly, output your thought then output "Action:" followed by your action. Finally, output "Observation:" to finish your output.
2. The action you provide should be the format like "tool_label(tool_arg1, tool_arg2)". Except this, do not output anything else. You need to provide the specified tool_label and then provide your parameters in parentheses.
3. You can only use the tools we provide to you. The list of tools you can use will be showed later.
4. You can ONLY invoke ONE tool at each step. You CANNOT invoke two or more tools at one step."""

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:"""


REACT_PROMPT = PromptTemplate(
    input_variables=["prompt", "tool_description",
                     "task", "history", "examples"],
    template="""{prompt}

{tool_description}
There are some examples, which describes the output format (including the corresponding observation) you need to follow:
{examples}

Task: {task}
{history}

Then take your next step.
Thought:
"""
)


AGENT_NETWORK_PROMPT = PromptTemplate(
    input_variables=["prompt", "examples",
                     "tool_description", "task", "history", "external_prompt"],
    template="""{prompt}
{tool_description}
There are some examples, which describes the output format (including the corresponding observation) you need to follow:
{examples}

{tool_description}

Now based on the following information and the task, take the next step.
Task: {task}
{history}

Then take your next step.
{external_prompt}
Thought:
"""
)

AGENT_NETWORK_REWARD_PROMPT = PromptTemplate(
    input_variables=["tool_description", "task", "history"],
    template="""There are some tools with their name, descrption and confidence score. Higher confidence score means more helpful performance.
Now you are an judger who need to judge each tool and provide the scores of each tool.
The tools you need to judge are listed here:
{tool_description}

Now there is another agent who aim to use these tools to finish a task. The task is:
{task}
The hisory of the whole process is:
{history}

According to the performance of these tools, provide your scores of each tool.
1. If the final answer of agent is corret, you should determine which tool is helphul and provide higher score.
2. If the final answer of agent is incorrect or not given, you should determine which tool is noisy and provide lower score.
3. You must provide the score of tool based on the its usefulness on the given history.
4. If the error is caused by agent itself instead of the tool, you can not provide lower score just because of this error.
To provide your scores, you need to output as the following format:
First, output \"Tool:\" followed by the tool name.
Next line, output \"Score:\" followed by your score, which is an integer between -3 and 3, higher score means higher performance.
Then, in the next line, output \"Tought:\" followed by your reason why you give such a score to this tool.
Then provide the score of the next tool in the next following lines.
Output END to finish your answer."""
)

COT_EXAMPLES = """Relevant Context: The Nile River is the longest river in the world, spanning approximately 6,650 kilometers (4,132 miles) in length. It flows through eleven countries in northeastern Africa, including Egypt, Sudan, and Uganda.
Question: What is the longest river in the world?
Thought: The question asks for the longest river in the world, which I know is the Nile River based on the context provided.
Action: Answer(Nile River)

Relevant Context: Ludwig van Beethoven was a German composer and pianist who was a crucial figure in the transition between the Classical and Romantic eras in Western classical music. One of his most famous compositions is the Symphony No. 9, also known as the "Choral" symphony.
Question: Which composer created Symphony No. 9?
Thought: The question is asking for the composer of Symphony No. 9. Based on the context, I know that Ludwig van Beethoven composed this symphony.
Action: Answer(Ludwig van Beethoven)

Relevant Context: Photosynthesis is the process by which green plants and some other organisms convert light energy into chemical energy. During this process, plants absorb sunlight, carbon dioxide, and water to produce glucose and oxygen.
Question: What do plants produce during photosynthesis?
Thought: The question is asking about the products of photosynthesis. From the context, I know that plants produce glucose and oxygen during this process.
Action: Answer(Glucose and Oxygen)

(END OF EXAMPLES)
"""

COT_PROMPT = PromptTemplate(
    input_variables=["examples", "context", "task"],
    template="""Solve a question answering task by having a Thought, then Answer with your answer. Thought can reason about the current situation. Answer(answer) returns the answer and finishes the task. You will be given context that you should use to help you answer the question.
Here are some examples:
{examples}

Relevant Context: {context}
Question: {task}
Thought:
"""
)
