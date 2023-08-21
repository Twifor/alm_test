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

REFLECT_INSTRUCTION = """You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to multiple tools and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Answer(<answer>), or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.
Here are some examples:"""

REFLEXION_PROMPT = PromptTemplate(
    input_variables=["prompt", "tool_description",
                     "task", "history", "examples", "reflexion_prompt"],
    template="""{prompt}
{tool_description}
There are some examples, which describes the output format (including the corresponding observation) you need to follow:
{examples}

{reflexion_prompt}

Task: {task}
{history}

Then take your next step.
Thought:
""")

REFLEXION_EXAMPLE = """Previous Trial:
Question: The Rome Protocols were signed by three Prime Ministers one of which was assassinated as part of what?
Thought: I need to search Rome Protocols, find the three Prime Ministers, then find what they were assassinated as part of.
Action: Search(Rome Protocols)
Observation: The Rome Protocols were a series of three international agreements concluded in Rome on 17 March 1934 between the governments of Austria, Hungary and Italy. They were signed by Italian Prime Minister Benito Mussolini, Austrian Chancellor Engelbert Dollfuss and Hungarian Prime Minister Gyula Gömbös. All of the protocols went into effect on 12 July 1934 and were registered in League of Nations Treaty Series on 12 December 1934.
Thought: The three Prime Ministers were Benito Mussolini, Engelbert Dollfuss, and Gyula Gömbös. I need to search Benito Mussolini and find what he was assassinated as part of.
Action: Search(Benito Mussolini)
Observation: Benito Amilcare Andrea Mussolini (UK:  MU(U)SS-ə-LEE-nee, US:  MOOSS-, Italian: [beˈniːto aˈmilkare anˈdrɛːa mussoˈliːni]; 29 July 1883 – 28 April 1945) was an Italian politician and journalist who founded and led the National Fascist Party (PNF). He was Prime Minister of Italy from the March on Rome in 1922 until his deposition in 1943, as well as "Duce" of Italian fascism from the establishment of the Italian Fasces of Combat in 1919 until his summary execution in 1945 by Italian partisans. As dictator of Italy and principal founder of fascism, Mussolini inspired and supported the international spread of fascist movements during the inter-war period.Mussolini was originally a socialist politician and a journalist at the Avanti! newspaper. In 1912, he became a member of the National Directorate of the Italian Socialist Party (PSI), but he was expelled from the PSI for advocating military intervention in World War I, in opposition to the party's stance on neutrality. In 1914, Mussolini founded a new journal, Il Popolo d'Italia, and served in the Royal Italian Army during the war until he was wounded and discharged in 1917. Mussolini denounced the PSI, his views now centering on Italian nationalism instead of socialism, and later founded the fascist movement which came to oppose egalitarianism and class conflict, instead advocating "revolutionary nationalism" transcending class lines. On 31 October 1922, following the March on Rome (28–30 October), Mussolini was appointed prime minister by King Victor Emmanuel III, becoming the youngest individual to hold the office up to that time. After removing all political opposition through his secret police and outlawing labor strikes, Mussolini and his followers consolidated power through a series of laws that transformed the nation into a one-party dictatorship. Within five years, Mussolini had established dictatorial authority by both legal and illegal means and aspired to create a totalitarian state. In 1929, Mussolini signed the Lateran Treaty with the Holy See to establish Vatican City.
Mussolini's foreign policy aimed to restore the ancient grandeur of the Roman Empire by expanding Italian colonial possessions and the fascist sphere of influence. In the 1920s, he ordered the Pacification of Libya, instructed the bombing of Corfu over an incident with Greece, established a protectorate over Albania, and incorporated the city of Fiume into the Italian state via agreements with Yugoslavia. In 1936, Ethiopia was conquered following the Second Italo-Ethiopian War and merged into Italian East Africa (AOI) with Eritrea and Somalia. In 1939, Italian forces annexed Albania. Between 1936 and 1939, Mussolini ordered the successful Italian military intervention in Spain in favor of Francisco Franco during the Spanish Civil War. Mussolini's Italy initially tried to avoid the outbreak of a second global war, sending troops at the Brenner Pass to delay Anschluss and taking part in the Stresa Front, the Lytton Report, the Treaty of Lausanne, the Four-Power Pact and the Munich Agreement. However, Italy then alienated itself from Britain and France by aligning with Germany and Japan. Germany invaded Poland on 1 September 1939, resulting in declarations of war by France and the UK and the start of World War II.
On 10 June 1940, Mussolini decided to enter the war on the Axis side. Despite initial success, the subsequent Axis collapse on multiple fronts and eventual Allied invasion of Sicily made Mussolini lose the support of the population and members of the Fascist Party. As a consequence, early on 25 July 1943, the Grand Council of Fascism passed a motion of no confidence in Mussolini; later that day King Victor Emmanuel III dismissed him as head of government and had him placed in custody, appointing Pietro Badoglio to succeed him as Prime Minister. After the king agreed to an armistice with the Allies, on 12 September 1943 Mussolini was rescued from captivity in the Gran Sasso raid by German paratroopers and Waffen-SS commandos led by Major Otto-Harald Mors. Adolf Hitler, after meeting with the rescued former dictator, then put Mussolini in charge of a puppet regime in northern Italy, the Italian Social Republic (Italian: Repubblica Sociale Italiana, RSI), informally known as the Salò Republic, causing a civil war. In late April 1945, in the wake of near total defeat, Mussolini and his mistress Clara Petacci attempted to flee to Switzerland, but both were captured by Italian communist partisans and summarily executed by firing squad on 28 April 1945 near Lake Como. The bodies of Mussolini and his mistress were then taken to Milan, where they were hung upside down at a service station to publicly confirm their demise.
Thought: Benito Mussolini was assassinated as part of the collapse of the Axis powers during World War II. So the Rome Protocols were signed by three Prime Ministers one of which was assassinated as part of World War II.
Action: Finish(World War II)

Reflection: I searched one of the prime ministers involved in the signing, then attemted to answer right away. I should have searched each of the prime ministers, then looked up 'death' on each of their pages in order to get more information before answering.
"""

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
3. [MOST IMPORTANT]. Some tools may provide errors. You should discover them and provide lower scores to these tools.

To provide your scores, you need to output as the following format:
First, output \"Tool:\" followed by the tool name.
Next line, output \"Score:\" followed by your score, which is an integer between -3 and 3, higher score means higher performance.
Then, in the next line, output \"Tought:\" followed by your reason why you give such a score to this tool.
Then provide the score of the next tool in the next following lines.
Output END to finish your answer."""
)

#
# 4. If the error is caused by agent itself instead of the tool, you can not provide lower score just because of this error.
# 3. You must provide the score of tool based on the its usefulness on the given history.
# 4. Some tools may provide errors. You should discover them and provide lower scores to these tools.


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
