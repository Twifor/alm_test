from memory.vectorstores.vectorstore import VectorStoreRetrieverMemory
from memory.base_memory import BaseMemory
from memory.vectorstores.pinecone import Pinecone
from memory.vectorstores.chroma import Chroma
from memory.embeddings import OpenAIEmbeddings
import pydantic
import os

class Config:
    arbitrary_types_allowed = True

@pydantic.dataclasses.dataclass(config=Config)
class MemoryWrapper:
    memory: BaseMemory
    
    def __init__(self, memory: VectorStoreRetrieverMemory):
        self.memory = memory

    def save_history(self, input, output):
        self.memory.save_context({"input": input}, {"output": output})
    
    def query(self, prompt):
        return self.memory.load_memory_variables({"prompt": prompt})

def create_memory(memory_type, **kwargs) -> MemoryWrapper:
    # choose desirable memory you need!
    memory: BaseMemory = None
    if memory_type == "pinecone":
        # according to params, initialize your memory.
        import pinecone
        embedding_fn = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]).embed_query
        pinecone.init(api_key=os.environ["PINECONE_API_KEY"],environment=os.environ["PINECONE_ENVIRONMENT"])
        index = pinecone.Index(kwargs["index"])
        vectorstore = Pinecone(index, embedding_fn, kwargs["text_key"], namespace=kwargs.get("namespace"))
        retriever = vectorstore.as_retriever(search_kwargs=dict(k=kwargs["k"]))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
    elif memory_type == "chroma":
        chroma = Chroma(kwargs["index"], OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"]))
        retriever = chroma.as_retriever(search_kwargs=dict(k=kwargs["k"]))
        memory = VectorStoreRetrieverMemory(retriever=retriever)
    else:
        raise ValueError(f"Memory {memory_type} is not supported currently.")   
    return MemoryWrapper(memory=memory)