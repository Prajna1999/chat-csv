import langchain
from langchain.llms import CTransformers

def load_llm():
    langchain.verbose = False
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm
