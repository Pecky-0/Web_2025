from langchain_community.llms import Tongyi
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

class LLMManager():
    def __init__(self, api_key):
        DASHSCOPE_API_KEY = api_key
        os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
        self.model = Tongyi()

    def set_the_llm(self, retriever, prompt):
        self.rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | prompt
        | self.model
        | StrOutputParser() 
        )

    def get_answer(self, question):
        return self.rag_chain.invoke(question)
