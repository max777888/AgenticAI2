from typing import TypedDict
from langchain_ollama import OllamaLLM
from langchain.memory import ConversationBufferMemory
from agents.ollama_agent import OllamaAgent
from agents.gemini_agent import GeminiAgent


class WorkflowState(TypedDict):
    query: str
    is_complex: bool
    response: str


class ClaimProcessingWorkflow:

    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.ollama_agent = OllamaAgent(self.memory)
        self.gemini_agent = GeminiAgent(self.memory)

    def _is_complex(self, query: str) -> bool:
        prompt = f"""Classify if this insurance query is COMPLEX.
Complex = multi-step reasoning / legal interpretation / calculations / comparison of multiple policies / detailed analysis

Respond with ONLY one word: Yes or No

Query: {query}"""
        llm = OllamaLLM(model="llama3")
        answer = llm.invoke(prompt).strip().lower()
        return "yes" in answer.lower()

    def run(self, query: str) -> str:
        if self._is_complex(query):
            return self.gemini_agent.run(query)
        else:
            return self.ollama_agent.run(query)