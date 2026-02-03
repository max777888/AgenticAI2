from langchain_ollama import OllamaLLM
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent


class OllamaAgent(BaseAgent):

    def __init__(self, memory):
        super().__init__(memory)
        self.llm = OllamaLLM(model="llama3")
        self.executor = self._create_executor()

    def _create_executor(self):
        prompt = PromptTemplate.from_template(
            """You are a helpful insurance claims assistant.
Use the provided tools when necessary.

Chat history:
{chat_history}

Question: {input}

Thought process: {agent_scratchpad}"""
        )

        agent = create_react_agent(self.llm, self.tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, query: str) -> str:
        result = self.executor.invoke({"input": query})
        return result["output"]