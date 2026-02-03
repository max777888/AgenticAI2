import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from .base_agent import BaseAgent


class GeminiAgent(BaseAgent):

    def __init__(self, memory):
        super().__init__(memory)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.4
        )
        self.executor = self._create_executor()

    def _create_executor(self):
        prompt = PromptTemplate.from_template(
            """You are an expert insurance claims assistant.
Be precise and use tools when appropriate.

Chat history:
{chat_history}

User: {input}

Thought: {agent_scratchpad}"""
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