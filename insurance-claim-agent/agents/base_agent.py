from abc import ABC, abstractmethod
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory


class BaseAgent(ABC):
    """Base class for all claim agents"""

    def __init__(self, memory: ConversationBufferMemory):
        self.memory = memory
        self.tools = self._get_tools()

    @abstractmethod
    def run(self, query: str) -> str:
        """Execute the query and return the final answer"""

    def _get_tools(self):
        from policy_service import PolicyService
        from claim_storage import ClaimStorage

        policy_service = PolicyService()
        claim_storage = ClaimStorage()

        @tool
        def get_policy_details(policy_id: str) -> str:
            """Get details of an insurance policy by ID"""
            policy = policy_service.get_policy(policy_id)
            return str(policy) if policy else "Policy not found."

        @tool
        def retrieve_similar_claims(query: str) -> str:
            """Find previously submitted similar claims"""
            results = claim_storage.find_similar_claims(query)
            if not results:
                return "No similar claims found."
            return "\n".join(
                f"Claim {r['claim_id']}: {r['content']}"
                for r in results
            )

        return [get_policy_details, retrieve_similar_claims]