from typing import Dict, Any, List
import uuid
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


class ClaimStorage:
    """Handles storage and retrieval of claims using Chroma vector database"""

    def __init__(self, embedding_model: str = "mxbai-embed-large"):
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore = Chroma(
            collection_name="insurance_claims",
            embedding_function=self.embeddings,
            persist_directory="./chroma_db"   # optional – saves to disk
        )

    def submit_claim(self, text: str) -> str:
        """Store a new claim and return its generated ID"""
        claim_id = str(uuid.uuid4())[:8]
        self.vectorstore.add_texts(
            texts=[text],
            metadatas=[{"claim_id": claim_id, "source": "user"}]
        )
        # self.vectorstore.persist()  # only needed in older Chroma versions
        return claim_id

    def find_similar_claims(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve top-k similar claims"""
        docs = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                "claim_id": doc.metadata.get("claim_id", "unknown"),
                "content": doc.page_content,
            }
            for doc in docs
        ]