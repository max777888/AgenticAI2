import os, json
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# 1. SETUP MODELS & RAG
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
local_llm = ChatOllama(model="llama3", temperature=0)
cloud_llm = ChatOllama(model="gpt-oss:120b-cloud", temperature=0)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], "History"]
    complexity: int
    context: str

# 2. NODES
def ingest_node(state: AgentState):
    with open('claim_doc.txt', 'r') as f:
        text = f.read()
    vectorstore.add_texts([text])
    return {"context": "Document Ingested"}

def router_node(state: AgentState):
    query = state['messages'][-1].content
    # Fetch from Vector DB
    docs = vectorstore.similarity_search(query, k=1)
    context = docs[0].page_content if docs else "No docs found."
    
    # Logic: If query is long or mentions liability, escalate
    complexity = 10 if len(query) > 100 or "liability" in query.lower() else 1
    return {"complexity": complexity, "context": context}

def local_handler(state: AgentState):
    msg = f"CONTEXT: {state['context']}\n\nUSER: {state['messages'][-1].content}"
    return {"messages": [local_llm.invoke([SystemMessage(content="Simple Assistant"), HumanMessage(content=msg)])]}

def cloud_handler(state: AgentState):
    msg = f"CONTEXT: {state['context']}\n\nUSER: {state['messages'][-1].content}"
    return {"messages": [cloud_llm.invoke([SystemMessage(content="Senior Adjuster"), HumanMessage(content=msg)])]}

# 3. GRAPH
workflow = StateGraph(AgentState)
workflow.add_node("ingest", ingest_node)
workflow.add_node("router", router_node)
workflow.add_node("local", local_handler)
workflow.add_node("cloud", cloud_handler)

workflow.add_edge(START, "ingest")
workflow.add_edge("ingest", "router")
workflow.add_conditional_edges("router", lambda x: "cloud" if x["complexity"] > 5 else "local")
workflow.add_edge("local", END)
workflow.add_edge("cloud", END)

app = workflow.compile()

if __name__ == "__main__":
    print("Agent ready. Testing simple query...")
    user_input = "What happened in policy POL-991's claim?"
    for output in app.stream({"messages": [HumanMessage(content=user_input)]}):
        print(output)