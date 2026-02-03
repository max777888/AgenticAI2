from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage
from prompts import NEWS_AGENT_PROMPT, REFLECTION_PROMPT

from tools import get_tools


# ── Force-load .env as early as possible ───────────────────────────────
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Explicitly locate .env relative to this file (most reliable on Windows)
env_path = Path(__file__).parent / '.env'

# Load it
loaded = load_dotenv(env_path)

# Debug prints – keep these until it works
print("Current working directory       :", os.getcwd())
print(".env file path being checked    :", env_path.absolute())
print(".env file actually exists?      :", env_path.is_file())
print("load_dotenv() success?          :", loaded)
print("GROQ_API_KEY after loading      :", os.getenv("GROQ_API_KEY", "NOT_FOUND"))

print("-" * 60)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ── LLM ────────────────────────────────────────────────────────────────
# In agent_graph.py → replace your llm = ChatGroq(...) block with:

# from langchain_groq import ChatGroq

# llm = ChatGroq(
#     model="llama-3-3-70b-versatile",          # ← Best for agents / tools (high accuracy)
#     temperature=0,
#     max_retries=2,
# )


llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",  # This is the current best-practice model
    temperature=0,
    max_retries=6,  # Increase retries to handle the 429 cooling period
    timeout=None,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Optional: faster/cheaper/smaller alternatives (change model=...)
# "llama-3.1-8b-instant"     ← very fast, cheaper, still good for agents
# "mixtral-8x7b-32768"       ← classic strong performer
# "gemma2-9b-it"             ← quick & capable



tools = get_tools()
llm_with_tools = llm.bind_tools(tools)

agent_runnable = NEWS_AGENT_PROMPT | llm_with_tools
reflect_runnable = REFLECTION_PROMPT | llm


def call_agent(state: AgentState):
    """Agent node: reasons and decides whether to call tools"""
    messages = state["messages"]
    response: AIMessage = agent_runnable.invoke({"messages": messages})
    return {"messages": [response]}


def call_tools(state: AgentState):
    """Tool execution node"""
    messages = state["messages"]
    last_message = messages[-1]

    tool_responses = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        tool = next(t for t in tools if t.name == tool_name)
        try:
            result = tool.invoke(tool_args)
            tool_responses.append({
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_id
            })
        except Exception as e:
            tool_responses.append({
                "role": "tool",
                "content": f"Tool error: {str(e)}",
                "tool_call_id": tool_id
            })

    return {"messages": tool_responses}


def should_continue(state: AgentState):
    """Router: decide next step after agent"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "reflect"  # go to reflection before finishing


def reflect(state: AgentState):
    """Reflection node: critique the current summary"""
    # 1. Extract the latest AI message
    messages = state["messages"]
    last_ai_msg = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    
    if not last_ai_msg or not last_ai_msg.content:
        return {"messages": [AIMessage(content="GOOD_ENOUGH")]}

    # 2. Extract subject from the first human message
    subject = "the topic"
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == "human":
            subject = msg.content
            break
        elif isinstance(msg, tuple) and msg[0] == "human":
            subject = msg[1]
            break

    # 3. Call the model using the CORRECT UPPERCASE NAME from prompts.py
    # We use .format_messages because REFLECTION_PROMPT is a ChatPromptTemplate
    formatted_prompt = REFLECTION_PROMPT.format_messages(
        subject=subject,
        summary=last_ai_msg.content
    )
    critique = llm.invoke(formatted_prompt)
    
    # 4. Handle Gemini's "List of Blocks" content format
    if isinstance(critique.content, list):
        content = "".join([block.get("text", "") for block in critique.content if isinstance(block, dict)])
    else:
        content = critique.content
    
    content = content.strip()
    
    if "NEEDS_MORE" in content:
        return {"messages": [AIMessage(content=f"Reflection: {content}")]}
    else:
        return {"messages": [AIMessage(content="GOOD_ENOUGH")]}

def route_after_reflect(state: AgentState):
    last_msg = state["messages"][-1]
    if "NEEDS_MORE" in last_msg.content:
        return "agent"
    return END


# ── Build Graph ────────────────────────────────────────────────────────
builder = StateGraph(state_schema=AgentState)

builder.add_node("agent", call_agent)
builder.add_node("tools", call_tools)
builder.add_node("reflect", reflect)

builder.set_entry_point("agent")

builder.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "reflect": "reflect"}
)

builder.add_edge("tools", "agent")           # loop back after tools
builder.add_conditional_edges(
    "reflect",
    route_after_reflect,
    {"agent": "agent", END: END}
)

# Add memory / persistence
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)