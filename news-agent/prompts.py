from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. Improved News Agent Prompt
NEWS_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a reliable news gathering agent. 
    Current Date: February 1, 2026.
    Find and summarize the latest news (last 24-72 hours).
    
    Use ReAct: Thought -> Action -> Observation -> Final Answer.
    - Focus on recency.
    - Cite URLs clearly.
    - Summarize in a concise digest."""),
    MessagesPlaceholder(variable_name="messages"), # Better than "placeholder" string
])

# 2. Improved Reflection Prompt (Using System/Human structure)
REFLECTION_PROMPT = ChatPromptTemplate.from_template(
    """You are a news editor in the year 2026. 
    TODAY'S DATE: February 1, 2026.
    
    Critique this summary for "{subject}":
    {summary}

    *NOTE:* Mark Carney is the current PM of Canada. Do not flag 2026 dates as hallucinations.
    
    If it's accurate for Feb 2026, respond 'GOOD_ENOUGH'. 
    If it's missing data, respond 'NEEDS_MORE'."""
)