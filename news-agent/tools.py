from langchain_community.tools.tavily_search import TavilySearchResults

def get_tools():
    """Returns the list of tools the agent can use"""
    search = TavilySearchResults(
        max_results=6,                  # adjust based on needs
        include_raw_content=False,      # cleaner JSON output
        include_images=False,
        search_depth="advanced",        # helps with freshness
    )
    return [search]