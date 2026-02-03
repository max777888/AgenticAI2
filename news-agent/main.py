import os
import time
from dotenv import load_dotenv

# Use 'rich' for professional terminal formatting
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner

from agent_graph import graph

# Initialize Rich console
console = Console()
load_dotenv()

def run_news_agent(subject: str, thread_id: str = "news-1"):
    """
    Runs the agentic news graph and formats the output for the terminal.
    """
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {
        "messages": [{"role": "user", "content": f"Latest news on {subject}"}]
    }

    console.print(Panel(f"[bold cyan]Gathering latest news on:[/bold cyan] [yellow]{subject}[/yellow]"), style="blue")

    final_news_content = ""
    
    # Use 'Live' to show status updates while the agent works
    with Live(Spinner("dots", text="[bold green]Agent Thinking...[/bold green]"), refresh_per_second=10) as live:
        for chunk in graph.stream(inputs, config=config, stream_mode="updates"):
            
            # 1. Handle Agent Node Updates
            if "agent" in chunk:
                msg = chunk["agent"]["messages"][-1]
                if msg.tool_calls:
                    live.update(f"[bold blue]→ Calling tools:[/bold blue] [white]{[c['name'] for c in msg.tool_calls]}[/white]")
                elif msg.content:
                    # Capture the content but don't print yet (we filter for the final summary)
                    final_news_content = msg.content

            # 2. Handle Tool Results
            elif "tools" in chunk:
                live.update(f"[bold magenta]→ Reading search results...[/bold magenta]")

            # 3. Handle Reflection Node
            elif "reflect" in chunk:
                reflection_msg = chunk["reflect"]["messages"][-1].content
                if "NEEDS_MORE" in reflection_msg:
                    live.update(f"[bold yellow]⚠ Reflection:[/bold yellow] [italic]Summary needs more detail. Researching further...[/italic]")
                else:
                    live.update(f"[bold green]✔ Reflection:[/bold green] [italic]Summary verified and complete![/italic]")
                    time.sleep(1) # Brief pause for readability

    # --- FINAL OUTPUT FORMATTING ---
    console.print("\n")
    if final_news_content and "###" in final_news_content:
        # We look for the message containing Markdown headers (###) 
        # to ensure we don't accidentally print "GOOD_ENOUGH" as the final news.
        
        # Create a beautiful Markdown display
        md = Markdown(final_news_content)
        console.print(Panel(md, title="[bold green]FINAL 2026 NEWS DIGEST[/bold green]", border_style="green", expand=False))
    else:
        # Fallback if the last message was the internal 'GOOD_ENOUGH' string
        # We fetch the actual summary from the state
        state = graph.get_state(config)
        for m in reversed(state.values["messages"]):
            if "###" in m.content:
                console.print(Panel(Markdown(m.content), title="[bold green]FINAL 2026 NEWS DIGEST[/bold green]", border_style="green"))
                break
        else:
            console.print("[bold red]No detailed summary was generated.[/bold red]")

if __name__ == "__main__":
    try:
        console.print("[bold]Enter news subject/topic (default: 'AI agents'): [/bold]", end="")
        user_input = input().strip() or "AI agents developments"
        run_news_agent(user_input)
    except KeyboardInterrupt:
        console.print("\n[bold red]Exiting...[/bold red]")