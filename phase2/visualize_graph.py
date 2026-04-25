from phase2.langgraph_agent import ContentEngineAgent
from pathlib import Path

def visualize_graph():
    """Export LangGraph diagram."""
    agent = ContentEngineAgent()
    
    # Generate PNG
    output_dir = Path("logs/graphs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Method 1: Using Mermaid (recommended)
        mermaid_diagram = agent.graph.get_graph().draw_mermaid()
        
        with open(output_dir / "langgraph_mermaid.md", "w") as f:
            f.write("# LangGraph State Machine\n\n")
            f.write("```mermaid\n")
            f.write(mermaid_diagram)
            f.write("\n```\n")
        
        print(f"[+] Mermaid diagram saved to logs/graphs/langgraph_mermaid.md")
        
    except Exception as e:
        print(f"[-] Mermaid export failed: {e}")
    
    try:
        # Method 2: ASCII representation
        ascii_diagram = agent.graph.get_graph().draw_ascii()
        
        with open(output_dir / "langgraph_ascii.txt", "w") as f:
            f.write(ascii_diagram)
        
        print(f"[+] ASCII diagram saved to logs/graphs/langgraph_ascii.txt")
        
    except Exception as e:
        print(f"[-] ASCII export failed: {e}")

if __name__ == "__main__":
    visualize_graph()
