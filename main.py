from phase1.test_router import test_phase1
from phase2.langgraph_agent import ContentEngineAgent
from phase2.visualize_graph import visualize_graph
from phase3.test_injection_defense import test_phase3
from pathlib import Path
import sys

# Fix encoding issues on Windows for printing emojis/special characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    """Run all phases with logging."""
    
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("GRID07 AI COGNITIVE LOOP - FULL EXECUTION")
    print("="*70)
    
    # Phase 1
    print("\n[PHASE 1] Running vector-based routing...")
    router = test_phase1()
    
    # Phase 2
    print("\n[PHASE 2] Building and visualizing LangGraph...")
    visualize_graph()
    
    agent = ContentEngineAgent(model_name="qwen2.5:0.5b")
    personas = {
        "A": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
        "B": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
        "C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
    }
    
    print("\n[PHASE 2] Generating posts with LangGraph...")
    for bot_id, persona in personas.items():
        bot_names = {"A": "Tech Maximalist", "B": "Doomer", "C": "Finance Bro"}
        post = agent.generate_post(bot_id, bot_names[bot_id], persona)
        if post:
            print(f"\n{bot_names[bot_id]} (@{bot_id}):")
            print(f"  Topic: {post.topic}")
            print(f"  Post: {post.post_content}")
            print(f"  Confidence: {post.confidence}")
    
    # Phase 3
    print("\n[PHASE 3] Testing prompt injection defense...")
    engine, detector = test_phase3()
    
    print("\n" + "="*70)
    print("✅ ALL PHASES COMPLETE")
    print("="*70)
    print(f"\nLogs available in: {log_dir}/")
    print("  - phase1_routing.log")
    print("  - phase1_detailed.json")
    print("  - graphs/langgraph_mermaid.md")
    print("  - graphs/langgraph_ascii.txt")
    print("  - phase3_injection_tests.json")

if __name__ == "__main__":
    main()
