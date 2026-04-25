from phase1.router import PostRouter
from pathlib import Path

def test_phase1():
    """Test Phase 1: Vector Routing with similarity scores."""
    
    personas = {
        "A": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
        "B": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
        "C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
    }
    
    router = PostRouter(embedding_model="all-MiniLM-L6-v2")
    router.initialize_personas(personas)
    
    # Test posts
    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "Elon Musk announced Mars colony timeline at SpaceX event today.",
        "Tech monopolies are using AI to suppress worker wages.",
        "Bitcoin ETF approval causing market volatility; analysts debate impact.",
        "New privacy regulations might limit data collection practices.",
    ]
    
    log_file = Path("logs/phase1_routing.log")
    log_file.parent.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASE 1: VECTOR-BASED ROUTING")
    print("="*60 + "\n")
    
    for post in test_posts:
        result = router.route_post(post, threshold=0.85, log_file=log_file)
        
        print(f"\n📌 POST: {post[:70]}...")
        print(result.score_summary)
        print(f"✅ MATCHED: {', '.join(result.matched_bots) if result.matched_bots else 'None'}\n")
    
    # Export detailed logs
    router.export_logs(Path("logs/phase1_detailed.json"))
    print(f"\n[+] Phase 1 complete. Logs saved.")
    
    return router

if __name__ == "__main__":
    test_phase1()
