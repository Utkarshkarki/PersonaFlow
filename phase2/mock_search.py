from models.phase2 import SearchResult
from typing import List

class MockSearchxng:
    """Hardcoded fake news generator with keyword mapping."""
    
    # Keyword → list of fake headlines
    HEADLINE_MAP = {
        "crypto": [
            "Bitcoin surges past $50K on institutional ETF demand",
            "Ethereum founders debate network scaling amid gas fees",
            "Dogecoin community rallies behind Elon Musk endorsement",
        ],
        "ai": [
            "OpenAI releases advanced reasoning model beating GPT-4 benchmarks",
            "Tech researchers warn about AI bias in hiring algorithms",
            "Google's new LLM catches up to competitors in coding tasks",
        ],
        "regulation": [
            "EU imposes strict AI transparency requirements by 2025",
            "US lawmakers debate liability for AI-generated content",
            "China expands content moderation rules for tech platforms",
        ],
        "market": [
            "S&P 500 hits record high as rate cut fears ease",
            "Fed signals inflation controls working; analysts predict stability",
            "Tech stocks rally on earnings beat; Nvidia leads gains",
        ],
        "environment": [
            "Electric vehicle adoption accelerates; battery recycling emerges",
            "Carbon capture tech attracts billions in venture funding",
            "Renewable energy now 40% of global power generation",
        ],
        "space": [
            "SpaceX Starship achieves full orbital test flight",
            "NASA selects lunar lander design for 2026 Moon mission",
            "Commercial space tourism expands with suborbital flights",
        ],
    }
    
    @staticmethod
    def search(query: str) -> List[SearchResult]:
        """
        Mock search that returns headlines based on keywords in query.
        """
        results = []
        query_lower = query.lower()
        
        # Find matching keywords
        matched_headlines = []
        for keyword, headlines in MockSearchxng.HEADLINE_MAP.items():
            if keyword in query_lower:
                matched_headlines.extend(headlines)
        
        # If no keywords match, return general headlines
        if not matched_headlines:
            matched_headlines = [
                "Tech industry sees continued growth amid economic uncertainty",
                "Market analysts debate recession probability",
                "Innovation in AI and green energy accelerates",
            ]
        
        # Convert to SearchResult objects
        for i, headline in enumerate(matched_headlines[:3]):  # Max 3 results
            results.append(SearchResult(
                headline=headline,
                source="searxng-mock",
                relevance_score=0.95 - (i * 0.05)  # Decrease relevance for later results
            ))
        
        return results
