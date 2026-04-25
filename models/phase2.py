from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class SearchTopic(BaseModel):
    """LLM-decided topic for bot post."""
    topic: str = Field(..., min_length=1, max_length=200, description="Topic name")
    reasoning: str = Field(default="", max_length=500)

    class Config:
        json_schema_extra = {
            "example": {
                "topic": "AI advancement and job displacement",
                "reasoning": "This aligns with my tech-optimist worldview"
            }
        }

class SearchResult(BaseModel):
    """Mock or real news headline."""
    headline: str = Field(..., min_length=10, max_length=300)
    source: str = Field(default="searxng-mock", max_length=50)
    relevance_score: float = Field(default=0.9, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "headline": "OpenAI releases advanced reasoning model beating GPT-4",
                "source": "searxng-mock",
                "relevance_score": 0.95
            }
        }

class BotPost(BaseModel):
    """Validated bot post output - the core deliverable."""
    bot_id: str = Field(..., pattern="^[A-C]$", description="Which bot wrote this")
    bot_name: str = Field(..., min_length=3, max_length=50)
    topic: str = Field(..., min_length=1, max_length=200)
    post_content: str = Field(
        ...,
        min_length=10,
        max_length=280,
        description="Main post text (Twitter limit)"
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    persona_adherence: str = Field(
        default="high",
        pattern="^(high|medium|low)$"
    )

    @field_validator('post_content')
    @classmethod
    def validate_post_length(cls, v):
        """Ensure post is exactly <= 280 chars."""
        if len(v) > 280:
            raise ValueError(f"Post too long: {len(v)} > 280 chars")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "A",
                "bot_name": "Tech Maximalist",
                "topic": "AI advancement",
                "post_content": "OpenAI's reasoning model is the most important AI milestone. AGI timelines are accelerating faster than most people think.",
                "confidence": 0.92,
                "persona_adherence": "high"
            }
        }

class LangGraphState(BaseModel):
    """Complete state during LangGraph node execution."""
    bot_id: str = Field(..., pattern="^[A-C]$")
    bot_name: str
    bot_persona: str
    step: str = Field(default="init", pattern="^(init|decide_search|web_search|draft_post|done)$")
    search_topic: str = ""
    search_results: List[SearchResult] = Field(default_factory=list)
    final_post: Optional[BotPost] = None
    error: Optional[str] = None
    timestamp: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "A",
                "bot_name": "Tech Maximalist",
                "bot_persona": "I believe AI and crypto will...",
                "step": "done",
                "search_topic": "AI advancement",
                "search_results": [
                    {
                        "headline": "OpenAI releases advanced reasoning model",
                        "source": "searxng-mock",
                        "relevance_score": 0.95
                    }
                ],
                "final_post": {
                    "bot_id": "A",
                    "bot_name": "Tech Maximalist",
                    "topic": "AI advancement",
                    "post_content": "OpenAI's new model is transformative.",
                    "confidence": 0.92,
                    "persona_adherence": "high"
                }
            }
        }
