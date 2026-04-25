from pydantic import BaseModel, Field
from typing import List, Dict

class BotPersona(BaseModel):
    """Represents a bot's persona for vector embedding."""
    bot_id: str = Field(..., description="Unique bot identifier (A, B, C)")
    name: str = Field(..., description="Bot name")
    description: str = Field(..., description="Full persona text for embedding")
    embedding: List[float] = Field(default_factory=list, description="Vector embedding (768+ dims)")

    class Config:
        json_schema_extra = {
            "example": {
                "bot_id": "A",
                "name": "Tech Maximalist",
                "description": "I believe AI and crypto will solve all human problems..."
            }
        }

class SimilarityScore(BaseModel):
    """Individual bot similarity match."""
    bot_id: str
    bot_name: str
    similarity: float = Field(..., ge=0.0, le=1.0)
    matches: bool = Field(..., description="True if similarity >= threshold")

class RoutingResult(BaseModel):
    """Complete routing output with transparency logs."""
    post_content: str = ""
    post_embedding_dims: int
    threshold: float
    matched_bots: List[str]
    similarity_scores: List[SimilarityScore] = Field(
        ..., 
        description="All scores, sorted descending"
    )
    timestamp: str
    notes: str = ""

    @property
    def score_summary(self) -> str:
        """Human-readable similarity table."""
        lines = ["Similarity Scores:", "-" * 40]
        for score in self.similarity_scores:
            match_marker = "✓" if score.matches else "✗"
            lines.append(f"{match_marker} {score.bot_name:20} = {score.similarity:.4f}")
        return "\n".join(lines)
