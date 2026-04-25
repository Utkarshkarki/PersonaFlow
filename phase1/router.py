from sentence_transformers import SentenceTransformer
from models.phase1 import RoutingResult, BotPersona
from phase1.vector_store import VectorStore
import json
from pathlib import Path
from typing import List, Dict

class PostRouter:
    """Routes posts to bots using vector similarity with full logging."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store: VectorStore = None
    ):
        self.embedder = SentenceTransformer(embedding_model)
        self.vector_store = vector_store or VectorStore(embedding_dim=384)  # MiniLM = 384 dims
        self.routing_logs: List[RoutingResult] = []
    
    def initialize_personas(self, personas_dict: Dict[str, str]):
        """
        Initialize bot personas from a dict.
        
        Args:
            personas_dict: {"A": "I believe AI...", "B": "I believe late-stage...", ...}
        """
        persona_names = {"A": "Tech Maximalist", "B": "Doomer/Skeptic", "C": "Finance Bro"}
        
        for bot_id, description in personas_dict.items():
            embedding = self.embedder.encode(description, convert_to_tensor=False).tolist()
            persona = BotPersona(
                bot_id=bot_id,
                name=persona_names.get(bot_id, f"Bot {bot_id}"),
                description=description,
                embedding=embedding
            )
            self.vector_store.add_persona(persona)
    
    def route_post(
        self,
        post_content: str,
        threshold: float = 0.85,
        log_file: Path = None
    ) -> RoutingResult:
        """
        Route a post to matching bots and log results.
        """
        # Embed the post
        post_embedding = self.embedder.encode(post_content, convert_to_tensor=False).tolist()
        
        # Query vector store
        result = self.vector_store.query_with_scores(
            post_content=post_content,
            post_embedding=post_embedding,
            k=3,
            threshold=threshold
        )
        
        # Log to memory
        self.routing_logs.append(result)
        
        # Log to file if requested
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"POST: {post_content[:80]}...\n")
                f.write(result.score_summary)
                f.write(f"\nMATCHED BOTS: {', '.join(result.matched_bots) or 'None'}\n")
        
        return result
    
    def export_logs(self, output_file: Path):
        """Export all routing logs as JSON."""
        logs_json = [log.model_dump() for log in self.routing_logs]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(logs_json, f, indent=2)
        print(f"[+] Exported {len(self.routing_logs)} routing logs to {output_file}")
