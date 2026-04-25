import faiss
import numpy as np
from datetime import datetime
from typing import List, Tuple, Dict
from models.phase1 import BotPersona, SimilarityScore, RoutingResult

class VectorStore:
    """FAISS-backed vector store with transparency logging."""
    
    def __init__(self, embedding_dim: int = 384): # all-MiniLM-L6-v2 uses 384 dims
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.bot_personas: Dict[int, BotPersona] = {}
        self.embedding_dim = embedding_dim
        self.id_to_bot = {}  # faiss_id → bot_id mapping
    
    def add_persona(self, persona: BotPersona):
        """Add bot persona to the store."""
        idx = self.index.ntotal
        embedding = np.array(persona.embedding, dtype=np.float32).reshape(1, -1)
        self.index.add(embedding)
        self.id_to_bot[idx] = persona.bot_id
        self.bot_personas[idx] = persona
        print(f"[+] Added {persona.name} (ID: {persona.bot_id}) at index {idx}")
    
    def query_with_scores(
        self, 
        post_content: str,
        post_embedding: List[float],
        k: int = 3,
        threshold: float = 0.85
    ) -> RoutingResult:
        """
        Query vector store and return similarity scores for all bots.
        """
        post_vec = np.array(post_embedding, dtype=np.float32).reshape(1, -1)
        
        # FAISS uses L2 distance; convert to cosine similarity
        distances, indices = self.index.search(post_vec, k)
        
        similarity_scores = []
        for faiss_id in sorted(self.bot_personas.keys()):
            bot_persona = self.bot_personas[faiss_id]
            cosine_sim = np.dot(post_vec[0], np.array(bot_persona.embedding)) / (
                np.linalg.norm(post_vec[0]) * np.linalg.norm(np.array(bot_persona.embedding)) + 1e-8
            )
            
            score = SimilarityScore(
                bot_id=bot_persona.bot_id,
                bot_name=bot_persona.name,
                similarity=max(0.0, min(1.0, float(cosine_sim))),
                matches=float(cosine_sim) >= threshold
            )
            similarity_scores.append(score)
        
        # Sort by similarity descending
        similarity_scores.sort(key=lambda x: x.similarity, reverse=True)
        matched_bots = [s.bot_id for s in similarity_scores if s.matches]
        
        return RoutingResult(
            post_content=post_content[:100] + "..." if len(post_content) > 100 else post_content,
            post_embedding_dims=len(post_embedding),
            threshold=threshold,
            matched_bots=matched_bots,
            similarity_scores=similarity_scores,
            timestamp=datetime.now().isoformat(),
            notes=f"Matched {len(matched_bots)} bot(s) above threshold"
        )
