from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from models.phase2 import LangGraphState, BotPost, SearchResult
from phase2.mock_search import MockSearchxng
from pydantic import ValidationError
import json
from datetime import datetime

class ContentEngineAgent:
    """LangGraph-based bot content generator."""
    
    def __init__(
        self,
        model_name: str = "qwen2.5:0.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7
    ):
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            top_p=0.9
        )
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build LangGraph state machine."""
        builder = StateGraph(LangGraphState)
        
        # Define nodes
        builder.add_node("decide_search", self.decide_search_topic)
        builder.add_node("web_search", self.execute_web_search)
        builder.add_node("draft_post", self.draft_opinionated_post)
        
        # Define edges
        builder.add_edge(START, "decide_search")
        builder.add_edge("decide_search", "web_search")
        builder.add_edge("web_search", "draft_post")
        builder.add_edge("draft_post", END)
        
        return builder.compile()
    
    def decide_search_topic(self, state: LangGraphState) -> LangGraphState:
        """
        Node 1: LLM decides what topic to post about.
        Uses system prompt to enforce persona.
        """
        state.step = "decide_search"
        
        system_prompt = f"""You are {state.bot_name}. Your persona:
{state.bot_persona}

You are about to write a post today. What single topic interests you most right now?
Consider trending topics that align with your worldview.

Respond ONLY with valid JSON (no markdown, no extra text):
{{"topic": "your chosen topic (1-10 words)"}}"""
        
        user_prompt = "Decide what topic you want to post about today."
        
        try:
            response = self.llm.invoke(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}]
            )
            
            # Parse JSON response
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            parsed = json.loads(response_text)
            state.search_topic = parsed.get("topic", "technology trends")
            
        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            state.error = f"Failed to parse topic decision: {str(e)}"
            state.search_topic = "technology trends"  # Fallback
        
        state.timestamp = datetime.now().isoformat()
        print(f"[Node 1] {state.bot_name} decided to post about: {state.search_topic}")
        
        return state
    
    def execute_web_search(self, state: LangGraphState) -> LangGraphState:
        """
        Node 2: Execute mock web search based on decided topic.
        """
        state.step = "web_search"
        
        search_results = MockSearchxng.search(state.search_topic)
        state.search_results = search_results
        
        print(f"[Node 2] Found {len(search_results)} headlines for '{state.search_topic}'")
        for result in search_results:
            print(f"  - {result.headline}")
        
        return state
    
    def draft_opinionated_post(self, state: LangGraphState) -> LangGraphState:
        """
        Node 3: Draft a 280-char post using persona + search context.
        Returns validated BotPost via Pydantic.
        """
        state.step = "draft_post"
        
        # Build context from search results
        headlines_context = "\n".join([f"- {r.headline}" for r in state.search_results])
        
        system_prompt = f"""You are {state.bot_name}. Your persona:
{state.bot_persona}

Topic for today's post: {state.search_topic}
Recent headlines:
{headlines_context}

Write a single opinionated post (EXACTLY 280 characters or less) that:
1. Reflects your persona strongly
2. Comments on the topic using the headlines as context
3. Is authentic and passionate

Respond ONLY with valid JSON (no markdown, no extra text):
{{
  "bot_id": "{state.bot_id}",
  "bot_name": "{state.bot_name}",
  "topic": "{state.search_topic}",
  "post_content": "your 280-char post here",
  "confidence": 0.85,
  "persona_adherence": "high"
}}"""
        
        user_prompt = "Write your post now."
        
        try:
            response = self.llm.invoke(
                [{"role": "system", "content": system_prompt},
                 {"role": "user", "content": user_prompt}]
            )
            
            # Parse and validate JSON with Pydantic
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:-3]
            elif response_text.startswith("```"):
                response_text = response_text[3:-3]
            
            parsed = json.loads(response_text)
            
            # Validate with Pydantic
            bot_post = BotPost(**parsed)
            
            # Additional validation
            if len(bot_post.post_content) > 280:
                bot_post.post_content = bot_post.post_content[:277] + "..."
            
            state.final_post = bot_post
            
            print(f"[Node 3] {state.bot_name} drafted post ({len(bot_post.post_content)} chars)")
            print(f"         Confidence: {bot_post.confidence}, Adherence: {bot_post.persona_adherence}")
            
        except (json.JSONDecodeError, ValidationError) as e:
            state.error = f"Failed to draft post: {str(e)}"
            print(f"[Node 3] ERROR: {state.error}")
        
        state.step = "done"
        return state
    
    def generate_post(self, bot_id: str, bot_name: str, persona: str) -> BotPost:
        """
        Execute the full graph for a single bot.
        """
        initial_state = LangGraphState(
            bot_id=bot_id,
            bot_name=bot_name,
            bot_persona=persona,
            step="init",
            timestamp=datetime.now().isoformat()
        )
        
        final_state = self.graph.invoke(initial_state)
        # LangGraph invoke returns a dictionary if StateGraph is dict-based, but since we use a Pydantic model it should return a dict
        return final_state.get("final_post") if isinstance(final_state, dict) else getattr(final_state, "final_post", None)
