# src/services/rag_service.py
"""RAG service combining vector search with Gemini AI for conversational interface."""

from typing import List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI


# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.models import SearchResult, RAGResponse
from core.config import get_config, get_prompts
from services.vector_service import VectorService

@dataclass
class RAGConfig:
    """RAG service configuration."""
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    context_window: int = 3
    min_similarity_score: float = 0.5

class RAGService:
    """RAG service for conversational interface with YouTube content."""
    
    def __init__(self):
        """Initialize RAG service."""
        # Get configuration
        config = get_config()
        prompts = get_prompts()
        
        # Validate API key
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables!")
        
        # Create RAG config
        self.config = RAGConfig(
            model_name=config.model_name,
            api_key=config.gemini_api_key,
            context_window=getattr(config, 'retrieval_k', 3),
            min_similarity_score=getattr(config, 'similarity_threshold', 0.5)
        )
        
        # Initialize Gemini AI
        genai.configure(api_key=self.config.api_key)
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.model_name,
            google_api_key=self.config.api_key,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens
        )
        
        # Initialize vector service
        self.vector_service = VectorService()
        self.vector_service.initialize_vector_store()
        
        # Get prompts from prompts.yaml file only
        prompts = get_prompts()
        
        # Debug: show what prompts are available
        print(f"Available prompts in config: {list(prompts.keys())}")
        
        # Load prompts from YAML - fail if not found
        if not prompts:
            raise ValueError("No prompts found in config/prompts.yaml file!")
        
        # Load required prompts
        self.system_prompt = prompts.get('system_prompt')
        self.rag_prompt = prompts.get('rag_prompt') 
        self.no_context_prompt = prompts.get('no_context_prompt')
        
        # Validate required prompts exist
        if not self.rag_prompt:
            raise ValueError("rag_prompt not found in config/prompts.yaml!")
        
        if not self.no_context_prompt:
            raise ValueError("no_context_prompt not found in config/prompts.yaml!")
        
        print("RAG service initialized with Gemini AI!")
        print(f"Using prompts from config/prompts.yaml: {len(prompts)} prompts loaded")
    
    def build_context(self, search_results: List[SearchResult]) -> str:
        """Build context string from search results."""
        if not search_results:
            return ""
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_part = f"""
KAYNAK {i}:
Video: {result.video_title}
URL: {result.video_url}
İçerik: {result.text_content}
Benzerlik Skoru: {result.similarity_score:.3f}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_response(self, query: str) -> RAGResponse:
        """Generate RAG response for user query."""
        try:
            # Step 1: Search for relevant content
            print(f"Searching for: '{query}'")
            search_results = self.vector_service.search(
                query=query, 
                top_k=self.config.context_window
            )
            
            # Filter by similarity threshold
            filtered_results = [
                result for result in search_results 
                if result.similarity_score >= self.config.min_similarity_score
            ]
            
            print(f"Found {len(search_results)} results, {len(filtered_results)} above threshold")
            
            # Step 2: Handle no relevant content
            if not filtered_results:
                no_context_response = self.no_context_prompt.format(question=query)
                
                return RAGResponse(
                    query=query,
                    answer=no_context_response,
                    sources=[],
                    confidence_score=0.0
                )
            
            # Step 3: Build context
            context = self.build_context(filtered_results)
            
            # Step 4: Create prompt using loaded template
            prompt = self.rag_prompt.format(
                context=context,
                question=query
            )
            
            # Step 5: Generate response with Gemini
            print("Generating response with Gemini AI...")
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Step 6: Calculate confidence score (simple average)
            confidence_score = sum(r.similarity_score for r in filtered_results) / len(filtered_results)
            
            print("Response generated successfully")
            
            return RAGResponse(
                query=query,
                answer=answer,
                sources=filtered_results,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            error_response = f"Üzgünüm, yanıt oluştururken bir hata oluştu: {str(e)}"
            
            return RAGResponse(
                query=query,
                answer=error_response,
                sources=[],
                confidence_score=0.0
            )
    
def main():
    """Simple test of RAG service."""
    print("=== YouTube RAG Service Test ===")
    
    try:
        # Initialize service
        print("Initializing RAG Service...")
        rag_service = RAGService()
        
        # Test single query
        test_query = "Liderlik becerileri nasıl geliştirilir?"
        print(f"\nTesting query: '{test_query}'")
        
        response = rag_service.generate_response(test_query)
        
        print(f"\nAnswer: {response.answer}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Sources: {len(response.sources)} videos")
        
        if response.sources:
            print("\nSource videos:")
            for source in response.sources:
                print(f"  - {source.video_title} (Score: {source.similarity_score:.3f})")
        
        print("\nRAG service test completed!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()