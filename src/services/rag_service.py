# src/services/rag_service.py
from typing import List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import os

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.models import SearchResult, RAGResponse
from core.config import get_config, get_prompts
from services.vector_service import VectorService

@dataclass
class RAGConfig:
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    min_similarity_threshold: float = 0.25
    search_top_k: int = 5

class RAGService:
    """Clean RAG service for YouTube content."""
    
    def __init__(self):
        print("Initializing RAG Service...")
        
        # Load configuration
        config = get_config()
        gemini_api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found")
        
        # Create RAG configuration
        self.config = RAGConfig(
            model_name=config.model_name or "gemini-1.5-flash",
            api_key=gemini_api_key
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
        
        # Simple prompt template
        self.answer_prompt = """Sen bir YouTube video içeriklerine dayanan AI asistansın. 
Görevin, kullanıcının sorusunu yalnızca video içeriğine dayanarak ve özlü şekilde yanıtlamak.

Aşağıdaki video içeriğine dayanarak soruyu Türkçe ve kısa cevapla:
Video İçeriği: {video_content}

Soru: {question}

Talimatlar:
- Sadece video içeriğinde yer alan bilgilere dayan
- Gereksiz detaylardan kaçın, cevabı kısa ve net tut
- Maddeler hâlinde yazmak yerine 1-2 paragraflık sade bir açıklama yap
- Profesyonel ama samimi bir dil kullan

Yanıt:"""

        
        print("RAG Service initialized successfully")
    
    def get_best_video(self, query: str) -> Optional[SearchResult]:
        """Get the most relevant video."""
        try:
            search_results = self.vector_service.search(
                query=query,
                top_k=self.config.search_top_k
            )
            
            if not search_results:
                return None
            
            # Filter results above minimum threshold
            filtered_results = [
                result for result in search_results
                if result.similarity_score >= self.config.min_similarity_threshold
            ]
            
            # Return best result
            best_video = filtered_results[0] if filtered_results else search_results[0]
            
            print(f"Selected video: {best_video.video_title}")
            print(f"Confidence score: {best_video.similarity_score:.3f}")
            
            return best_video
            
        except Exception as e:
            print(f"Error finding best video: {e}")
            return None
    
    def generate_answer(self, question: str, video: SearchResult) -> str:
        """Generate answer based on video content."""
        try:
            prompt = self.answer_prompt.format(
                video_content=video.text_content,
                question=question
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "Yanıt oluşturulurken bir hata oluştu."
    
    def generate_response(self, query: str) -> RAGResponse:
        """Generate complete response."""
        try:
            print(f"Processing query: {query}")
            
            # Get best matching video
            best_video = self.get_best_video(query)
            
            if not best_video:
                return RAGResponse(
                    query=query,
                    answer="Bu soruya yanıt verebilmek için uygun video içeriği bulunamadı.",
                    sources=[],
                    confidence_score=0.0
                )
            
            # Generate answer
            answer = self.generate_answer(query, best_video)
            
            # Return clean response with sources
            return RAGResponse(
                query=query,
                answer=answer,
                sources=[best_video],
                confidence_score=best_video.similarity_score
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return RAGResponse(
                query=query,
                answer=f"Yanıt oluşturulurken hata oluştu: {str(e)}",
                sources=[],
                confidence_score=0.0
            )