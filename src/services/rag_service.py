"""RAG service with dataclass structure and formatted output."""

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
    """RAG service configuration."""
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    min_similarity_threshold: float = 0.25
    search_top_k: int = 5

@dataclass
class FormattedResponse:
    """Formatted response structure."""
    answer: str
    video_title: str
    video_url: str
    confidence_score: float
    
    def format_output(self) -> str:
        """Format the complete response with source."""
        formatted_response = f"{self.answer}\\n\\n"
        formatted_response += f"**Source:** \"{self.video_title}\" {self.video_url} "
        formatted_response += f"Confidence Score: {self.confidence_score:.2f}"
        return formatted_response

class RAGService:
    """Clean RAG service with structured output."""
    
    def __init__(self):
        """Initialize RAG service."""
        print("Initializing RAG Service...")
        
        # Load configuration
        config = get_config()
        gemini_api_key = config.gemini_api_key or os.getenv("GEMINI_API_KEY")
        
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
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
        
        # Clean prompt template for answer generation
        self.answer_prompt = """TALİMAT: Aşağıdaki video içeriğine dayanarak kullanıcının sorusunu açık, yapılandırılmış ve Türkçe olarak yanıtla.

VIDEO İÇERİĞİ:
{video_content}

KULLANICI SORUSU:
{question}

GÖREVLER:

Yalnızca verilen video içeriğini kullanarak yanıt oluştur.
Yanıtı, video içeriğinden alınan bilgileri genel geçer bilgi gibi sunarak oluştur. "Videoya göre" veya "videodaki kişiye göre" gibi ifadelerden kaçın.
Net, akıcı ve yapılandırılmış bir metin yaz.
Gerekirse örnekler ve açıklayıcı detaylar ekle.
Türkçe yaz.
Video başlığı, bağlantısı ve güven skorunu yanıta dahil etme; bu bilgiler dışarıdan eklenecek.

YANITINIZ:"""
        
        print("RAG Service initialized successfully")
    
    def get_best_video(self, query: str) -> Optional[SearchResult]:
        """Get the most relevant video based on highest confidence score."""
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
            
            # Return best result or first available if none meet threshold
            best_video = filtered_results[0] if filtered_results else search_results[0]
            
            print(f"Selected video: {best_video.video_title}")
            print(f"Confidence score: {best_video.similarity_score:.3f}")
            
            return best_video
            
        except Exception as e:
            print(f"Error finding best video: {e}")
            return None
    
    def generate_structured_answer(self, question: str, video: SearchResult) -> str:
        """Generate structured answer based on video content."""
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
        """Generate complete response with formatted output."""
        try:
            print(f"Processing query: {query}")
            
            # Get best matching video
            best_video = self.get_best_video(query)
            
            if not best_video:
                return RAGResponse(
                    query=query,
                    answer="Bu soruya yanıt verebilmek için uygun video içeriği bulunamadı. Lütfen farklı kelimeler deneyin.",
                    sources=[],
                    confidence_score=0.0
                )
            
            # Generate structured answer
            answer = self.generate_structured_answer(query, best_video)
            
            # Create formatted response
            formatted_response = FormattedResponse(
                answer=answer,
                video_title=best_video.video_title,
                video_url=best_video.video_url,
                confidence_score=best_video.similarity_score
            )
            
            # Return RAG response with formatted output
            return RAGResponse(
                query=query,
                answer=formatted_response.format_output(),
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

def main():
    """Test function for RAG service."""
    print("Testing Clean RAG Service")
    print("=" * 40)
    
    try:
        service = RAGService()
        
        test_queries = [
            "Nasıl iyi lider olunur?",
            "Liderlik nasıl geliştirilir?",
            "Başarı faktörleri nelerdir?"
        ]
        
        for query in test_queries:
            print(f"\\nTesting query: {query}")
            print("-" * 30)
            
            response = service.generate_response(query)
            
            print(f"Confidence: {response.confidence_score:.3f}")
            print(f"Sources: {len(response.sources)}")
            print(f"Answer:\\n{response.answer}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
