# Updated src/services/rag_service.py - Fixed Language + Threshold Issues

from typing import List, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import os
import re

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.models import SearchResult, RAGResponse
from core.config import get_config, get_prompts
from services.vector_service import VectorService
from services.web_search_service import WebSearchService

@dataclass
class RAGConfig:
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    min_similarity_threshold: float = 0.8  # RAISED to force web search for testing
    search_top_k: int = 5

class RAGService:
    """RAG service with web search fallback and language detection."""
    
    def __init__(self):
        print("Initializing RAG Service with Web Fallback...")
        
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
        
        # Initialize services
        self.vector_service = VectorService()
        self.vector_service.initialize_vector_store()
        self.web_search_service = WebSearchService()
        
        # Turkish YouTube prompt
        self.turkish_youtube_prompt = """Sen, yalnızca aşağıdaki içerikten yola çıkarak, kullanıcının sorusunu Türkçe ve profesyonel bir dille yanıtlayan bir yapay zekâ asistansın.

İçerik:
{video_content}

Soru:
{question}

Kurallar:
- 'Ben', 'bana', 'bizi', 'bence' gibi ifadeleri **kullanma**
- 'Videoda', 'video içeriğine göre' gibi ifadeler **kullanma**
- Bilgiyi tarafsız, profesyonel ve akademik bir dilde aktar
- Cevap 4–6 cümle uzunluğunda olsun
- Gereksiz süsleme yapma, net ol
- Bilgiyi sen yazıyormuşsun gibi değil; konunun uzmanı tarafsız biri anlatıyormuş gibi yaz

Yanıt:"""

        # English YouTube prompt
        self.english_youtube_prompt = """You are an AI assistant that answers user questions professionally in English, based solely on the provided content.

Content:
{video_content}

Question:
{question}

Rules:
- Don't use phrases like 'I', 'me', 'us', 'in my opinion'
- Don't use phrases like 'in the video', 'according to the video'
- Present information neutrally and professionally
- Keep answer 4-6 sentences long
- Be direct and clear
- Write as if you're an expert presenting facts, not referencing a video

Answer:"""

        # Turkish web search prompt
        self.turkish_web_prompt = """Sen, yalnızca aşağıdaki web araması sonucundan yola çıkarak, kullanıcının sorusunu Türkçe ve profesyonel bir dille yanıtlayan bir yapay zekâ asistansın.

Web Araması Sonucu:
{web_content}

Soru:
{question}

Kurallar:
- 'Ben', 'bana', 'bizi', 'bence' gibi ifadeleri **kullanma**
- 'Web'de', 'araştırmaya göre' gibi ifadeler **kullanma**
- Bilgiyi tarafsız, profesyonel ve akademik bir dilde aktar
- Cevap 4–6 cümle uzunluğunda olsun
- Gereksiz süsleme yapma, net ol
- Bilgiyi sen yazıyormuşsun gibi değil; konunun uzmanı tarafsız biri anlatıyormuş gibi yaz

Yanıt:"""

        # English web search prompt
        self.english_web_prompt = """You are an AI assistant that answers user questions professionally in English, based solely on the provided web search results.

Web Search Results:
{web_content}

Question:
{question}

Rules:
- Don't use phrases like 'I', 'me', 'us', 'in my opinion'
- Don't use phrases like 'according to web search', 'the search shows'
- Present information neutrally and professionally
- Keep answer 4-6 sentences long
- Be direct and clear
- Write as if you're an expert presenting current facts

Answer:"""
        
        print("RAG Service with Web Fallback initialized successfully")
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Turkish or English."""
        turkish_chars = 'çğıöşüÇĞIİÖŞÜ'
        turkish_words = ['nedir', 'nasıl', 'neden', 'hangi', 'kimse', 'hiç', 'için', 'olan', 'olan', 'bu', 'bir', 'de', 'da', 'ile']
        
        # Check for Turkish characters
        if any(char in text for char in turkish_chars):
            return 'turkish'
        
        # Check for Turkish words
        text_lower = text.lower()
        if any(word in text_lower for word in turkish_words):
            return 'turkish'
        
        # Default to English if no Turkish indicators
        return 'english'
    
    def generate_response(self, query: str) -> RAGResponse:
        """Generate response: YouTube first, then web fallback."""
        try:
            print(f"Processing query: {query}")
            
            # Detect language
            language = self.detect_language(query)
            print(f"Detected language: {language}")
            
            # Step 1: Try YouTube first
            youtube_result = self._get_youtube_content(query)
            
            # Step 2: Check if YouTube result is good enough
            if youtube_result and youtube_result.similarity_score >= self.config.min_similarity_threshold:
                print(f"Using YouTube (confidence: {youtube_result.similarity_score:.3f})")
                answer = self._generate_youtube_answer(query, youtube_result, language)
                
                return RAGResponse(
                    query=query,
                    answer=answer,
                    sources=[youtube_result],
                    confidence_score=youtube_result.similarity_score
                )
            
            # Step 3: Fallback to web search
            if youtube_result:
                print(f"YouTube confidence too low ({youtube_result.similarity_score:.3f} < {self.config.min_similarity_threshold}), using web search")
            else:
                print("No YouTube results found, using web search")
                
            web_result = self._get_web_content(query)
            
            if web_result:
                answer = self._generate_web_answer(query, web_result.snippet, language)
                
                # Convert web result to SearchResult format
                web_search_result = SearchResult(
                    video_id="web_search",
                    video_title=web_result.title,
                    video_url=web_result.url,
                    text_content=web_result.snippet,
                    similarity_score=0.6  # Standard confidence for web results
                )
                
                return RAGResponse(
                    query=query,
                    answer=answer,
                    sources=[web_search_result],
                    confidence_score=0.6
                )
            
            # Step 4: No content found anywhere
            no_content_message = "Bu soruya yanıt verebilmek için hem video arşivimde hem de web'de yeterli bilgi bulunamadı." if language == 'turkish' else "Insufficient information found in both video archive and web search to answer this question."
            
            return RAGResponse(
                query=query,
                answer=no_content_message,
                sources=[],
                confidence_score=0.0
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            error_message = f"Yanıt oluşturulurken hata oluştu: {str(e)}" if language == 'turkish' else f"Error generating response: {str(e)}"
            return RAGResponse(
                query=query,
                answer=error_message,
                sources=[],
                confidence_score=0.0
            )
    
    def _get_youtube_content(self, query: str) -> Optional[SearchResult]:
        """Get best YouTube content."""
        try:
            search_results = self.vector_service.search(query, top_k=self.config.search_top_k)
            if search_results:
                best = search_results[0]
                print(f"YouTube result: {best.video_title} (confidence: {best.similarity_score:.3f})")
                return best
            return None
        except Exception as e:
            print(f"YouTube search error: {e}")
            return None
    
    def _get_web_content(self, query: str):
        """Get web search content."""
        try:
            return self.web_search_service.search(query)
        except Exception as e:
            print(f"Web search error: {e}")
            return None
    
    def _generate_youtube_answer(self, question: str, video: SearchResult, language: str) -> str:
        """Generate answer from YouTube content."""
        try:
            if language == 'turkish':
                prompt = self.turkish_youtube_prompt.format(
                    video_content=video.text_content,
                    question=question
                )
            else:
                prompt = self.english_youtube_prompt.format(
                    video_content=video.text_content,
                    question=question
                )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating YouTube answer: {e}")
            return "YouTube içeriği işlenirken hata oluştu." if language == 'turkish' else "Error processing YouTube content."
    
    def _generate_web_answer(self, question: str, web_content: str, language: str) -> str:
        """Generate answer from web content."""
        try:
            if language == 'turkish':
                prompt = self.turkish_web_prompt.format(
                    web_content=web_content,
                    question=question
                )
            else:
                prompt = self.english_web_prompt.format(
                    web_content=web_content,
                    question=question
                )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating web answer: {e}")
            return "Web içeriği işlenirken hata oluştu." if language == 'turkish' else "Error processing web content."