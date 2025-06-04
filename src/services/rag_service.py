# src/services/rag_service.py
"""RAG Service with improved language detection and dynamic confidence scoring."""

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
    min_similarity_threshold: float = 0.4  # Lowered back to reasonable threshold
    search_top_k: int = 5

class RAGService:
    """RAG service with web search fallback and improved language detection."""
    
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
        
        # Prompt templates with proper language handling
        self.prompts = {
            'turkish': {
                'youtube': """Sen, yalnızca aşağıdaki içerikten yola çıkarak, kullanıcının sorusunu Türkçe ve profesyonel bir dille yanıtlayan bir yapay zekâ asistansın.

İçerik:
{video_content}

Soru:
{question}

Kurallar:
- Yanıtı tamamen Türkçe yaz
- 'Ben', 'bana', 'bizi', 'bence' gibi ifadeleri kullanma
- 'Videoda', 'video içeriğine göre' gibi ifadeler kullanma
- Bilgiyi tarafsız, profesyonel ve akademik bir dilde aktar
- Cevap 4–6 cümle uzunluğunda olsun
- Gereksiz süsleme yapma, net ol

Yanıt:""",
                'web': """Sen, yalnızca aşağıdaki web araması sonucundan yola çıkarak, kullanıcının sorusunu Türkçe ve profesyonel bir dille yanıtlayan bir yapay zekâ asistansın.

Web Araması Sonucu:
{web_content}

Soru:
{question}

Kurallar:
- Yanıtı tamamen Türkçe yaz
- 'Ben', 'bana', 'bizi', 'bence' gibi ifadeleri kullanma
- 'Web'de', 'araştırmaya göre' gibi ifadeler kullanma
- Bilgiyi tarafsız, profesyonel ve akademik bir dilde aktar
- Cevap 4–6 cümle uzunluğunda olsun
- Güncel bilgileri vurgula

Yanıt:"""
            },
            'english': {
                'youtube': """You are an AI assistant that answers user questions professionally in English, based solely on the provided content.

Content:
{video_content}

Question:
{question}

Rules:
- Answer entirely in English
- Don't use phrases like 'I', 'me', 'us', 'in my opinion'
- Don't use phrases like 'in the video', 'according to the video'
- Present information neutrally and professionally
- Keep answer 4-6 sentences long
- Be direct and clear

Answer:""",
                'web': """You are an AI assistant that answers user questions professionally in English, based solely on the provided web search results.

Web Search Results:
{web_content}

Question:
{question}

Rules:
- Answer entirely in English
- Don't use phrases like 'I', 'me', 'us', 'in my opinion'
- Don't use phrases like 'according to web search', 'the search shows'
- Present information neutrally and professionally
- Keep answer 4-6 sentences long
- Emphasize current information

Answer:"""
            }
        }
        
        print("RAG Service with Web Fallback initialized successfully")
    
    def detect_language(self, text: str) -> str:
        """Enhanced language detection for Turkish vs English."""
        # Turkish-specific characters
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        
        # Common Turkish words (expanded list)
        turkish_words = {
            'nedir', 'nasıl', 'neden', 'hangi', 'kimse', 'hiç', 'için', 'olan',
            'bu', 'bir', 'de', 'da', 'ile', 've', 'veya', 'ama', 'fakat',
            'çünkü', 'belki', 'her', 'bazı', 'tüm', 'bütün', 'şey', 'zaman',
            'yer', 'gün', 'yıl', 'kişi', 'insan', 'iyi', 'kötü', 'büyük',
            'küçük', 'yeni', 'eski', 'var', 'yok', 'et', 'ol', 'yap', 'gel',
            'git', 'al', 'ver', 'gör', 'bil', 'iste', 'söyle', 'çalış',
            'yaşa', 'öğren', 'anla', 'düşün', 'inan', 'hakkında', 'üzerine',
            'karşı', 'doğru', 'göre', 'kadar', 'önce', 'sonra', 'şimdi',
            'daha', 'çok', 'az', 'en', 'mi', 'mı', 'mu', 'mü', 'misin',
            'musun', 'neler', 'ne', 'kim', 'nerede', 'ne zaman', 'niçin',
            'merhaba', 'günaydın', 'iyi günler', 'teşekkür', 'lütfen'
        }
        
        # Common English words
        english_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us', 'is', 'are', 'been',
            'has', 'had', 'were', 'was', 'latest', 'best', 'top', 'why',
            'where', 'what', 'when', 'how', 'hello', 'hi', 'thanks'
        }
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Check for Turkish characters
        if any(char in text for char in turkish_chars):
            return 'turkish'
        
        # Count language indicators
        turkish_score = 0
        english_score = 0
        
        for word in words:
            if word in turkish_words:
                turkish_score += 2  # Give more weight to exact matches
            elif word in english_words:
                english_score += 2
            # Check for Turkish suffixes
            elif any(word.endswith(suffix) for suffix in ['ler', 'lar', 'dir', 'dır', 'miş', 'muş', 'lik', 'lık']):
                turkish_score += 1
        
        # Check for question patterns
        if re.search(r'\b(nasıl|neden|ne|nedir|hangi|kim|nerede|ne zaman)\b', text_lower):
            turkish_score += 3
        if re.search(r'\b(what|why|how|which|who|where|when)\b', text_lower):
            english_score += 3
        
        # Make decision
        if turkish_score > english_score:
            return 'turkish'
        elif english_score > turkish_score:
            return 'english'
        else:
            # Default to English for ambiguous cases
            return 'english'
    
    def generate_response(self, query: str) -> RAGResponse:
        """Generate response with proper language handling."""
        try:
            print(f"\nProcessing query: {query}")
            
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
                
                # Simple confidence for web results
                # Higher confidence if we got a real URL, lower if it's a search page
                if "google.com/search" in web_result.url or "duckduckgo.com" in web_result.url:
                    confidence = 0.3  # Low confidence for search pages
                else:
                    confidence = 0.7  # Good confidence for actual articles
                
                # Convert web result to SearchResult format
                web_search_result = SearchResult(
                    video_id="web_search",
                    video_title=web_result.title,
                    video_url=web_result.url,
                    text_content=web_result.snippet,
                    similarity_score=confidence
                )
                
                return RAGResponse(
                    query=query,
                    answer=answer,
                    sources=[web_search_result],
                    confidence_score=confidence
                )
            
            # Step 4: No content found anywhere
            no_content_message = (
                "Bu soruya yanıt verebilmek için hem video arşivimde hem de web'de yeterli bilgi bulunamadı." 
                if language == 'turkish' 
                else "Insufficient information found in both video archive and web search to answer this question."
            )
            
            return RAGResponse(
                query=query,
                answer=no_content_message,
                sources=[],
                confidence_score=0.0
            )
            
        except Exception as e:
            print(f"Error generating response: {e}")
            language = self.detect_language(query)
            error_message = (
                f"Yanıt oluşturulurken hata oluştu: {str(e)}" 
                if language == 'turkish' 
                else f"Error generating response: {str(e)}"
            )
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
        """Generate answer from YouTube content in the detected language."""
        try:
            prompt = self.prompts[language]['youtube'].format(
                video_content=video.text_content,
                question=question
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating YouTube answer: {e}")
            return (
                "YouTube içeriği işlenirken hata oluştu." 
                if language == 'turkish' 
                else "Error processing YouTube content."
            )
    
    def _generate_web_answer(self, question: str, web_content: str, language: str) -> str:
        """Generate answer from web content in the detected language."""
        try:
            prompt = self.prompts[language]['web'].format(
                web_content=web_content,
                question=question
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"Error generating web answer: {e}")
            return (
                "Web içeriği işlenirken hata oluştu." 
                if language == 'turkish' 
                else "Error processing web content."
            )