# src/services/rag_service.py
"""
Professional RAG Service with LLM-based confidence evaluation.
Production-ready implementation for portfolio demonstration.
"""

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
    """RAG service configuration."""
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    min_similarity_threshold: float = 0.3
    search_top_k: int = 5
    # LLM confidence thresholds
    high_confidence_threshold: float = 0.75
    medium_confidence_threshold: float = 0.50

class RAGService:
    """
    Professional RAG service with LLM-based confidence evaluation.
    
    Features:
    - Vector-based semantic search
    - LLM confidence scoring
    - Intelligent web search fallback
    - Language matching validation
    - Professional error handling
    """
    
    def __init__(self):
        """Initialize RAG service with all components."""
        print("Initializing RAG Service with LLM Confidence...")
        
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
        
        # Prompt templates for response generation
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
        
        print("RAG Service initialized successfully")
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Turkish or English."""
        # Turkish-specific characters
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        
        # Turkish language indicators
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
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Check for Turkish characters
        if any(char in text for char in turkish_chars):
            return 'turkish'
        
        # Count Turkish words and suffixes
        turkish_score = 0
        for word in words:
            if word in turkish_words:
                turkish_score += 2
            elif any(word.endswith(suffix) for suffix in ['ler', 'lar', 'dir', 'dır', 'miş', 'muş', 'lik', 'lık']):
                turkish_score += 1
        
        # Check for Turkish question patterns
        if re.search(r'\b(nasıl|neden|ne|nedir|hangi|kim|nerede|ne zaman)\b', text_lower):
            turkish_score += 3
        
        return 'turkish' if turkish_score > 0 else 'english'
    
    def evaluate_response_quality(self, query: str, response: str, language: str) -> float:
        """Evaluate response quality using LLM."""
        try:
            if language == 'turkish':
                eval_prompt = f"""Bu RAG yanıtının kalitesini değerlendir:

Sorgu: {query}
Yanıt: {response}

Bu yanıt kullanıcının sorusunu ne kadar iyi yanıtlıyor? Alakalı, doğru, tam ve anlaşılır mı?

0.0 (çok kötü) ile 1.0 (mükemmel) arasında bir puan ver.

Sadece sayı ver:"""
            else:
                eval_prompt = f"""Evaluate the quality of this RAG response:

Query: {query}
Response: {response}

How well does this response answer the user's question? Consider relevance, accuracy, completeness, and clarity.

Rate from 0.0 (very poor) to 1.0 (excellent).

Return only the number:"""
            
            eval_response = self.llm.invoke(eval_prompt)
            confidence_text = eval_response.content.strip()
            
            # Extract confidence score
            numbers = re.findall(r'0\.\d+|1\.0|0\.0', confidence_text)
            if numbers:
                confidence = float(numbers[0])
                return max(0.0, min(1.0, confidence))
            
            return 0.5  # Fallback
            
        except Exception as e:
            print(f"Error evaluating response quality: {e}")
            return 0.5
    
    def generate_response(self, query: str) -> RAGResponse:
        """
        Generate response using RAG with LLM confidence evaluation.
        
        Flow:
        1. Vector search for relevant content
        2. Language compatibility check
        3. Generate RAG response
        4. Evaluate response quality with LLM
        5. Return high-quality response or fallback to web search
        """
        try:
            # Detect query language
            query_language = self.detect_language(query)
            
            # Get best YouTube content
            youtube_result = self._get_youtube_content(query)
            
            if not youtube_result:
                return self._web_search_fallback(query, query_language)
            
            # Check language compatibility
            content_language = self.detect_language(youtube_result.text_content[:500])
            if query_language != content_language:
                return self._web_search_fallback(query, query_language)
            
            # Check similarity threshold
            if youtube_result.similarity_score < self.config.min_similarity_threshold:
                return self._web_search_fallback(query, query_language)
            
            # Generate RAG response
            rag_answer = self._generate_youtube_answer(query, youtube_result, query_language)
            
            # Evaluate response quality using LLM
            llm_confidence = self.evaluate_response_quality(query, rag_answer, query_language)
            
            # Decision based on LLM confidence
            if llm_confidence >= self.config.medium_confidence_threshold:
                # High or medium confidence - use RAG response
                return RAGResponse(
                    query=query,
                    answer=rag_answer,
                    sources=[youtube_result],
                    confidence_score=llm_confidence
                )
            else:
                # Low confidence - fallback to web search
                return self._web_search_fallback(query, query_language)
            
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
        """Get best YouTube content from vector search."""
        try:
            search_results = self.vector_service.search(query, top_k=self.config.search_top_k)
            return search_results[0] if search_results else None
        except Exception as e:
            print(f"YouTube search error: {e}")
            return None
    
    def _generate_youtube_answer(self, question: str, video: SearchResult, language: str) -> str:
        """Generate answer from YouTube content."""
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
        """Generate answer from web content."""
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
    
    def _web_search_fallback(self, query: str, language: str) -> RAGResponse:
        """Fallback to web search when RAG confidence is low."""
        try:
            web_result = self.web_search_service.search(query)
            
            if web_result:
                web_answer = self._generate_web_answer(query, web_result.snippet, language)
                
                # Evaluate web response quality too
                web_confidence = self.evaluate_response_quality(query, web_answer, language)
                
                web_search_result = SearchResult(
                    video_id="web_search",
                    video_title=web_result.title,
                    video_url=web_result.url,
                    text_content=web_result.snippet,
                    similarity_score=web_confidence
                )
                
                return RAGResponse(
                    query=query,
                    answer=web_answer,
                    sources=[web_search_result],
                    confidence_score=web_confidence
                )
            
            # No content found anywhere
            no_content_message = (
                "Bu soruya yanıt verebilmek için hem video arşivimde hem de web'de yeterli bilgi bulunamadı." 
                if language == 'turkish' 
                else "Insufficient information found in both video archive and web search."
            )
            
            return RAGResponse(
                query=query,
                answer=no_content_message,
                sources=[],
                confidence_score=0.0
            )
            
        except Exception as e:
            print(f"Web search fallback error: {e}")
            return RAGResponse(
                query=query,
                answer=f"Web search failed: {str(e)}",
                sources=[],
                confidence_score=0.0
            )