# src/services/rag_service.py
"""RAG service with LLM confidence evaluation."""

from typing import Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import os
import re

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Add src to path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

from core.models import SearchResult, RAGResponse
from core.config import get_config
from services.vector_service import VectorService
from services.web_search_service import WebSearchService

@dataclass
class RAGConfig:
    """RAG service configuration."""
    model_name: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1024
    confidence_threshold: float = 0.3

class RAGService:
    """
    Simplified RAG service with LLM confidence evaluation.
    
    Flow:
    1. Get single best YouTube content
    2. Generate answer from content
    3. LLM evaluates answer quality
    4. If confidence >= 0.3 → use answer
    5. If confidence < 0.3 → web search fallback
    """
    
    def __init__(self):
        """Initialize RAG service."""
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
        
        # Prompt templates
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
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Turkish or English."""
        # Turkish-specific characters
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        
        # Turkish words
        turkish_words = {
            'nedir', 'nasıl', 'neden', 'hangi', 'için', 'olan', 'bu', 'bir', 
            'de', 'da', 'ile', 've', 'veya', 'çünkü', 'şey', 'zaman', 'mi', 
            'mı', 'mu', 'mü', 'ne', 'kim', 'nerede', 'önemli', 'söz'
        }
        
        text_lower = text.lower()
        
        # Check for Turkish characters
        if any(char in text for char in turkish_chars):
            return 'turkish'
        
        # Count Turkish words
        words = re.findall(r'\b\w+\b', text_lower)
        turkish_score = sum(1 for word in words if word in turkish_words)
        
        return 'turkish' if turkish_score > 0 else 'english'
    
    def evaluate_response_quality(self, query: str, response: str, language: str) -> float:
        """Evaluate response quality using LLM."""
        try:
            if language == 'turkish':
                eval_prompt = f"""Bu RAG yanıtını değerlendir:

Sorgu: {query}
Yanıt: {response}

Bu yanıt kullanıcının sorusunu gerçekten yanıtlıyor mu?

ÖNEMLI KURALLAR:
- Eğer yanıt "bulunamadı", "bilgi yok", "içerikte yer almıyor", "bahsedilmiyor" gibi olumsuz ifadeler içeriyorsa → 0.0 ver
- Eğer yanıt soruyla alakasız konulardan bahsediyorsa → 0.0 ver  
- Eğer yanıt soruyu doğrudan ve faydalı şekilde yanıtlıyorsa → 0.7-1.0 arasında ver
- Eğer yanıt kısmen faydalıysa → 0.3-0.6 arasında ver

0.0 (alakasız/bulunamadı) ile 1.0 (mükemmel yanıt) arasında puan ver.

Sadece sayı ver:"""
            else:
                eval_prompt = f"""Evaluate this RAG response:

Query: {query}
Response: {response}

Does this response actually answer the user's question?

IMPORTANT RULES:
- If response says "not found", "no information", "not mentioned", "not available" etc. → give 0.0
- If response talks about irrelevant topics instead of answering → give 0.0
- If response directly and helpfully answers the question → give 0.7-1.0
- If response is partially helpful → give 0.3-0.6

Rate from 0.0 (irrelevant/not found) to 1.0 (excellent answer).

Return only the number:"""
            
            eval_response = self.llm.invoke(eval_prompt)
            confidence_text = eval_response.content.strip()
            
            # Extract confidence score
            numbers = re.findall(r'0\.\d+|1\.0|0\.0', confidence_text)
            
            if numbers:
                confidence = float(numbers[0])
                return max(0.0, min(1.0, confidence))
            
            # Keyword fallback
            negative_keywords_tr = ['bulunamadı', 'bulunmamaktadır', 'yer almıyor', 'bahsedilmiyor', 'bilgi yok']
            negative_keywords_en = ['not found', 'no information', 'not mentioned', 'not available']
            
            keywords = negative_keywords_tr if language == 'turkish' else negative_keywords_en
            
            if any(keyword in response.lower() for keyword in keywords):
                return 0.0
            
            return 0.5  # Fallback
            
        except Exception:
            # Emergency keyword check
            if any(word in response.lower() for word in ['bulunamadı', 'not found']):
                return 0.0
            return 0.5

    def generate_response(self, query: str) -> RAGResponse:
        """Generate response using simplified RAG flow."""
        try:
            # Detect language
            query_language = self.detect_language(query)
            
            # Get single best YouTube content
            youtube_result = self._get_best_youtube_content(query)
            
            if not youtube_result:
                return self._web_search_fallback(query, query_language)
            
            # Check language compatibility
            content_language = self.detect_language(youtube_result.text_content[:500])
            
            if query_language != content_language:
                return self._web_search_fallback(query, query_language)
            
            # Generate answer from YouTube content
            rag_answer = self._generate_youtube_answer(query, youtube_result, query_language)
            
            # Evaluate answer quality with LLM
            llm_confidence = self.evaluate_response_quality(query, rag_answer, query_language)
            
            # Decision based on LLM confidence
            if llm_confidence >= self.config.confidence_threshold:
                # High confidence - use YouTube answer
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
    
    def _get_best_youtube_content(self, query: str) -> Optional[SearchResult]:
        """Get single best YouTube content from vector search."""
        try:
            # Get only the top 1 result (best match)
            search_results = self.vector_service.search(query, top_k=1)
            return search_results[0] if search_results else None
        except Exception:
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
        except Exception:
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
        except Exception:
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
                
                # Evaluate web response quality
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
            
        except Exception:
            return RAGResponse(
                query=query,
                answer="Web search failed",
                sources=[],
                confidence_score=0.0
            )