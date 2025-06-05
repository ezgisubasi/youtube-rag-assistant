# src/services/vector_service.py
"""Vector service using LangChain and Qdrant."""

import json
import shutil
from typing import List
from pathlib import Path
from dataclasses import dataclass
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Add src to path for imports
sys.path.append('src')

from core.models import SearchResult
from core.config import get_config

@dataclass
class VectorConfig:
    """Vector service configuration."""
    embedding_model: str
    vector_db_path: str
    collection_name: str
    transcripts_json: str
    retrieval_k: int
    chunk_size: int = 1000
    chunk_overlap: int = 200

class VectorService:
    """Colab-compatible vector service for YouTube RAG."""
    
    def __init__(self):
        """Initialize vector service."""
        config = get_config()
        
        self.config = VectorConfig(
            embedding_model=config.embedding_model,
            vector_db_path=config.vector_db_path,
            collection_name=config.collection_name,
            transcripts_json=config.transcripts_json,
            retrieval_k=config.retrieval_k
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.vector_store = None
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store - Colab compatible version."""
        try:
            # Setup path
            vector_db_path = Path(self.config.vector_db_path)
            vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Check if collection already exists using direct path check
            if vector_db_path.exists() and any(vector_db_path.iterdir()):
                # Try to load existing collection
                try:
                    self.vector_store = Qdrant(
                        path=str(vector_db_path),
                        collection_name=self.config.collection_name,
                        embeddings=self.embeddings
                    )
                    # Test if it works
                    self.vector_store.similarity_search("test", k=1)
                    return True
                except:
                    # If loading fails, recreate
                    shutil.rmtree(vector_db_path)
                    vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Create new vector store
            if not Path(self.config.transcripts_json).exists():
                return False
            
            with open(self.config.transcripts_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create documents
            documents = []
            for item in data:
                if item.get('video_text', '').strip():
                    doc = Document(
                        page_content=item['video_text'],
                        metadata={
                            'video_id': item['video_id'],
                            'video_title': item['video_title'],
                            'video_url': item['video_url']
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                return False
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Create vector store - COLAB COMPATIBLE METHOD
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                path=str(vector_db_path),  # Use path, not client
                collection_name=self.config.collection_name
            )
            
            return True
            
        except Exception:
            return False
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search for relevant content."""
        if not self.vector_store:
            self.initialize_vector_store()
        
        try:
            k = top_k or self.config.retrieval_k
            
            # Get results with scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query, k=k
            )
            
            # Convert to SearchResult objects
            results = []
            for doc, score in docs_with_scores:
                # Convert distance to similarity
                similarity_score = 1.0 / (1.0 + abs(score))
                
                result = SearchResult(
                    video_id=doc.metadata['video_id'],
                    video_title=doc.metadata['video_title'],
                    video_url=doc.metadata['video_url'],
                    text_content=doc.page_content,
                    similarity_score=similarity_score
                )
                results.append(result)
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results
            
        except Exception:
            return []