# src/services/vector_service.py
"""Vector service using LangChain and Qdrant."""

import json
from typing import List
from pathlib import Path
from dataclasses import dataclass
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

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
    """Professional vector service for YouTube RAG."""
    
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
        self.qdrant_client = None
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store - use existing or create new."""
        try:
            # Setup paths
            vector_db_path = Path(self.config.vector_db_path)
            vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Create client only once
            if self.qdrant_client is None:
                self.qdrant_client = QdrantClient(path=str(vector_db_path))
            
            # Check if vector DB exists
            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                
                if self.config.collection_name in collection_names:
                    # Use existing vector DB
                    self.vector_store = Qdrant(
                        client=self.qdrant_client,
                        collection_name=self.config.collection_name,
                        embeddings=self.embeddings
                    )
                    self.vector_store.similarity_search("test", k=1)
                    return True
                    
            except Exception:
                pass
            
            # Create new vector DB
            with open(self.config.transcripts_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
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
            
            chunks = self.text_splitter.split_documents(documents)
            
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                client=self.qdrant_client,
                collection_name=self.config.collection_name
            )
            
            return True
            
        except Exception:
            return False
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search for relevant content."""
        if not self.vector_store:
            return []
        
        try:
            k = top_k or self.config.retrieval_k
            
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query, k=k
            )
            
            results = []
            for doc, score in docs_with_scores:
                similarity_score = 1.0 / (1.0 + abs(score))
                
                result = SearchResult(
                    video_id=doc.metadata['video_id'],
                    video_title=doc.metadata['video_title'],
                    video_url=doc.metadata['video_url'],
                    text_content=doc.page_content,
                    similarity_score=similarity_score
                )
                results.append(result)
            
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results
            
        except Exception:
            return []