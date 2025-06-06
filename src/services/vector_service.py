# src/services/vector_service.py
"""Vector service using Qdrant and HuggingFace Embeddings based on transcripts."""

import json
import shutil
from typing import List
from pathlib import Path
from dataclasses import dataclass
import sys

from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# Add src to path for imports
sys.path.append('src')

from core.models import SearchResult
from core.config import get_config

@dataclass
class VectorConfig:
    """Vector service configuration for whole transcript processing."""
    embedding_model: str
    vector_db_path: str
    collection_name: str
    transcripts_json: str
    retrieval_k: int

class VectorService:
    """Vector service initialized with transcripts."""
    
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
        
        # No text splitter - use transcripts for complete context
        self.vector_store = None
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store using transcripts as documents."""
        try:
            # ADD DEBUGGING HERE:
            import os
            print(f"🔍 Current working directory: {os.getcwd()}")
            print(f"🔍 Vector DB path: {self.config.vector_db_path}")
            print(f"🔍 Transcripts JSON: {self.config.transcripts_json}")
        
            # Setup path
            vector_db_path = Path(self.config.vector_db_path)
            transcripts_path = Path(self.config.transcripts_json)
        
            print(f"🔍 Vector DB exists: {vector_db_path.exists()}")
            print(f"🔍 Transcripts exists: {transcripts_path.exists()}")
        
            if transcripts_path.exists():
                print(f"🔍 Transcripts file size: {transcripts_path.stat().st_size} bytes")
        
            # List what's actually in the directory
            print(f"🔍 Files in current dir: {list(Path('.').iterdir())}")
            if Path('data').exists():
                print(f"🔍 Files in data/: {list(Path('data').iterdir())}")
        
            vector_db_path.mkdir(parents=True, exist_ok=True)

            # Check if collection already exists
            if vector_db_path.exists() and any(vector_db_path.iterdir()):
                try:
                    self.vector_store = Qdrant(
                        path=str(vector_db_path),
                        collection_name=self.config.collection_name,
                        embeddings=self.embeddings
                    )
                    return True
                except:
                    # If loading fails, recreate
                    shutil.rmtree(vector_db_path)
                    vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Create new vector store from whole transcripts
            if not Path(self.config.transcripts_json).exists():
                return False
            
            with open(self.config.transcripts_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Create documents from transcripts (no chunking)
            documents = []
            for item in data:
                transcript = item.get('video_text', '').strip()
                if transcript:
                    # Each complete transcript becomes one document
                    doc = Document(
                        page_content=transcript,
                        metadata={
                            'video_id': item['video_id'],
                            'video_title': item['video_title'],
                            'video_url': item['video_url']
                        }
                    )
                    documents.append(doc)
            
            if not documents:
                return False
            
            # Create vector store with transcripts
            self.vector_store = Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                path=str(vector_db_path),
                collection_name=self.config.collection_name
            )
            
            return True
            
        except Exception:
            return False
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search with similarity scores as-is from Qdrant."""
        if not self.vector_store:
            self.initialize_vector_store()
        
        try:
            k = top_k or self.config.retrieval_k
            
            # Get results with similarity scores from Qdrant
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query, k=k
            )
            
            results = []
            for doc, similarity_score in docs_with_scores:
                # Use similarity score as-is (already in 0-1 range)
                result = SearchResult(
                    video_id=doc.metadata['video_id'],
                    video_title=doc.metadata['video_title'],
                    video_url=doc.metadata['video_url'],
                    text_content=doc.page_content,  # Complete transcript
                    similarity_score=similarity_score  # Keep as-is
                )
                results.append(result)
            
            # Sort by similarity (highest first)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results
            
        except Exception:
            return []