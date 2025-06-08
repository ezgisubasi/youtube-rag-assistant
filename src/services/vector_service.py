# src/services/vector_service.py
"""Vector service using QdrantVectorStore and HuggingFace Embeddings based on transcripts."""

import json
from typing import List
from pathlib import Path
from dataclasses import dataclass
import sys

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from qdrant_client import QdrantClient

# Add path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

from core.models import SearchResult
from core.config import get_config

@dataclass
class VectorConfig:
    """Vector service configuration for transcript processing."""
    embedding_model: str
    vector_db_path: str
    collection_name: str
    transcripts_json: str
    retrieval_k: int

class VectorService:
    """Vector service using modern LangChain with Qdrant."""
    
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
        
        # Initialize embeddings with fallback
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'},
                show_progress=False
            )
        except Exception:
            # Fallback to smaller model
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                show_progress=False
            )
        
        self.vector_store = None
        self.qdrant_client = None
        self._initialization_attempted = False
        self._initialization_successful = False
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store using transcripts as documents."""
        # Check if already successfully initialized
        if self._initialization_successful and self.vector_store is not None:
            return True
        
        if self._initialization_attempted:
            return False
        
        self._initialization_attempted = True
        
        try:
            # Setup paths
            vector_db_path = Path(self.config.vector_db_path)
            transcripts_path = Path(self.config.transcripts_json)
        
            if not transcripts_path.exists():
                return False
        
            # Create vector DB directory
            vector_db_path.mkdir(parents=True, exist_ok=True)

            # Initialize Qdrant client
            if self.qdrant_client is None:
                try:
                    self.qdrant_client = QdrantClient(path=str(vector_db_path))
                except Exception:
                    return False

            # Check if collection exists
            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                collection_exists = self.config.collection_name in collection_names
            except Exception:
                collection_exists = False

            if collection_exists:
                # Load existing collection
                try:
                    self.vector_store = QdrantVectorStore(
                        client=self.qdrant_client,
                        collection_name=self.config.collection_name,
                        embedding=self.embeddings
                    )
                    
                    # Test the vector store
                    test_results = self.vector_store.similarity_search("test", k=1)
                    self._initialization_successful = True
                    return True
                    
                except Exception:
                    # Delete problematic collection and recreate
                    try:
                        self.qdrant_client.delete_collection(self.config.collection_name)
                    except:
                        pass
            
            # Read and process transcripts
            try:
                with open(transcripts_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                return False
            
            # Create documents
            documents = []
            for item in data:
                try:
                    transcript = item.get('video_text', '').strip()
                    if transcript:
                        doc = Document(
                            page_content=transcript,
                            metadata={
                                'video_id': item['video_id'],
                                'video_title': item['video_title'],
                                'video_url': item['video_url']
                            }
                        )
                        documents.append(doc)
                except Exception:
                    continue
            
            if not documents:
                return False
            
            # Create vector store
            try:
                self.vector_store = QdrantVectorStore(
                    client=self.qdrant_client,
                    collection_name=self.config.collection_name,
                    embedding=self.embeddings
                )
                
                # Add documents to the vector store
                self.vector_store.add_documents(documents)
                
                # Test the vector store
                test_results = self.vector_store.similarity_search("test", k=1)
                
                self._initialization_successful = True
                return True
                
            except Exception:
                # Try alternative approach: from_documents with in-memory
                try:
                    self.vector_store = QdrantVectorStore.from_documents(
                        documents,
                        self.embeddings,
                        url=":memory:",
                        collection_name=self.config.collection_name,
                    )
                    self._initialization_successful = True
                    return True
                except Exception:
                    return False
            
        except Exception:
            return False
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search with similarity scores from QdrantVectorStore."""
        if not self.vector_store or not self._initialization_successful:
            if not self.initialize_vector_store():
                return []
        
        try:
            k = top_k or self.config.retrieval_k
            
            # Perform similarity search
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query, k=k
            )
            
            results = []
            for doc, similarity_score in docs_with_scores:
                result = SearchResult(
                    video_id=doc.metadata['video_id'],
                    video_title=doc.metadata['video_title'],
                    video_url=doc.metadata['video_url'],
                    text_content=doc.page_content,
                    similarity_score=similarity_score
                )
                results.append(result)
            
            # Sort by similarity (higher = better)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            return results
            
        except Exception:
            return []