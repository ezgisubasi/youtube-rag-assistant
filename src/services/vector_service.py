# src/services/vector_service.py
"""Simple vector service using LangChain and Qdrant."""

import json
from typing import List
from pathlib import Path
from dataclasses import dataclass
import sys

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

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
    """Simple vector service for YouTube RAG."""
    
    def __init__(self):
        """Initialize vector service."""
        config = get_config()
        
        # Create vector config from app config
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
        print("Vector service initialized!")
    
    def create_vector_store(self) -> bool:
        """Create vector store from transcripts."""
        try:
            # Load transcripts
            if not Path(self.config.transcripts_json).exists():
                print(f"Transcripts file not found: {self.config.transcripts_json}")
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
                print("No documents with transcripts found!")
                return False
            
            print(f"Processing {len(documents)} documents...")
            
            # Split into chunks
            chunks = self.text_splitter.split_documents(documents)
            print(f"Created {len(chunks)} chunks")
            
            # Create vector store
            self.vector_store = Qdrant.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                path=self.config.vector_db_path,
                collection_name=self.config.collection_name,
                force_recreate=True
            )
            
            print("Vector store created successfully!")
            return True
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search for relevant content."""
        if not self.vector_store:
            print("Vector store not initialized!")
            return []
        
        try:
            k = top_k or self.config.retrieval_k
            
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query,
                k=k
            )
            
            results = []
            for doc, score in docs_with_scores:
                result = SearchResult(
                    video_id=doc.metadata['video_id'],
                    video_title=doc.metadata['video_title'],
                    video_url=doc.metadata['video_url'],
                    text_content=doc.page_content,
                    similarity_score=float(score)
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []

def main():
    """Test the vector service."""
    service = VectorService()
    
    # Create vector store
    success = service.create_vector_store()
    if not success:
        return
    
    # Test search
    results = service.search("leadership skills")
    print(f"Found {len(results)} results")
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.video_title} (Score: {result.similarity_score:.3f})")

if __name__ == "__main__":
    main()