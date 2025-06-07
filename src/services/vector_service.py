# src/services/vector_service.py
"""Vector service using QdrantVectorStore and HuggingFace Embeddings based on transcripts."""

import json
import shutil
from typing import List
from pathlib import Path
from dataclasses import dataclass
import sys
import traceback

print("🔍 [DEBUG] Starting vector_service.py import")

try:
    from langchain_qdrant import QdrantVectorStore
    print("✅ [DEBUG] QdrantVectorStore imported successfully")
except Exception as e:
    print(f"❌ [DEBUG] QdrantVectorStore import failed: {e}")
    raise

try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ [DEBUG] HuggingFaceEmbeddings imported successfully")
except Exception as e:
    print(f"❌ [DEBUG] HuggingFaceEmbeddings import failed: {e}")
    raise

try:
    from langchain.schema import Document
    print("✅ [DEBUG] Document imported successfully")
except Exception as e:
    print(f"❌ [DEBUG] Document import failed: {e}")
    raise

try:
    from qdrant_client import QdrantClient
    print("✅ [DEBUG] QdrantClient imported successfully")
except Exception as e:
    print(f"❌ [DEBUG] QdrantClient import failed: {e}")
    raise

# Add path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
sys.path.append(str(src_dir))

try:
    from core.models import SearchResult
    print("✅ [DEBUG] SearchResult imported successfully")
except Exception as e:
    print(f"❌ [DEBUG] SearchResult import failed: {e}")
    raise

try:
    from core.config import get_config
    print("✅ [DEBUG] get_config imported successfully")
except Exception as e:
    print(f"❌ [DEBUG] get_config import failed: {e}")
    raise

@dataclass
class VectorConfig:
    """Vector service configuration for whole transcript processing."""
    embedding_model: str
    vector_db_path: str
    collection_name: str
    transcripts_json: str
    retrieval_k: int

class VectorService:
    """Vector service initialized with transcripts using modern LangChain."""
    
    def __init__(self):
        """Initialize vector service."""
        print("🔍 [DEBUG] VectorService.__init__ started")
        
        try:
            config = get_config()
            print("✅ [DEBUG] Config loaded successfully")
        except Exception as e:
            print(f"❌ [DEBUG] Config loading failed: {e}")
            raise
        
        self.config = VectorConfig(
            embedding_model=config.embedding_model,
            vector_db_path=config.vector_db_path,
            collection_name=config.collection_name,
            transcripts_json=config.transcripts_json,
            retrieval_k=config.retrieval_k
        )
        
        print(f"🔍 [DEBUG] VectorConfig created:")
        print(f"  - embedding_model: {self.config.embedding_model}")
        print(f"  - vector_db_path: {self.config.vector_db_path}")
        print(f"  - collection_name: {self.config.collection_name}")
        print(f"  - transcripts_json: {self.config.transcripts_json}")
        print(f"  - retrieval_k: {self.config.retrieval_k}")
        
        # Initialize embeddings
        print("🔍 [DEBUG] Starting embeddings initialization...")
        try:
            print(f"🔍 [DEBUG] About to load model: {self.config.embedding_model}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model,
                model_kwargs={'device': 'cpu'},
                show_progress=True
            )
            print("✅ [DEBUG] Embeddings initialized successfully")
        except Exception as e:
            print(f"❌ [DEBUG] Embeddings initialization failed: {e}")
            print("🔄 [DEBUG] Trying fallback embedding model...")
            try:
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    show_progress=True
                )
                print("✅ [DEBUG] Fallback embeddings initialized successfully")
            except Exception as e2:
                print(f"❌ [DEBUG] Fallback embeddings also failed: {e2}")
                raise e2
        
        self.vector_store = None
        self.qdrant_client = None
        self._initialization_attempted = False
        self._initialization_successful = False
        print("✅ [DEBUG] VectorService.__init__ completed")
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store using transcripts as documents."""
        print("🔍 [DEBUG] initialize_vector_store started")
        
        # Check if already successfully initialized
        if self._initialization_successful and self.vector_store is not None:
            print("✅ [DEBUG] Vector store already initialized successfully")
            return True
        
        if self._initialization_attempted:
            print("⚠️ [DEBUG] Initialization already attempted and failed")
            return False
        
        self._initialization_attempted = True
        
        try:
            import os
            print(f"🔍 [DEBUG] Current working directory: {os.getcwd()}")
            print(f"🔍 [DEBUG] Vector DB path: {self.config.vector_db_path}")
            print(f"🔍 [DEBUG] Transcripts JSON: {self.config.transcripts_json}")
        
            # Setup paths
            vector_db_path = Path(self.config.vector_db_path)
            transcripts_path = Path(self.config.transcripts_json)
        
            print(f"🔍 [DEBUG] Vector DB absolute path: {vector_db_path.absolute()}")
            print(f"🔍 [DEBUG] Transcripts absolute path: {transcripts_path.absolute()}")
            print(f"🔍 [DEBUG] Vector DB exists: {vector_db_path.exists()}")
            print(f"🔍 [DEBUG] Transcripts exists: {transcripts_path.exists()}")
        
            if transcripts_path.exists():
                print(f"🔍 [DEBUG] Transcripts file size: {transcripts_path.stat().st_size} bytes")
            else:
                print("❌ [DEBUG] Transcripts file does not exist!")
                return False
        
            # Create vector DB directory
            print("🔍 [DEBUG] Creating vector DB directory...")
            vector_db_path.mkdir(parents=True, exist_ok=True)
            print("✅ [DEBUG] Vector DB directory created/exists")

            # Initialize Qdrant client only once
            if self.qdrant_client is None:
                print("🔍 [DEBUG] Initializing Qdrant client...")
                try:
                    self.qdrant_client = QdrantClient(path=str(vector_db_path))
                    print("✅ [DEBUG] Qdrant client initialized")
                except Exception as e:
                    print(f"❌ [DEBUG] Failed to initialize Qdrant client: {e}")
                    return False

            # Check if collection already exists
            print("🔍 [DEBUG] Checking if collection exists...")
            try:
                collections = self.qdrant_client.get_collections()
                collection_names = [col.name for col in collections.collections]
                collection_exists = self.config.collection_name in collection_names
                print(f"🔍 [DEBUG] Collection '{self.config.collection_name}' exists: {collection_exists}")
            except Exception as e:
                print(f"⚠️ [DEBUG] Error checking collections: {e}")
                collection_exists = False

            if collection_exists:
                print("🔍 [DEBUG] Existing collection found, attempting to load...")
                try:
                    self.vector_store = QdrantVectorStore(
                        client=self.qdrant_client,
                        collection_name=self.config.collection_name,
                        embeddings=self.embeddings
                    )
                    
                    # Test the vector store
                    test_results = self.vector_store.similarity_search("test", k=1)
                    print(f"✅ [DEBUG] Existing vector store loaded successfully ({len(test_results)} test results)")
                    self._initialization_successful = True
                    return True
                    
                except Exception as e:
                    print(f"⚠️ [DEBUG] Failed to load existing vector store: {e}")
                    # Delete the problematic collection and recreate
                    try:
                        self.qdrant_client.delete_collection(self.config.collection_name)
                        print("🗑️ [DEBUG] Deleted problematic collection")
                    except:
                        pass
            
            # Read transcripts file
            print("🔍 [DEBUG] Reading transcripts file...")
            try:
                with open(transcripts_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"✅ [DEBUG] Transcripts loaded: {len(data)} items")
            except Exception as e:
                print(f"❌ [DEBUG] Failed to read transcripts file: {e}")
                return False
            
            # Create documents from transcripts
            print("🔍 [DEBUG] Creating documents from transcripts...")
            documents = []
            for i, item in enumerate(data):
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
                        if i < 3:  # Show first few items
                            print(f"✅ [DEBUG] Doc {i+1}: {item['video_title'][:50]}...")
                    else:
                        print(f"⚠️ [DEBUG] Item {i} has empty transcript")
                except Exception as e:
                    print(f"❌ [DEBUG] Error processing item {i}: {e}")
            
            if not documents:
                print("❌ [DEBUG] No valid documents found in transcripts")
                return False
            
            print(f"✅ [DEBUG] Created {len(documents)} documents")
            
            # Create vector store using modern QdrantVectorStore
            print("🔍 [DEBUG] Creating vector store with QdrantVectorStore.from_documents...")
            try:
                self.vector_store = QdrantVectorStore.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    client=self.qdrant_client,
                    collection_name=self.config.collection_name
                )
                print("✅ [DEBUG] Vector store created successfully")
                
                # Test the newly created vector store
                test_results = self.vector_store.similarity_search("liderlik", k=1)
                print(f"✅ [DEBUG] Vector store test successful ({len(test_results)} results)")
                
                self._initialization_successful = True
                return True
                
            except Exception as e:
                print(f"❌ [DEBUG] Vector store creation failed: {e}")
                print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
                return False
            
        except Exception as e:
            print(f"❌ [DEBUG] initialize_vector_store failed with unexpected error: {e}")
            print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
            return False
    
    def search(self, query: str, top_k: int = None) -> List[SearchResult]:
        """Search with similarity scores from QdrantVectorStore."""
        print(f"🔍 [DEBUG] search called with query: '{query}', top_k: {top_k}")
        print(f"🔍 [DEBUG] Vector store status: {self.vector_store is not None}")
        print(f"🔍 [DEBUG] Initialization status: successful={self._initialization_successful}, attempted={self._initialization_attempted}")
        
        if not self.vector_store or not self._initialization_successful:
            print("⚠️ [DEBUG] Vector store not initialized, attempting to initialize...")
            if not self.initialize_vector_store():
                print("❌ [DEBUG] Failed to initialize vector store for search")
                return []
        
        try:
            k = top_k or self.config.retrieval_k
            print(f"🔍 [DEBUG] Searching with k={k}")
            
            # Get results with similarity scores
            docs_with_scores = self.vector_store.similarity_search_with_score(
                query=query, k=k
            )
            
            print(f"✅ [DEBUG] Search returned {len(docs_with_scores)} results")
            
            results = []
            for i, (doc, similarity_score) in enumerate(docs_with_scores):
                result = SearchResult(
                    video_id=doc.metadata['video_id'],
                    video_title=doc.metadata['video_title'],
                    video_url=doc.metadata['video_url'],
                    text_content=doc.page_content,
                    similarity_score=similarity_score
                )
                results.append(result)
                print(f"🔍 [DEBUG] Result {i+1}: {doc.metadata['video_title'][:30]}... (score: {similarity_score:.3f})")
            
            # Sort by similarity (higher score = more similar with modern Qdrant)
            results.sort(key=lambda x: x.similarity_score, reverse=True)
            print(f"✅ [DEBUG] Returning {len(results)} sorted results")
            return results
            
        except Exception as e:
            print(f"❌ [DEBUG] Search failed: {e}")
            print(f"❌ [DEBUG] Traceback: {traceback.format_exc()}")
            return []

print("✅ [DEBUG] vector_service.py import completed")