# src/core/models.py
"""Data models for the YouTube RAG Assistant."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path
import os

class ProcessingStatus(Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class VideoMetadata:
    """Simple video metadata model."""
    video_id: str
    title: str
    url: str
    file_name: str
    video_text: str = ""  # For transcript storage
    
    def __post_init__(self):
        """Validate video metadata."""
        if not self.video_id or not self.title or not self.url:
            raise ValueError("video_id, title, and url are required")

@dataclass
class SearchResult:
    """Search result from vector database."""
    video_id: str
    video_title: str
    video_url: str
    text_content: str
    similarity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "video_url": self.video_url,
            "text_content": self.text_content,
            "similarity_score": self.similarity_score
        }

@dataclass
class RAGResponse:
    """RAG system response."""
    query: str
    answer: str
    sources: List[SearchResult]
    confidence_score: Optional[float] = None
    
    def get_source_titles(self) -> List[str]:
        """Get video titles used as sources."""
        return [source.video_title for source in self.sources]
    
    def get_source_urls(self) -> List[str]:
        """Get video URLs used as sources."""
        return [source.video_url for source in self.sources]

@dataclass
class AppConfig:
    """Application configuration - structure only, values from YAML."""
    # API Settings (from environment only)
    gemini_api_key: str = ""
    
    # All other settings (from YAML files)
    model_name: str = ""
    playlist_url: str = ""
    whisper_model: str = ""
    language: str = ""
    embedding_model: str = ""
    vector_db_path: str = ""
    collection_name: str = ""
    retrieval_k: int = 0
    similarity_threshold: float = 0.0
    transcripts_json: str = ""
    
    # File Paths
    data_dir: Path = Path("data")
    audio_dir: Path = Path("data/audio")
    transcripts_dir: Path = Path("data/transcripts")
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
    
    def validate(self) -> bool:
        """Validate required configuration."""
        if not self.gemini_api_key:
            print("ERROR: GEMINI_API_KEY environment variable is required!")
            print("Please set: export GEMINI_API_KEY=your_api_key_here")
            return False
        
        if not self.model_name:
            print("ERROR: model_name not found in settings.yaml")
            return False
            
        return True

@dataclass
class SystemStatus:
    """Simple system status."""
    total_videos: int = 0
    processed_videos: int = 0
    is_ready: bool = False
    error_message: Optional[str] = None