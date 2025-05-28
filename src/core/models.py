# src/core/models.py
"""Data models for the YouTube RAG Assistant."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
from pathlib import Path

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
    processing_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_source_titles(self) -> List[str]:
        """Get video titles used as sources."""
        return [source.video_title for source in self.sources]
    
    def get_source_urls(self) -> List[str]:
        """Get video URLs used as sources."""
        return [source.video_url for source in self.sources]

@dataclass
class AppConfig:
    """Application configuration."""
    # API Settings
    gemini_api_key: str = ""
    model_name: str = "gemini-2.0-flash-exp"
    
    # YouTube Settings
    playlist_url: str = ""
    
    # Transcription Settings
    whisper_model: str = "medium"
    language: str = "tr"
    
    # Vector Database Settings
    embedding_model: str = "sentence-transformers/altaidevorg/bge-m3-distill-8l"
    vector_db_path: str = "data/vector_db"
    collection_name: str = "youtube_transcripts"
    
    # RAG Settings
    retrieval_k: int = 3
    similarity_threshold: float = 0.7
    
    # File Paths
    data_dir: Path = Path("data")
    audio_dir: Path = Path("data/audio")
    transcripts_dir: Path = Path("data/transcripts")
    transcripts_json: str = "data/transcripts.json"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.audio_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        import os
        return cls(
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            playlist_url=os.getenv("YOUTUBE_PLAYLIST_URL", ""),
            whisper_model=os.getenv("WHISPER_MODEL", "medium"),
            language=os.getenv("LANGUAGE_CODE", "tr"),
            vector_db_path=os.getenv("VECTOR_DB_PATH", "data/vector_db"),
            collection_name=os.getenv("COLLECTION_NAME", "youtube_transcripts")
        )

@dataclass
class SystemStatus:
    """Simple system status."""
    total_videos: int = 0
    processed_videos: int = 0
    is_ready: bool = False
    error_message: Optional[str] = None