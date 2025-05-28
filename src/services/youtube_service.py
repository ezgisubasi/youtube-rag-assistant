# src/services/youtube_service.py

import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pytubefix import Playlist, YouTube
from core.models import VideoMetadata
from core.config import get_config

@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    video_metadata: Optional[VideoMetadata] = None
    error_message: str = ""

class YouTubeService:
    """YouTube video downloading service."""
    
    def __init__(self):
        """Initialize YouTube service."""
        self.config = get_config()
        self.audio_dir = self.config.audio_dir
        self.transcripts_json = self.config.transcripts_json
        
        # Ensure directories exist
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        Path(self.transcripts_json).parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing videos
        self.videos: List[VideoMetadata] = []
        self._load_existing_videos()
        
        print(f"YouTube Service initialized with {len(self.videos)} existing audio files")
    
    def _load_existing_videos(self) -> None:
        """Load existing metadata from JSON file."""
        if not Path(self.transcripts_json).exists():
            return
        
        try:
            with open(self.transcripts_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                video = VideoMetadata(
                    video_id=item['video_id'],
                    title=item['video_title'],
                    url=item['video_url'],
                    file_name=item['file_name'],
                    video_text=item.get('video_text', '')
                )
                self.videos.append(video)
                
        except Exception as e:
            print(f"Warning: Could not load existing metadata: {e}")
    
    def _save_videos(self) -> None:
        """Save metadata to JSON file."""
        try:
            data = []
            for video in self.videos:
                data.append({
                    'video_id': video.video_id,
                    'video_title': video.title,
                    'video_url': video.url,
                    'file_name': video.file_name,
                    'video_text': video.video_text
                })
            
            with open(self.transcripts_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Error saving metadata: {e}")
    
    def download_single_video(self, video_url: str) -> DownloadResult:
        """Download audio from a single video."""
        try:
            yt = YouTube(video_url)
            video_id = yt.video_id
            file_name = f"{video_id}.mp4"  # Keep mp4 extension like previous project
            file_path = self.audio_dir / file_name
            
            # Check if audio file already exists
            if file_path.exists():
                print(f"Already exists: {yt.title}")
                existing_video = next((v for v in self.videos if v.video_id == video_id), None)
                if existing_video:
                    return DownloadResult(success=True, video_metadata=existing_video)
            
            # Download audio stream only
            print(f"Downloading: {yt.title}")
            audio_stream = yt.streams.filter(only_audio=True).first()
            
            if not audio_stream:
                return DownloadResult(success=False, error_message=f"No audio stream found")
            
            # Download audio as mp4 file (audio only)
            audio_stream.download(
                output_path=str(self.audio_dir), 
                filename=file_name
            )
            
            print(f"Downloaded: {yt.title}")
            
            # Create metadata
            video_metadata = VideoMetadata(
                video_id=video_id,
                title=yt.title,
                url=video_url,
                file_name=file_name,
                video_text=""
            )
            
            return DownloadResult(success=True, video_metadata=video_metadata)
            
        except Exception as e:
            return DownloadResult(success=False, error_message=str(e))
    
    def download_playlist(self, playlist_url: Optional[str] = None) -> bool:
        """Download all videos from playlist."""
        if not playlist_url:
            playlist_url = self.config.playlist_url
        
        if not playlist_url:
            print("Error: No playlist URL provided")
            return False
        
        try:
            print(f"Processing playlist: {playlist_url}")
            playlist = Playlist(playlist_url)
            
            print(f"Found {len(playlist.video_urls)} videos to download as audio")
            
            new_videos = 0
            for i, video_url in enumerate(playlist.video_urls, 1):
                print(f"\n[{i}/{len(playlist.video_urls)}] Processing...")
                
                result = self.download_single_video(video_url)
                
                if result.success and result.video_metadata:
                    # Check if video already exists in our list
                    video_exists = any(v.video_id == result.video_metadata.video_id for v in self.videos)
                    
                    if not video_exists:
                        self.videos.append(result.video_metadata)
                        new_videos += 1
                else:
                    print(f"Failed: {result.error_message}")
            
            # Save all videos
            self._save_videos()
            
            print(f"\nCompleted! Total: {len(self.videos)}, New: {new_videos}")
            return True
            
        except Exception as e:
            print(f"Error processing playlist: {e}")
            return False
    
    def get_downloaded_videos(self) -> List[VideoMetadata]:
        """Get all downloaded audio files for transcription service to use."""
        return self.videos.copy()

def main():
    """Test the service."""
    print("YouTube Service Test")
    print("=" * 30)
    
    service = YouTubeService()
    
    # Download playlist
    success = service.download_playlist()
    
    if success:
        status = service.get_status()
        print(f"Success! {status}")
    else:
        print("Failed!")

if __name__ == "__main__":
    main()