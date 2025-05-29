# src/services/transcription_service.py
"""Transcription service using OpenAI Whisper."""

import whisper
import os
import json
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.models import VideoMetadata
from core.config import get_config

@dataclass
class TranscriptionResult:
    """Result of a transcription operation."""
    success: bool
    video_id: str
    transcript: str = ""
    error_message: str = ""

class TranscriptionService:
    """Transcription service using Whisper."""
    
    def __init__(self):
        """Initialize transcription service."""
        self.config = get_config()
        self.model_size = self.config.whisper_model
        self.language = self.config.language
        self.audio_dir = Path(self.config.audio_dir)
        self.transcripts_json = self.config.transcripts_json
        self.transcripts_dir = Path(self.config.transcripts_dir)
        
        # Create transcripts directory
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper model
        print(f"Loading Whisper {self.model_size} model...")
        self.model = whisper.load_model(self.model_size)
        print("Whisper model loaded!")
        
    def transcribe_single_file(self, audio_path: Path, video_id: str) -> TranscriptionResult:
        """Transcribe a single audio file."""
        try:
            print(f"Transcribing: {audio_path.name}")
            
            # Transcribe with language specification
            result = self.model.transcribe(str(audio_path), language=self.language)
            
            transcript = result["text"].strip()
            
            if transcript:
                print(f"Completed: {audio_path.name})")
                return TranscriptionResult(
                    success=True,
                    video_id=video_id,
                    transcript=transcript
                )
            else:
                return TranscriptionResult(
                    success=False,
                    video_id=video_id,
                    error_message="Empty transcript generated"
                )
                
        except Exception as e:
            print(f"Error transcribing {audio_path.name}: {e}")
            return TranscriptionResult(
                success=False,
                video_id=video_id,
                error_message=str(e)
            )
    
    def save_individual_transcript(self, video_metadata: VideoMetadata, transcript: str) -> None:
        """Save transcript as individual text file."""
        filename = f"{video_metadata.video_id}.txt"
        file_path = self.transcripts_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Error saving {filename}: {e}")
    
    def load_videos_metadata(self) -> List[VideoMetadata]:
        """Load video metadata from JSON file."""
        if not Path(self.transcripts_json).exists():
            print(f"Metadata file not found: {self.transcripts_json}")
            return []
        
        try:
            with open(self.transcripts_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            videos = []
            for item in data:
                video = VideoMetadata(
                    video_id=item['video_id'],
                    title=item['video_title'],
                    url=item['video_url'],
                    file_name=item['file_name'],
                    video_text=item.get('video_text', '')
                )
                videos.append(video)
            
            return videos
            
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return []
    
    def save_videos_metadata(self, videos: List[VideoMetadata]) -> bool:
        """Save updated video metadata to JSON file."""
        try:
            data = []
            for video in videos:
                data.append({
                    'video_id': video.video_id,
                    'video_title': video.title,
                    'video_url': video.url,
                    'file_name': video.file_name,
                    'video_text': video.video_text
                })
            
            with open(self.transcripts_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error saving metadata: {e}")
            return False
    
    def transcribe_all_videos(self) -> bool:
        """Transcribe all videos that need transcription."""
        # Load videos metadata
        videos = self.load_videos_metadata()
        
        if not videos:
            print("No videos found to transcribe")
            return False
        
        print(f"Processing {len(videos)} videos...")
        print(f"Saving individual transcripts to: {self.transcripts_dir}")
        
        for i, video in enumerate(videos, 1):
            # Skip if already transcribed
            if video.video_text.strip():
                print(f"[{i}/{len(videos)}] Already done: {video.title}")
                continue
            
            # Check if audio file exists
            audio_file = self.audio_dir / video.file_name
            
            if not audio_file.exists():
                print(f"[{i}/{len(videos)}] Audio file not found: {video.file_name}")
                continue
            
            print(f"[{i}/{len(videos)}] Processing: {video.title}")
            
            # Transcribe
            result = self.transcribe_single_file(audio_file, video.video_id)
            
            if result.success:
                # Update video metadata
                video.video_text = result.transcript
                
                # Save individual text file
                self.save_individual_transcript(video, result.transcript)
                
                print(f"Done: {video.title}")
            else:
                print(f"Failed: {video.title} - {result.error_message}")
        
        # Save all metadata
        success = self.save_videos_metadata(videos)
        if success:
            print(f"\nTranscription completed!")
            print(f"Updated metadata file: {self.transcripts_json}")
            print(f"Individual transcripts saved to: {self.transcripts_dir}")
        else:
            print("Error saving metadata")
            return False
        
        return True
    

def main():
    """Test the transcription service."""
    print("Transcription Service Test")
    print("=" * 40)
    
    try:
        service = TranscriptionService()
        
        # Start transcription
        print("Starting transcription...")
        success = service.transcribe_all_videos()
        
        if success:
            print("Transcription completed successfully!")
        else:
            print("Transcription failed!")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()