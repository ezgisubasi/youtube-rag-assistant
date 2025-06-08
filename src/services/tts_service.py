# src/services/simple_tts_service.py
"""Simple and efficient TTS service using ElevenLabs API."""

import os
import requests
import base64
from typing import Optional

class SimpleTTSService:
    """Simple ElevenLabs TTS service."""
    
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam voice
        self.base_url = "https://api.elevenlabs.io/v1"
    
    def is_available(self) -> bool:
        """Check if TTS is available."""
        return bool(self.api_key)
    
    def generate_speech(self, text: str) -> Optional[bytes]:
        """Generate speech from text."""
        if not self.api_key or not text:
            return None
        
        # Limit text length
        if len(text) > 500:
            text = text[:500]
        
        try:
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"TTS Error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"TTS Error: {e}")
            return None
    
    def create_audio_player(self, audio_data: bytes) -> str:
        """Create HTML audio player."""
        if not audio_data:
            return ""
        
        b64 = base64.b64encode(audio_data).decode()
        
        return f"""
        <audio controls style="width: 100%; margin: 10px 0;">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
        """