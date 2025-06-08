# src/services/tts_service.py
"""Text-to-Speech service using ElevenLabs API."""

import os
import requests
import streamlit as st
from typing import Optional, Dict, Any
from pathlib import Path
import base64
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.config import get_config

class TTSService:
    """ElevenLabs Text-to-Speech service."""
    
    def __init__(self):
        """Initialize TTS service."""
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.base_url = "https://api.elevenlabs.io/v1"
        
        # Voice configurations for different languages
        self.voice_configs = {
            'turkish': {
                'voice_id': 'pNInz6obpgDQGcFmaJgB',  # Adam (multilingual)
                'model_id': 'eleven_multilingual_v2',
                'stability': 0.75,
                'similarity_boost': 0.75,
                'style': 0.0,
                'use_speaker_boost': True
            },
            'english': {
                'voice_id': 'pNInz6obpgDQGcFmaJgB',  # Adam (multilingual)
                'model_id': 'eleven_multilingual_v2',
                'stability': 0.75,
                'similarity_boost': 0.75,
                'style': 0.0,
                'use_speaker_boost': True
            }
        }
        
        # Alternative voices (you can customize these)
        self.available_voices = {
            'adam': 'pNInz6obpgDQGcFmaJgB',      # Male, deep
            'bella': 'EXAVITQu4vr4xnSDxMaL',     # Female, young
            'antoni': 'ErXwobaYiN019PkySvjV',    # Male, well-rounded
            'elli': 'MF3mGyEYCl7XYWbV9V6O',      # Female, emotional
            'josh': 'TxGEqnHWrfWFTfGW9XjX',      # Male, deep
            'arnold': 'VR6AewLTigWG4xSOukaG',    # Male, crisp
            'charlotte': 'XB0fDUnXU5powFXDhCwa', # Female, seductive
            'matilda': 'XrExE9yKIg1WjnnlVkGX'    # Female, warm
        }
    
    def is_available(self) -> bool:
        """Check if TTS service is available."""
        return bool(self.api_key)
    
    def get_voices(self) -> Optional[Dict[str, Any]]:
        """Get available voices from ElevenLabs."""
        if not self.api_key:
            return None
        
        try:
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }
            
            response = requests.get(f"{self.base_url}/voices", headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error getting voices: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error getting voices: {e}")
            return None
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Turkish or English (reuse from RAG service)."""
        import re
        
        # Turkish-specific characters
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        
        # Turkish words
        turkish_words = {
            'nedir', 'nasıl', 'neden', 'hangi', 'için', 'olan', 'bu', 'bir', 
            'de', 'da', 'ile', 've', 'veya', 'çünkü', 'şey', 'zaman', 'mi', 
            'mı', 'mu', 'mü', 'ne', 'kim', 'nerede', 'önemli', 'söz', 'lider',
            'liderlik', 'takım', 'başarı', 'iş', 'şirket', 'yönetim'
        }
        
        text_lower = text.lower()
        
        # Check for Turkish characters
        if any(char in text for char in turkish_chars):
            return 'turkish'
        
        # Count Turkish words
        words = re.findall(r'\b\w+\b', text_lower)
        turkish_score = sum(1 for word in words if word in turkish_words)
        
        return 'turkish' if turkish_score > 0 else 'english'
    
    def generate_speech(self, text: str, language: Optional[str] = None, voice_name: str = 'adam') -> Optional[bytes]:
        """Generate speech from text using ElevenLabs."""
        if not self.api_key:
            return None
        
        # Auto-detect language if not provided
        if not language:
            language = self.detect_language(text)
        
        # Get voice configuration
        voice_config = self.voice_configs.get(language, self.voice_configs['english'])
        
        # Override voice if specified
        if voice_name in self.available_voices:
            voice_config['voice_id'] = self.available_voices[voice_name]
        
        try:
            url = f"{self.base_url}/text-to-speech/{voice_config['voice_id']}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key
            }
            
            data = {
                "text": text,
                "model_id": voice_config['model_id'],
                "voice_settings": {
                    "stability": voice_config['stability'],
                    "similarity_boost": voice_config['similarity_boost'],
                    "style": voice_config['style'],
                    "use_speaker_boost": voice_config['use_speaker_boost']
                }
            }
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"TTS Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"TTS generation error: {e}")
            return None
    
    def create_audio_player(self, audio_data: bytes, autoplay: bool = False) -> str:
        """Create HTML audio player for Streamlit."""
        if not audio_data:
            return ""
        
        # Encode audio data to base64
        b64 = base64.b64encode(audio_data).decode()
        
        # Create HTML audio player
        audio_html = f"""
        <audio {'autoplay' if autoplay else ''} controls style="width: 100%; margin: 10px 0;">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
            Your browser does not support the audio element.
        </audio>
        """
        
        return audio_html
    
    def get_usage_info(self) -> Optional[Dict[str, Any]]:
        """Get current usage information from ElevenLabs."""
        if not self.api_key:
            return None
        
        try:
            headers = {
                "Accept": "application/json",
                "xi-api-key": self.api_key
            }
            
            response = requests.get(f"{self.base_url}/user", headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            print(f"Error getting usage info: {e}")
            return None

# Streamlit components for TTS integration
class StreamlitTTSComponents:
    """Streamlit-specific TTS components."""
    
    @staticmethod
    def render_tts_controls(tts_service: TTSService) -> Dict[str, Any]:
        """Render TTS control panel in sidebar."""
        tts_settings = {}
        
        with st.sidebar:
            st.subheader("🔊 Text-to-Speech")
            
            if not tts_service.is_available():
                st.error("ElevenLabs API key not configured")
                st.info("Add ELEVENLABS_API_KEY to your environment")
                tts_settings['enabled'] = False
                return tts_settings
            
            # TTS enabled toggle
            tts_settings['enabled'] = st.checkbox("Enable TTS", value=True)
            
            if tts_settings['enabled']:
                # Voice selection
                voice_options = {
                    'Adam (Male, Deep)': 'adam',
                    'Bella (Female, Young)': 'bella',
                    'Antoni (Male, Well-rounded)': 'antoni',
                    'Elli (Female, Emotional)': 'elli',
                    'Josh (Male, Deep)': 'josh',
                    'Charlotte (Female, Seductive)': 'charlotte',
                    'Matilda (Female, Warm)': 'matilda'
                }
                
                selected_voice_name = st.selectbox(
                    "Voice",
                    options=list(voice_options.keys()),
                    index=0
                )
                tts_settings['voice'] = voice_options[selected_voice_name]
                
                # Auto-play option
                tts_settings['autoplay'] = st.checkbox("Auto-play responses", value=False)
                
                # Usage info
                if st.button("Check Usage"):
                    usage_info = tts_service.get_usage_info()
                    if usage_info:
                        subscription = usage_info.get('subscription', {})
                        st.info(f"Characters used: {subscription.get('character_count', 0):,} / {subscription.get('character_limit', 0):,}")
                    else:
                        st.warning("Could not fetch usage info")
        
        return tts_settings
    
    @staticmethod
    def display_message_with_tts(message: Dict[str, Any], tts_service: TTSService, tts_settings: Dict[str, Any]):
        """Display message with TTS option."""
        role = message["role"]
        
        if role == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant"):
                content = message["content"]
                sources = message.get("sources", [])
                confidence = message.get("confidence", 0.0)
                
                # Display the main answer
                st.write(content)
                
                # TTS controls for assistant messages
                if tts_settings.get('enabled', False) and content:
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        if st.button("🔊 Play", key=f"tts_{hash(content)}"):
                            with st.spinner("Generating speech..."):
                                audio_data = tts_service.generate_speech(
                                    text=content,
                                    voice_name=tts_settings.get('voice', 'adam')
                                )
                                
                                if audio_data:
                                    audio_html = tts_service.create_audio_player(
                                        audio_data, 
                                        autoplay=tts_settings.get('autoplay', False)
                                    )
                                    st.markdown(audio_html, unsafe_allow_html=True)
                                else:
                                    st.error("Failed to generate speech")
                
                # Display source information if available
                if sources and len(sources) > 0:
                    source = sources[0]
                    video_title = getattr(source, 'video_title', 'Unknown')
                    video_url = getattr(source, 'video_url', '#')
                    
                    st.markdown(f"""
                    <div class="source-info">
                        <strong>Source:</strong> {video_title}<br>
                        <strong>Link:</strong> <a href="{video_url}" target="_blank">{video_url}</a><br>
                        <strong>Confidence Score:</strong> {confidence:.2f}
                    </div>
                    """, unsafe_allow_html=True)