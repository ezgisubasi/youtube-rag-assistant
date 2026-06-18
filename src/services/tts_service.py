# src/services/tts_service.py
"""Multi-provider TTS service (ElevenLabs + 60db) with a single consistent interface.

Every provider exposes the same surface:
    - is_available() -> bool
    - generate_speech(text) -> Optional[bytes]   # always returns MP3 bytes
and shares one create_audio_player(). The TTSManager routes between them so the
rest of the app never needs to know which engine produced the audio.
"""

import os
import base64
import requests
from typing import Optional, List


class BaseTTSService:
    """Common interface + shared helpers for all TTS providers."""

    name: str = "base"

    def is_available(self) -> bool:
        raise NotImplementedError

    def generate_speech(self, text: str) -> Optional[bytes]:
        raise NotImplementedError

    def create_audio_player(self, audio_data: bytes) -> str:
        """Create an inline HTML5 audio player from MP3 bytes (shared by all providers)."""
        if not audio_data:
            return ""

        b64 = base64.b64encode(audio_data).decode()

        return f"""
        <audio controls style="width: 100%; margin: 10px 0;">
            <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
        </audio>
        """


class ElevenLabsTTSService(BaseTTSService):
    """ElevenLabs TTS via raw REST. Returns MP3 bytes."""

    name = "ElevenLabs"

    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.voice_id = "pNInz6obpgDQGcFmaJgB"  # Adam voice
        self.base_url = "https://api.elevenlabs.io/v1"
        self.max_chars = 500

    def is_available(self) -> bool:
        """Check if ElevenLabs TTS is available."""
        return bool(self.api_key)

    def generate_speech(self, text: str) -> Optional[bytes]:
        """Generate speech from text. Returns MP3 bytes or None."""
        if not self.api_key or not text:
            return None

        # Limit text length
        if len(text) > self.max_chars:
            text = text[:self.max_chars]

        try:
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"

            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            }

            data = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                },
            }

            response = requests.post(url, json=data, headers=headers, timeout=30)

            if response.status_code == 200:
                return response.content

            print(f"[ElevenLabs] TTS Error: {response.status_code}")
            return None

        except Exception as e:
            print(f"[ElevenLabs] TTS Error: {e}")
            return None


class SixtyDBTTSService(BaseTTSService):
    """60db TTS via the synchronous /tts-synthesize endpoint.

    60db returns JSON with a base64 `audio_base64` field, which we decode to MP3
    bytes so the output is identical in shape to ElevenLabs.
    """

    name = "60db"

    def __init__(self):
        self.api_key = os.getenv("SIXTYDB_API_KEY")
        # Per-account voice UUID. If unset, 60db uses its system default voice.
        self.voice_id = os.getenv("SIXTYDB_VOICE_ID")
        self.base_url = "https://api.60db.ai"
        self.max_chars = 5000  # documented 60db limit

    def is_available(self) -> bool:
        """Check if 60db TTS is available."""
        return bool(self.api_key)

    def generate_speech(self, text: str) -> Optional[bytes]:
        """Generate speech from text. Returns MP3 bytes or None."""
        if not self.api_key or not text:
            return None

        # Limit text length
        if len(text) > self.max_chars:
            text = text[:self.max_chars]

        try:
            url = f"{self.base_url}/tts-synthesize"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            data = {
                "text": text,
                "output_format": "mp3",  # keep MP3 so the shared player works
                "enhance": True,
                "speed": 1,
                "stability": 50,      # 60db scale is 0-100 (lower = expressive)
                "similarity": 75,     # 60db scale is 0-100 (voice match)
            }

            # voice_id is optional; omit it entirely to use the account default.
            if self.voice_id:
                data["voice_id"] = self.voice_id

            response = requests.post(url, json=data, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"[60db] TTS Error: {response.status_code}")
                return None

            payload = response.json()

            if payload.get("success") and payload.get("audio_base64"):
                return base64.b64decode(payload["audio_base64"])

            print(f"[60db] TTS Error: {payload.get('message', 'unknown error')}")
            return None

        except Exception as e:
            print(f"[60db] TTS Error: {e}")
            return None


class TTSManager(BaseTTSService):
    """Routes between TTS providers behind one consistent interface.

    Drop-in compatible with the old single-provider TTSService: exposes
    is_available(), create_audio_player() and generate_speech(text). Adds
    available_engines() and an optional `engine` argument so the UI can pick.
    """

    def __init__(self):
        self.providers = {
            ElevenLabsTTSService.name: ElevenLabsTTSService(),
            SixtyDBTTSService.name: SixtyDBTTSService(),
        }

    def available_engines(self) -> List[str]:
        """Names of providers that have a configured API key."""
        return [name for name, p in self.providers.items() if p.is_available()]

    def is_available(self) -> bool:
        """True if at least one provider is configured."""
        return len(self.available_engines()) > 0

    def generate_speech(self, text: str, engine: Optional[str] = None) -> Optional[bytes]:
        """Generate speech using the chosen engine (or the first available one)."""
        provider = self._resolve(engine)
        if provider is None:
            return None
        return provider.generate_speech(text)

    def _resolve(self, engine: Optional[str]) -> Optional[BaseTTSService]:
        """Pick a usable provider: requested engine if available, else first available."""
        if engine and engine in self.providers and self.providers[engine].is_available():
            return self.providers[engine]

        available = self.available_engines()
        return self.providers[available[0]] if available else None


# Backwards-compatible alias: existing imports of TTSService keep working.
TTSService = TTSManager
