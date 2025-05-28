# src/core/config.py
"""Configuration management for YouTube RAG Assistant."""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.models import AppConfig

class ConfigManager:
    """Configuration manager with environment override capability."""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[AppConfig] = None
    
    def get_config(self) -> AppConfig:
        """Get application configuration (cached after first load)."""
        if self._config is None:
            self._config = self._load_configuration()
        return self._config
    
    def _load_configuration(self) -> AppConfig:
        """Load configuration: Environment variables override YAML settings."""
        
        # Start with empty config
        config = AppConfig()
        
        # Load sensitive data from environment
        config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        config.playlist_url = os.getenv("YOUTUBE_PLAYLIST_URL", "")
        
        # Load everything else from YAML (YAML must exist)
        settings_file = self.config_dir / "settings.yaml"
        if not settings_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {settings_file}")
        
        yaml_settings = self._load_yaml_settings(settings_file)
        
        # Apply all YAML settings
        config.model_name = yaml_settings.get('model_name', '')
        config.whisper_model = yaml_settings.get('whisper_model', '')
        config.language = yaml_settings.get('language', '')
        config.embedding_model = yaml_settings.get('embedding_model', '')
        config.retrieval_k = yaml_settings.get('retrieval_k', 0)
        config.similarity_threshold = yaml_settings.get('similarity_threshold', 0.0)
        config.vector_db_path = yaml_settings.get('vector_db_path', '')
        config.collection_name = yaml_settings.get('collection_name', '')
        config.transcripts_json = yaml_settings.get('transcripts_json', 'data/transcripts.json')
        
        # Override playlist_url from YAML if not set in environment
        if not config.playlist_url and yaml_settings.get('playlist_url'):
            config.playlist_url = yaml_settings['playlist_url']
        
        return config
    
    def _load_yaml_settings(self, file_path: Path) -> Dict:
        """Load and parse YAML configuration file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Failed to load configuration from {file_path}: {e}")
            return {}
    
    def get_prompts(self) -> Dict[str, str]:
        """Load prompt templates from YAML file."""
        prompts_file = self.config_dir / "prompts.yaml"
        
        if not prompts_file.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
        
        return self._load_yaml_settings(prompts_file)
    
    def validate_configuration(self) -> bool:
        """Validate that all required configuration is present."""
        config = self.get_config()
        
        validation_errors = []
        
        if not config.gemini_api_key:
            validation_errors.append("GEMINI_API_KEY is required")
        
        if not config.playlist_url:
            validation_errors.append("YOUTUBE_PLAYLIST_URL is required")
        
        if not config.model_name:
            validation_errors.append("model_name is required")
        
        if validation_errors:
            print("Configuration validation failed:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        
        print("Configuration validation successful")
        return True

# Global configuration manager instance
_config_manager = ConfigManager()

# Public API for application components
def get_config() -> AppConfig:
    """Get application configuration."""
    return _config_manager.get_config()

def get_prompts() -> Dict[str, str]:
    """Get prompt templates."""
    return _config_manager.get_prompts()

def validate_config() -> bool:
    """Validate application configuration."""
    return _config_manager.validate_configuration()