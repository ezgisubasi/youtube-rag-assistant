# src/core/config.py
"""Configuration management with YAML and environment variables."""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.models import AppConfig

class ConfigManager:
    """Configuration manager."""
    
    def __init__(self):
        self.config_dir = Path("config")
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[AppConfig] = None
        self._prompts: Optional[Dict[str, str]] = None
    
    def get_config(self) -> AppConfig:
        """Get application configuration (cached)."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_config(self) -> AppConfig:
        """Load configuration from YAML and environment."""
        # Start with default dataclass values
        config = AppConfig()
        
        # Override with YAML settings if file exists
        settings_file = self.config_dir / "settings.yaml"
        if settings_file.exists():
            yaml_settings = self._load_yaml_file(settings_file)
            
            # Update config with YAML values
            for key, value in yaml_settings.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # Override with environment variables (highest priority)
        config.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        
        if os.getenv("YOUTUBE_PLAYLIST_URL"):
            config.playlist_url = os.getenv("YOUTUBE_PLAYLIST_URL")
        
        return config
    
    def get_prompts(self) -> Dict[str, str]:
        """Get prompt templates from YAML file."""
        if self._prompts is None:
            prompts_file = self.config_dir / "prompts.yaml"
            if prompts_file.exists():
                self._prompts = self._load_yaml_file(prompts_file)
            else:
                self._prompts = {}
        return self._prompts
    
    def _load_yaml_file(self, file_path: Path) -> Dict:
        """Load YAML file safely."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            return {}

# Global configuration manager
_config_manager = ConfigManager()

def get_config() -> AppConfig:
    """Get application configuration."""
    return _config_manager.get_config()

def get_prompts() -> Dict[str, str]:
    """Get prompt templates."""
    return _config_manager.get_prompts()

def validate_config() -> bool:
    """Validate application configuration."""
    config = get_config()
    return config.validate()