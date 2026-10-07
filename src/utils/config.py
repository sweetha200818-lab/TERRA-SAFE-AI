"""
Configuration loader for TERRA SAFE AI
"""
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from functools import lru_cache


class Config:
    """Configuration manager with dot notation access"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation (e.g., 'models.random_forest.n_estimators')"""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        return self._config


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from YAML file"""
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def get_config() -> Config:
    """Get the global configuration instance"""
    return load_config()