"""
Applied AI Systems - AAS-TTS
Unified Text-to-Speech package with Kokoro TTS, FastAPI, CLI, and MCP server
"""

__version__ = "1.0.0"
__author__ = "Applied AI Systems"
__email__ = "hello@appliedai.systems"

# Core exports
from .core.config import get_settings, reload_settings
from .core.models import (
    TTSRequest,
    TTSResponse,
    VoiceInfo,
    AudioFormat,
    VoiceCategory,
    TTSEngine,
)
from .services.tts_service import get_tts_service, TTSService

__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__email__",
    
    # Configuration
    "get_settings",
    "reload_settings",
    
    # Core models
    "TTSRequest",
    "TTSResponse", 
    "VoiceInfo",
    "AudioFormat",
    "VoiceCategory",
    "TTSEngine",
    
    # Services
    "get_tts_service",
    "TTSService",
]