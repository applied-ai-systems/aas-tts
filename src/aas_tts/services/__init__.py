"""
Applied AI Systems - AAS-TTS Services Package
Business logic layer for TTS operations
"""
from .tts_service import TTSService, get_tts_service

__all__ = [
    "TTSService",
    "get_tts_service",
]