"""
Applied AI Systems - AAS-TTS API Package
FastAPI web server with OpenAI-compatible endpoints
"""
from .server import create_app

__all__ = ["create_app"]