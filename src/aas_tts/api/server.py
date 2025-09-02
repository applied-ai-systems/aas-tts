"""
Applied AI Systems - FastAPI Server
Main FastAPI application with OpenAI-compatible TTS endpoints
"""
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Dict, Any

from ..core.config import get_settings
from ..core.models import TTSRequest, AudioFormat
from ..services.tts_service import get_tts_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    settings = get_settings()
    
    # Initialize TTS service
    await get_tts_service()
    
    yield
    
    # Shutdown
    # TODO: Cleanup resources


async def create_app() -> FastAPI:
    """Create FastAPI application"""
    settings = get_settings()
    
    app = FastAPI(
        title="AAS-TTS",
        description="Applied AI Systems - Text-to-Speech API with OpenAI compatibility",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
    )
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        service = await get_tts_service()
        health_info = await service.health_check()
        
        if health_info["status"] != "healthy":
            raise HTTPException(status_code=503, detail=health_info)
        
        return health_info
    
    @app.post("/v1/audio/speech")
    async def create_speech(request: Dict[str, Any]):
        """
        OpenAI-compatible TTS endpoint
        Expected format from aas-chat:
        {
          "input": "text to synthesize", 
          "model": "tts-1",
          "voice": "af_bella",
          "format": "wav"
        }
        """
        try:
            service = await get_tts_service()
            
            # Map OpenAI request to our format
            tts_request = TTSRequest(
                text=request["input"],
                voice=request.get("voice", "af_bella"),
                format=AudioFormat(request.get("format", "wav")),
                model=request.get("model", "tts-1"),  # For compatibility
                speed=request.get("speed", 1.0)
            )
            
            # Synthesize
            response = await service.synthesize(tts_request)
            
            if not response.success:
                raise HTTPException(status_code=500, detail=response.error)
            
            # Return audio data
            return Response(
                content=response.audio_data,
                media_type=f"audio/{response.format.value}",
                headers={
                    "Content-Length": str(len(response.audio_data)),
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/v1/models")
    async def list_models():
        """List available TTS models (OpenAI compatibility)"""
        return {
            "object": "list",
            "data": [
                {
                    "id": "tts-1",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "applied-ai-systems"
                }
            ]
        }
    
    @app.get("/v1/voices")
    async def list_voices():
        """List available voices"""
        service = await get_tts_service()
        voices = await service.get_available_voices()
        
        return {
            "voices": [voice.model_dump() for voice in voices]
        }
    
    return app