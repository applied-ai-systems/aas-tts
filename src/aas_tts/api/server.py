"""
Applied AI Systems - FastAPI Server
Main FastAPI application with OpenAI-compatible TTS endpoints
"""
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, AsyncGenerator
import json
import base64

from ..core.config import get_settings
from ..core.models import TTSRequest, AudioFormat
from ..services.tts_service import get_tts_service
from ..services.streaming_service import get_streaming_service, StreamingFormat


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    settings = get_settings()
    
    # Initialize services
    await get_tts_service()
    await get_streaming_service()
    
    yield
    
    # Shutdown
    streaming_service = await get_streaming_service()
    await streaming_service.cleanup()


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
    
    # === STREAMING ENDPOINTS ===
    
    @app.post("/v1/audio/speech/stream")
    async def stream_speech(request: Dict[str, Any]):
        """
        OpenAI-compatible streaming TTS endpoint
        Returns audio as streaming chunks
        """
        try:
            # Map OpenAI request to our format
            tts_request = TTSRequest(
                text=request["input"],
                voice=request.get("voice", "af_bella"),
                format=AudioFormat(request.get("format", "wav")),
                model=request.get("model", "tts-1"),
                speed=request.get("speed", 1.0)
            )
            
            chunk_size = request.get("chunk_size", 4096)
            
            async def audio_stream() -> AsyncGenerator[bytes, None]:
                """Generate streaming audio chunks"""
                streaming_service = await get_streaming_service()
                
                async for chunk in streaming_service.stream_synthesis(
                    request=tts_request,
                    chunk_size=chunk_size,
                    streaming_format=StreamingFormat.WAV_CHUNKS
                ):
                    yield chunk.data
            
            return StreamingResponse(
                audio_stream(),
                media_type=f"audio/{tts_request.format.value}",
                headers={
                    "Transfer-Encoding": "chunked",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/v1/audio/speech/sse")
    async def stream_speech_sse(request: Dict[str, Any]):
        """
        Server-Sent Events streaming TTS endpoint
        Returns audio chunks encoded as base64 in SSE format
        """
        try:
            # Map OpenAI request to our format
            tts_request = TTSRequest(
                text=request["input"],
                voice=request.get("voice", "af_bella"),
                format=AudioFormat(request.get("format", "wav")),
                model=request.get("model", "tts-1"),
                speed=request.get("speed", 1.0)
            )
            
            chunk_size = request.get("chunk_size", 4096)
            
            streaming_service = await get_streaming_service()
            
            return EventSourceResponse(
                streaming_service.create_sse_stream(
                    request=tts_request,
                    chunk_size=chunk_size
                ),
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.websocket("/v1/audio/speech/ws")
    async def websocket_speech(websocket: WebSocket):
        """
        WebSocket endpoint for bidirectional streaming TTS
        """
        await websocket.accept()
        
        try:
            streaming_service = await get_streaming_service()
            
            while True:
                # Receive request from client
                data = await websocket.receive_text()
                request_data = json.loads(data)
                
                # Validate request type
                if request_data.get("type") == "synthesize":
                    # Create TTS request
                    tts_request = TTSRequest(
                        text=request_data["input"],
                        voice=request_data.get("voice", "af_bella"),
                        format=AudioFormat(request_data.get("format", "wav")),
                        speed=request_data.get("speed", 1.0)
                    )
                    
                    chunk_size = request_data.get("chunk_size", 4096)
                    
                    # Send status message
                    await websocket.send_text(json.dumps({
                        "type": "status",
                        "status": "starting",
                        "message": "Starting synthesis..."
                    }))
                    
                    # Stream audio chunks
                    async for chunk in streaming_service.stream_synthesis(
                        request=tts_request,
                        chunk_size=chunk_size,
                        streaming_format=StreamingFormat.WAV_CHUNKS
                    ):
                        # Send audio chunk as base64
                        audio_b64 = base64.b64encode(chunk.data).decode('utf-8')
                        
                        message = {
                            "type": "audio_chunk",
                            "chunk_id": chunk.chunk_id,
                            "audio_data": audio_b64,
                            "format": chunk.format.value,
                            "sample_rate": chunk.sample_rate,
                            "is_final": chunk.is_final,
                            "timestamp": chunk.timestamp,
                            "metadata": chunk.metadata
                        }
                        
                        await websocket.send_text(json.dumps(message))
                    
                    # Send completion message
                    await websocket.send_text(json.dumps({
                        "type": "complete",
                        "status": "finished",
                        "message": "Synthesis completed"
                    }))
                
                elif request_data.get("type") == "ping":
                    # Handle ping for keepalive
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": request_data.get("timestamp")
                    }))
                
                else:
                    # Unknown request type
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": f"Unknown request type: {request_data.get('type')}"
                    }))
                    
        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": str(e)
                }))
            except:
                pass
    
    @app.get("/v1/audio/streams")
    async def list_active_streams():
        """List active streaming sessions"""
        streaming_service = await get_streaming_service()
        streams = await streaming_service.list_active_streams()
        
        return {
            "streams": streams,
            "total_active": len(streams)
        }
    
    @app.delete("/v1/audio/streams/{stream_id}")
    async def cancel_stream(stream_id: str):
        """Cancel an active stream"""
        streaming_service = await get_streaming_service()
        success = await streaming_service.cancel_stream(stream_id)
        
        if success:
            return {"message": f"Stream {stream_id} cancelled"}
        else:
            raise HTTPException(status_code=404, detail="Stream not found")
    
    return app