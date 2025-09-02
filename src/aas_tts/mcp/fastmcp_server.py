"""
Applied AI Systems - FastMCP Server
Modern MCP server implementation using FastMCP framework
"""
import asyncio
import base64
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel
from loguru import logger

from ..core.models import TTSRequest, AudioFormat
from ..core.config import get_settings
from ..services.tts_service import get_tts_service
from ..services.streaming_service import get_streaming_service
from ..orchestration.process_manager import ProcessManager, ProcessBackend


# Create FastMCP server
mcp = FastMCP("AAS-TTS", version="1.0.0")


class SynthesisRequest(BaseModel):
    """Request model for TTS synthesis"""
    text: str
    voice: str = "af_bella"
    format: str = "wav"
    speed: float = 1.0
    output_file: Optional[str] = None
    play_audio: bool = True  # Play through system speakers by default


class StreamingRequest(BaseModel):
    """Request model for streaming TTS synthesis"""
    text: str
    voice: str = "af_bella"
    format: str = "wav"
    speed: float = 1.0
    chunk_size: int = 4096


class ProcessRequest(BaseModel):
    """Request model for process management"""
    backend: str = "simple"


@mcp.tool
async def synthesize_speech(request: SynthesisRequest) -> Dict[str, Any]:
    """
    Synthesize text to speech and return audio data or save to file
    
    This tool converts text to natural-sounding speech using advanced TTS models.
    It supports multiple voices, formats, system speaker playback, and can save directly to files.
    
    Args:
        request: Synthesis parameters including text, voice, format, speed, playback option, and optional output file
        
    Returns:
        Dictionary with synthesis results including success status, audio data (base64), 
        processing time, and file path if saved
    """
    try:
        logger.info(f"FastMCP synthesis request: {request.text[:50]}...")
        
        # Get TTS service
        service = await get_tts_service()
        
        # Create TTS request
        tts_request = TTSRequest(
            text=request.text,
            voice=request.voice,
            format=AudioFormat(request.format),
            speed=request.speed
        )
        
        if request.output_file:
            # Synthesize to file
            output_path = Path(request.output_file)
            response = await service.synthesize_to_file(
                request=tts_request,
                output_path=output_path,
                overwrite=True
            )
            
            return {
                "success": response.success,
                "audio_path": str(response.audio_path) if response.audio_path else None,
                "processing_time": response.processing_time,
                "audio_duration": response.audio_duration,
                "error": response.error if not response.success else None
            }
        else:
            # Synthesize to memory
            response = await service.synthesize(tts_request)
            
            if response.success:
                # Encode audio data as base64
                audio_b64 = base64.b64encode(response.audio_data).decode('utf-8')
                
                result = {
                    "success": True,
                    "audio_data": audio_b64,
                    "audio_size": len(response.audio_data),
                    "processing_time": response.processing_time,
                    "audio_duration": response.audio_duration,
                    "format": response.format.value,
                    "sample_rate": response.sample_rate,
                    "voice": response.voice
                }
                
                # Play audio through system speakers if requested
                if request.play_audio:
                    try:
                        from ..services.audio_playback_service import get_audio_playback_service, PlaybackRequest
                        
                        playback_service = await get_audio_playback_service()
                        playback_request = PlaybackRequest(
                            audio_data=response.audio_data,
                            format=request.format
                        )
                        
                        playback_response = await playback_service.play_audio(playback_request)
                        
                        result["playback"] = {
                            "success": playback_response.success,
                            "backend_used": playback_response.backend_used,
                            "error": playback_response.error if not playback_response.success else None
                        }
                        
                        logger.info(f"Audio played through {playback_response.backend_used} speakers: {playback_response.success}")
                        
                    except Exception as e:
                        logger.error(f"Audio playback failed: {e}")
                        result["playback"] = {
                            "success": False,
                            "error": str(e)
                        }
                
                return result
            else:
                return {
                    "success": False,
                    "error": response.error
                }
                
    except Exception as e:
        logger.error(f"FastMCP synthesis error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool
async def stream_synthesis(request: StreamingRequest) -> Dict[str, Any]:
    """
    Stream text-to-speech synthesis in real-time chunks
    
    This tool provides real-time streaming TTS synthesis, allowing you to start
    receiving audio data before the entire text has been processed. Ideal for 
    long texts or real-time applications.
    
    Args:
        request: Streaming parameters including text, voice, format, speed, and chunk size
        
    Returns:
        Dictionary with streaming session information and the first audio chunk
    """
    try:
        logger.info(f"FastMCP streaming request: {request.text[:50]}...")
        
        # Get streaming service  
        service = await get_streaming_service()
        
        # Create TTS request
        tts_request = TTSRequest(
            text=request.text,
            voice=request.voice,
            format=AudioFormat(request.format),
            speed=request.speed
        )
        
        # Start streaming synthesis and collect first few chunks
        chunks = []
        chunk_count = 0
        
        async for chunk in service.stream_synthesis(
            request=tts_request,
            chunk_size=request.chunk_size
        ):
            # Encode chunk data as base64
            chunk_b64 = base64.b64encode(chunk.data).decode('utf-8')
            
            chunks.append({
                "chunk_id": chunk.chunk_id,
                "audio_data": chunk_b64,
                "size": len(chunk.data),
                "is_final": chunk.is_final,
                "timestamp": chunk.timestamp
            })
            
            chunk_count += 1
            
            # For MCP tool response, return summary with first few chunks
            if chunk_count >= 3 or chunk.is_final:
                break
        
        return {
            "success": True,
            "streaming_session": f"stream_{int(chunks[0]['timestamp'] * 1000) if chunks else 'unknown'}",
            "total_chunks_preview": len(chunks),
            "chunk_size": request.chunk_size,
            "format": request.format,
            "voice": request.voice,
            "sample_chunks": chunks[:3],  # Return first 3 chunks as preview
            "message": "Streaming synthesis started. Use the streaming endpoints for full real-time access."
        }
        
    except Exception as e:
        logger.error(f"FastMCP streaming error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool
async def list_voices() -> Dict[str, Any]:
    """
    List all available TTS voices
    
    Returns information about all available voices including their IDs, names,
    categories, languages, and availability status.
    
    Returns:
        Dictionary with list of available voices and their metadata
    """
    try:
        service = await get_tts_service()
        voices = await service.get_available_voices()
        
        return {
            "success": True,
            "total_voices": len(voices),
            "voices": [
                {
                    "id": voice.id,
                    "name": voice.name,
                    "category": voice.category.value,
                    "language": voice.language,
                    "gender": voice.gender,
                    "available": voice.available,
                    "sample_rate": voice.sample_rate,
                    "description": voice.description
                }
                for voice in voices
            ]
        }
        
    except Exception as e:
        logger.error(f"FastMCP list voices error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool
async def get_voice_info(voice_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific voice
    
    Provides comprehensive information about a voice including its characteristics,
    capabilities, and availability status.
    
    Args:
        voice_id: The unique identifier for the voice (e.g., 'af_bella', 'am_adam')
        
    Returns:
        Dictionary with detailed voice information
    """
    try:
        service = await get_tts_service()
        voice = await service.get_voice_info(voice_id)
        
        if voice:
            return {
                "success": True,
                "voice": {
                    "id": voice.id,
                    "name": voice.name,
                    "category": voice.category.value,
                    "language": voice.language,
                    "gender": voice.gender,
                    "available": voice.available,
                    "sample_rate": voice.sample_rate,
                    "description": voice.description
                }
            }
        else:
            return {
                "success": False,
                "error": f"Voice '{voice_id}' not found"
            }
            
    except Exception as e:
        logger.error(f"FastMCP voice info error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool
async def check_api_health() -> Dict[str, Any]:
    """
    Check the health status of the TTS API and services
    
    Performs comprehensive health checks on all TTS components including
    service availability, voice loading, and basic synthesis testing.
    
    Returns:
        Dictionary with detailed health information
    """
    try:
        service = await get_tts_service()
        health_info = await service.health_check()
        
        return {
            "success": True,
            "health": health_info
        }
        
    except Exception as e:
        logger.error(f"FastMCP health check error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool
async def test_audio_playback() -> Dict[str, Any]:
    """
    Test system speaker audio playback functionality
    
    This tool tests the audio playback system by playing a simple test tone
    through the system speakers. Useful for verifying audio setup.
    
    Returns:
        Dictionary with test results including available backends and playback status
    """
    try:
        from ..services.audio_playback_service import get_audio_playback_service
        
        logger.info("Testing audio playback system...")
        
        # Get playback service
        service = await get_audio_playback_service()
        
        # Get available backends
        backends = await service.get_available_backends()
        
        # Run playback test
        test_result = await service.test_playback()
        
        return {
            "success": test_result["success"],
            "available_backends": backends,
            "test_result": test_result,
            "message": "Audio playback test completed successfully" if test_result["success"] else "Audio playback test failed"
        }
        
    except Exception as e:
        logger.error(f"Audio playback test error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


@mcp.tool
async def manage_api_server(request: ProcessRequest) -> Dict[str, Any]:
    """
    Manage the TTS API server process lifecycle
    
    Provides process management capabilities for the TTS API server including
    starting, stopping, and checking status using different orchestration backends.
    
    Args:
        request: Process management parameters including backend type
        
    Returns:
        Dictionary with process management results
    """
    try:
        # Get backend enum
        if request.backend == "circus":
            backend = ProcessBackend.CIRCUS
        elif request.backend == "docker":
            backend = ProcessBackend.DOCKER
        else:
            backend = ProcessBackend.SIMPLE
        
        # Create process manager
        manager = ProcessManager(backend=backend)
        await manager.initialize()
        
        # Check if TTS API is running
        is_running = await manager.is_process_running("tts-api")
        
        if not is_running:
            # Start TTS API
            success = await manager.ensure_tts_api_running()
            
            if success:
                return {
                    "success": True,
                    "action": "started",
                    "backend": request.backend,
                    "message": "TTS API server started successfully"
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to start TTS API server"
                }
        else:
            return {
                "success": True,
                "action": "already_running",
                "backend": request.backend,
                "message": "TTS API server is already running"
            }
            
    except Exception as e:
        logger.error(f"FastMCP process management error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# Resources for providing additional context
@mcp.resource("tts://config")
async def tts_config() -> str:
    """Current TTS service configuration and settings"""
    settings = get_settings()
    
    config_info = f"""# AAS-TTS Configuration

## Service Settings
- Device: {settings.device}
- Model Path: {settings.model_path}
- Default Voice: {settings.default_voice}
- Sample Rate: {settings.sample_rate}

## API Settings  
- Host: {settings.api_host}
- Port: {settings.api_port}
- Debug: {settings.debug}

## CORS Settings
- Origins: {settings.cors_origins}
- Methods: {settings.cors_methods}
- Headers: {settings.cors_headers}

## Streaming Settings
- Chunk Size: {getattr(settings, 'chunk_size', 4096)}
- Supported Formats: WAV, MP3
- Transport Protocols: HTTP, SSE, WebSocket
"""
    
    return config_info


@mcp.resource("tts://examples")
async def tts_examples() -> str:
    """Example usage patterns and code snippets for TTS operations"""
    
    examples = """# AAS-TTS Usage Examples

## Basic Synthesis
```python
# Synthesize text to audio file
result = await synthesize_speech({
    "text": "Hello, world! This is a test of our TTS system.",
    "voice": "af_bella",
    "format": "wav",
    "speed": 1.0,
    "output_file": "output.wav"
})
```

## Streaming Synthesis
```python
# Start streaming synthesis for long text
result = await stream_synthesis({
    "text": "This is a long piece of text that will be synthesized in real-time chunks...",
    "voice": "am_adam", 
    "format": "wav",
    "chunk_size": 4096
})
```

## Voice Management
```python
# List available voices
voices = await list_voices()

# Get specific voice info
voice_info = await get_voice_info("bf_emma")
```

## Health Monitoring
```python
# Check service health
health = await check_api_health()

# Manage API server
result = await manage_api_server({"backend": "simple"})
```

## OpenAI Compatibility
The TTS API is compatible with OpenAI's TTS endpoints:
- POST /v1/audio/speech - Standard synthesis
- POST /v1/audio/speech/stream - HTTP streaming
- POST /v1/audio/speech/sse - Server-sent events
- WebSocket /v1/audio/speech/ws - Bidirectional streaming
"""
    
    return examples


# Prompts for common TTS workflows
@mcp.prompt("synthesize")
async def synthesize_prompt(text: str, voice: str = "af_bella") -> str:
    """Generate a synthesis request prompt"""
    return f"""I need to synthesize the following text to speech:

Text: "{text}"
Voice: {voice}

Please use the synthesize_speech tool to convert this text to audio. Use WAV format for best quality."""


@mcp.prompt("voice_selection")  
async def voice_selection_prompt(language: str = "en", gender: str = "any") -> str:
    """Generate a voice selection prompt"""
    return f"""I need help selecting an appropriate voice for TTS synthesis.

Requirements:
- Language: {language}
- Gender preference: {gender}

Please use the list_voices tool to show me available options, then recommend the best voice based on these criteria."""


# Run FastMCP server directly
def run_fastmcp_server():
    """Run the FastMCP server in stdio mode"""
    logger.info("Starting AAS-TTS FastMCP server...")
    # FastMCP will handle service initialization through tool calls
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_fastmcp_server()