"""
Applied AI Systems - Streaming Service
Real-time audio streaming with chunked delivery and SSE support
"""
import asyncio
import io
import time
from typing import AsyncGenerator, Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from loguru import logger

from ..core.models import TTSRequest, TTSResponse, AudioFormat
from ..core.config import get_settings


class StreamingFormat(str, Enum):
    """Supported streaming formats"""
    RAW_AUDIO = "audio/raw"
    WAV_CHUNKS = "audio/wav"
    MP3_CHUNKS = "audio/mp3"
    WEBM_OPUS = "audio/webm;codecs=opus"
    SSE_BASE64 = "text/event-stream"


@dataclass
class AudioChunk:
    """Audio chunk for streaming"""
    data: bytes
    chunk_id: int
    total_chunks: Optional[int] = None
    timestamp: float = None
    format: AudioFormat = AudioFormat.WAV
    sample_rate: int = 22050
    is_final: bool = False
    metadata: Optional[Dict[str, Any]] = None


class StreamingService:
    """Service for real-time audio streaming capabilities"""
    
    def __init__(self):
        self.settings = get_settings()
        self._initialized = False
        self.active_streams: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialize streaming service"""
        if self._initialized:
            return
        
        logger.info("Initializing streaming service...")
        
        try:
            self._initialized = True
            logger.info("Streaming service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize streaming service: {e}")
            raise
    
    async def stream_synthesis(
        self,
        request: TTSRequest,
        chunk_size: int = 4096,
        streaming_format: StreamingFormat = StreamingFormat.WAV_CHUNKS
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        Stream TTS synthesis in real-time chunks
        
        Args:
            request: TTS synthesis request
            chunk_size: Size of audio chunks in bytes
            streaming_format: Format for streaming
            
        Yields:
            AudioChunk objects with audio data
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        stream_id = f"stream_{int(start_time * 1000)}"
        
        logger.info(f"Starting streaming synthesis {stream_id}: {request.text[:50]}...")
        
        try:
            # Store stream metadata
            self.active_streams[stream_id] = {
                "request": request,
                "start_time": start_time,
                "chunk_size": chunk_size,
                "format": streaming_format,
                "chunks_sent": 0
            }
            
            # Simulate streaming synthesis (in real implementation, this would be actual TTS)
            async for chunk in self._generate_streaming_audio(
                request=request,
                chunk_size=chunk_size,
                stream_id=stream_id
            ):
                yield chunk
                self.active_streams[stream_id]["chunks_sent"] += 1
            
            logger.info(f"Streaming synthesis {stream_id} completed")
            
        except Exception as e:
            logger.error(f"Streaming synthesis {stream_id} failed: {e}")
            raise
        finally:
            # Cleanup stream
            if stream_id in self.active_streams:
                del self.active_streams[stream_id]
    
    async def _generate_streaming_audio(
        self,
        request: TTSRequest,
        chunk_size: int,
        stream_id: str
    ) -> AsyncGenerator[AudioChunk, None]:
        """Generate streaming audio chunks"""
        
        # Simulate progressive TTS synthesis
        text_chunks = self._split_text_for_streaming(request.text)
        total_text_chunks = len(text_chunks)
        
        chunk_id = 0
        
        for i, text_chunk in enumerate(text_chunks):
            # Simulate processing time for each text chunk
            processing_delay = len(text_chunk) * 0.01  # 10ms per character
            await asyncio.sleep(min(processing_delay, 0.5))  # Cap at 500ms
            
            # Generate audio for this text chunk
            audio_data = await self._synthesize_text_chunk(
                text=text_chunk,
                voice=request.voice,
                format=request.format,
                speed=request.speed
            )
            
            # Split audio into chunks
            audio_chunks = self._split_audio_into_chunks(audio_data, chunk_size)
            
            for j, chunk_data in enumerate(audio_chunks):
                is_final = (i == total_text_chunks - 1) and (j == len(audio_chunks) - 1)
                
                chunk = AudioChunk(
                    data=chunk_data,
                    chunk_id=chunk_id,
                    total_chunks=None,  # Don't know total until finished
                    timestamp=time.time(),
                    format=request.format,
                    sample_rate=request.sample_rate or 22050,
                    is_final=is_final,
                    metadata={
                        "stream_id": stream_id,
                        "text_chunk_index": i,
                        "audio_chunk_index": j,
                        "text_preview": text_chunk[:30] + "..." if len(text_chunk) > 30 else text_chunk
                    }
                )
                
                yield chunk
                chunk_id += 1
                
                # Small delay between chunks for realistic streaming
                await asyncio.sleep(0.01)
    
    def _split_text_for_streaming(self, text: str, max_chunk_length: int = 100) -> List[str]:
        """Split text into chunks suitable for streaming synthesis"""
        
        # Split by sentences first
        import re
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If adding this sentence would exceed max length, start new chunk
            if current_chunk and len(current_chunk + " " + sentence) > max_chunk_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]  # Fallback to original text
    
    async def _synthesize_text_chunk(
        self,
        text: str,
        voice: str,
        format: AudioFormat,
        speed: float
    ) -> bytes:
        """Synthesize a single text chunk (mock implementation)"""
        
        # Mock audio generation - in real implementation, this would call TTS
        duration = len(text) * 0.05 * speed  # Rough estimate
        sample_rate = 22050
        num_samples = int(sample_rate * duration)
        
        # Generate simple sine wave as mock audio
        import numpy as np
        t = np.linspace(0, duration, num_samples)
        
        # Different frequencies for different voices
        voice_frequencies = {
            "af_bella": 440.0,
            "am_adam": 220.0,
            "bf_emma": 523.25,
        }
        freq = voice_frequencies.get(voice, 440.0)
        
        audio = np.sin(2 * np.pi * freq * t) * 0.3
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Convert to bytes based on format
        if format == AudioFormat.WAV:
            return self._create_wav_bytes(audio_int16, sample_rate)
        else:
            # For other formats, return raw PCM for now
            return audio_int16.tobytes()
    
    def _create_wav_bytes(self, audio_data: 'np.ndarray', sample_rate: int) -> bytes:
        """Create WAV file bytes from audio data"""
        import wave
        import numpy as np
        
        buffer = io.BytesIO()
        
        with wave.open(buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())
        
        buffer.seek(0)
        return buffer.getvalue()
    
    def _split_audio_into_chunks(self, audio_data: bytes, chunk_size: int) -> List[bytes]:
        """Split audio data into chunks for streaming"""
        chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            chunks.append(chunk)
        
        return chunks
    
    async def create_sse_stream(
        self,
        request: TTSRequest,
        chunk_size: int = 4096
    ) -> AsyncGenerator[str, None]:
        """
        Create Server-Sent Events stream for audio
        
        Args:
            request: TTS synthesis request
            chunk_size: Size of audio chunks
            
        Yields:
            SSE formatted strings with base64 encoded audio
        """
        import base64
        import json
        
        # Send initial event
        yield f"event: start\ndata: {json.dumps({'status': 'starting', 'request_id': request.model_dump()})}\n\n"
        
        chunk_count = 0
        
        async for chunk in self.stream_synthesis(
            request=request,
            chunk_size=chunk_size,
            streaming_format=StreamingFormat.SSE_BASE64
        ):
            # Encode audio data as base64
            audio_b64 = base64.b64encode(chunk.data).decode('utf-8')
            
            # Create SSE event data
            event_data = {
                "chunk_id": chunk.chunk_id,
                "audio_data": audio_b64,
                "format": chunk.format.value,
                "sample_rate": chunk.sample_rate,
                "timestamp": chunk.timestamp,
                "is_final": chunk.is_final,
                "metadata": chunk.metadata
            }
            
            yield f"event: audio_chunk\ndata: {json.dumps(event_data)}\n\n"
            chunk_count += 1
        
        # Send completion event
        yield f"event: complete\ndata: {json.dumps({'status': 'completed', 'total_chunks': chunk_count})}\n\n"
    
    async def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active stream"""
        return self.active_streams.get(stream_id)
    
    async def list_active_streams(self) -> List[Dict[str, Any]]:
        """List all active streams"""
        return [
            {
                "stream_id": stream_id,
                "start_time": info["start_time"],
                "chunks_sent": info["chunks_sent"],
                "duration": time.time() - info["start_time"],
                "text_preview": info["request"].text[:50] + "..."
            }
            for stream_id, info in self.active_streams.items()
        ]
    
    async def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            logger.info(f"Cancelled stream: {stream_id}")
            return True
        return False
    
    async def cleanup(self) -> None:
        """Cleanup streaming service"""
        logger.info("Cleaning up streaming service...")
        
        # Cancel all active streams
        for stream_id in list(self.active_streams.keys()):
            await self.cancel_stream(stream_id)
        
        self._initialized = False
        logger.info("Streaming service cleanup complete")


# Global streaming service instance
_streaming_service: Optional[StreamingService] = None


async def get_streaming_service() -> StreamingService:
    """Get global streaming service instance"""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingService()
        await _streaming_service.initialize()
    return _streaming_service