"""
Applied AI Systems - Audio Playback Service
System speaker playback for TTS audio output
"""
import asyncio
import io
import tempfile
import wave
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import Enum
import platform

from loguru import logger
from pydantic import BaseModel

from ..core.config import get_settings


class PlaybackBackend(str, Enum):
    """Available audio playback backends"""
    AUTO = "auto"           # Auto-select best backend
    PYGAME = "pygame"       # pygame (cross-platform, reliable)
    SIMPLEAUDIO = "simpleaudio"  # simpleaudio (lightweight)
    PLAYSOUND = "playsound"      # playsound (simple)
    SYSTEM = "system"            # System command (afplay/aplay)


class PlaybackRequest(BaseModel):
    """Request for audio playback"""
    audio_data: bytes
    format: str = "wav"
    volume: float = 1.0
    backend: PlaybackBackend = PlaybackBackend.AUTO


class PlaybackResponse(BaseModel):
    """Response from audio playback"""
    success: bool
    backend_used: str
    duration: Optional[float] = None
    error: Optional[str] = None


class AudioPlaybackService:
    """Service for playing audio through system speakers"""
    
    def __init__(self):
        self.settings = get_settings()
        self._initialized = False
        self._available_backends: List[PlaybackBackend] = []
        self._preferred_backend: Optional[PlaybackBackend] = None
        
    async def initialize(self) -> None:
        """Initialize the audio playback service"""
        if self._initialized:
            return
            
        logger.info("Initializing audio playback service...")
        
        # Test available backends
        await self._detect_available_backends()
        
        # Select preferred backend
        self._select_preferred_backend()
        
        self._initialized = True
        logger.info(f"Audio playback service initialized with backend: {self._preferred_backend}")
    
    async def _detect_available_backends(self) -> None:
        """Detect which audio backends are available on this system"""
        self._available_backends = []
        
        # Test pygame
        try:
            import pygame.mixer
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=512)
            pygame.mixer.init()
            pygame.mixer.quit()
            self._available_backends.append(PlaybackBackend.PYGAME)
            logger.debug("pygame backend available")
        except Exception as e:
            logger.debug(f"pygame backend not available: {e}")
        
        # Test simpleaudio
        try:
            import simpleaudio
            self._available_backends.append(PlaybackBackend.SIMPLEAUDIO)
            logger.debug("simpleaudio backend available")
        except Exception as e:
            logger.debug(f"simpleaudio backend not available: {e}")
        
        # Test playsound
        try:
            from playsound import playsound
            self._available_backends.append(PlaybackBackend.PLAYSOUND)
            logger.debug("playsound backend available")
        except Exception as e:
            logger.debug(f"playsound backend not available: {e}")
        
        # System commands are usually available
        system = platform.system().lower()
        if system in ["darwin", "linux", "windows"]:
            self._available_backends.append(PlaybackBackend.SYSTEM)
            logger.debug("system backend available")
    
    def _select_preferred_backend(self) -> None:
        """Select the preferred backend based on platform and availability"""
        if not self._available_backends:
            logger.warning("No audio playback backends available")
            return
        
        # Preference order by platform
        system = platform.system().lower()
        
        if system == "darwin":  # macOS
            preferences = [PlaybackBackend.SIMPLEAUDIO, PlaybackBackend.SYSTEM, PlaybackBackend.PYGAME]
        elif system == "linux":
            preferences = [PlaybackBackend.PYGAME, PlaybackBackend.SIMPLEAUDIO, PlaybackBackend.SYSTEM]
        else:  # Windows and others
            preferences = [PlaybackBackend.PYGAME, PlaybackBackend.SIMPLEAUDIO, PlaybackBackend.PLAYSOUND]
        
        # Select first available preferred backend
        for backend in preferences:
            if backend in self._available_backends:
                self._preferred_backend = backend
                return
        
        # Fallback to first available
        self._preferred_backend = self._available_backends[0]
    
    async def play_audio(self, request: PlaybackRequest) -> PlaybackResponse:
        """
        Play audio through system speakers
        
        Args:
            request: Playback request with audio data and options
            
        Returns:
            PlaybackResponse with success status and details
        """
        if not self._initialized:
            await self.initialize()
        
        # Select backend
        backend = request.backend
        if backend == PlaybackBackend.AUTO:
            backend = self._preferred_backend
            
        if backend not in self._available_backends:
            return PlaybackResponse(
                success=False,
                backend_used="none",
                error=f"Backend {backend} not available"
            )
        
        # Try the selected backend first, then fallback to others
        backends_to_try = [backend]
        
        # If the selected backend fails, try all other available backends
        if backend in self._available_backends:
            other_backends = [b for b in self._available_backends if b != backend]
            backends_to_try.extend(other_backends)
        
        last_error = None
        
        for try_backend in backends_to_try:
            logger.info(f"Playing audio using {try_backend} backend...")
            
            try:
                if try_backend == PlaybackBackend.PYGAME:
                    response = await self._play_with_pygame(request)
                elif try_backend == PlaybackBackend.SIMPLEAUDIO:
                    response = await self._play_with_simpleaudio(request)
                elif try_backend == PlaybackBackend.PLAYSOUND:
                    response = await self._play_with_playsound(request)
                elif try_backend == PlaybackBackend.SYSTEM:
                    response = await self._play_with_system(request)
                else:
                    continue
                
                # If successful, return immediately
                if response.success:
                    return response
                else:
                    last_error = response.error
                    logger.warning(f"Backend {try_backend} failed: {last_error}")
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Audio playback failed with {try_backend}: {e}")
                continue
        
        # All backends failed
        return PlaybackResponse(
            success=False,
            backend_used="all_failed",
            error=f"All audio backends failed. Last error: {last_error}"
        )
    
    async def _play_with_pygame(self, request: PlaybackRequest) -> PlaybackResponse:
        """Play audio using pygame backend"""
        import pygame.mixer
        
        # Initialize mixer with better settings for TTS audio
        pygame.mixer.pre_init(frequency=22050, size=-16, channels=1, buffer=1024)
        pygame.mixer.init()
        
        try:
            # For WAV files, try using Sound instead of music for better compatibility
            if request.format.lower() == "wav":
                try:
                    # Create audio buffer
                    audio_buffer = io.BytesIO(request.audio_data)
                    sound = pygame.mixer.Sound(audio_buffer)
                    sound.set_volume(request.volume)
                    
                    # Play sound
                    channel = sound.play()
                    
                    # Wait for playback to complete
                    while channel.get_busy():
                        await asyncio.sleep(0.1)
                    
                    return PlaybackResponse(
                        success=True,
                        backend_used="pygame"
                    )
                except Exception as e:
                    logger.debug(f"pygame Sound failed: {e}, trying music loader...")
                    # Fallback to music loader
                    pass
            
            # Fallback to music loader
            audio_buffer = io.BytesIO(request.audio_data)
            pygame.mixer.music.load(audio_buffer)
            pygame.mixer.music.set_volume(request.volume)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
            
            return PlaybackResponse(
                success=True,
                backend_used="pygame"
            )
            
        finally:
            pygame.mixer.quit()
    
    async def _play_with_simpleaudio(self, request: PlaybackRequest) -> PlaybackResponse:
        """Play audio using simpleaudio backend"""
        import simpleaudio as sa
        
        # For simpleaudio, we need raw PCM data
        if request.format.lower() == "wav":
            # Extract PCM data from WAV file
            with io.BytesIO(request.audio_data) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    sample_rate = wav_file.getframerate()
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
        else:
            # Assume raw PCM data
            frames = request.audio_data
            sample_rate = 22050
            channels = 1
            sample_width = 2
        
        # Play audio
        play_obj = sa.play_buffer(frames, channels, sample_width, sample_rate)
        
        # Wait for playback to complete (in executor to not block event loop)
        await asyncio.get_event_loop().run_in_executor(None, play_obj.wait_done)
        
        return PlaybackResponse(
            success=True,
            backend_used="simpleaudio"
        )
    
    async def _play_with_playsound(self, request: PlaybackRequest) -> PlaybackResponse:
        """Play audio using playsound backend"""
        from playsound import playsound
        
        # playsound requires a file, so write to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{request.format}", delete=False) as temp_file:
            temp_file.write(request.audio_data)
            temp_path = temp_file.name
        
        try:
            # Play in executor to not block event loop
            await asyncio.get_event_loop().run_in_executor(
                None, playsound, temp_path
            )
            
            return PlaybackResponse(
                success=True,
                backend_used="playsound"
            )
            
        finally:
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)
    
    async def _play_with_system(self, request: PlaybackRequest) -> PlaybackResponse:
        """Play audio using system commands"""
        system = platform.system().lower()
        
        # Write audio to temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{request.format}", delete=False) as temp_file:
            temp_file.write(request.audio_data)
            temp_path = temp_file.name
        
        try:
            # Select system command based on platform
            if system == "darwin":  # macOS
                cmd = ["afplay", temp_path]
            elif system == "linux":
                # Try multiple options
                for player in ["aplay", "paplay", "play"]:
                    try:
                        cmd = [player, temp_path]
                        break
                    except:
                        continue
                else:
                    cmd = ["aplay", temp_path]  # Default fallback
            elif system == "windows":
                # Use PowerShell to play audio
                cmd = ["powershell", "-c", f"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()"]
            else:
                raise ValueError(f"Unsupported system: {system}")
            
            # Run command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode == 0:
                return PlaybackResponse(
                    success=True,
                    backend_used="system"
                )
            else:
                return PlaybackResponse(
                    success=False,
                    backend_used="system",
                    error=stderr.decode() if stderr else "System command failed"
                )
                
        finally:
            # Cleanup temp file
            Path(temp_path).unlink(missing_ok=True)
    
    async def get_available_backends(self) -> List[str]:
        """Get list of available playback backends"""
        if not self._initialized:
            await self.initialize()
            
        return [backend.value for backend in self._available_backends]
    
    async def test_playback(self, backend: Optional[PlaybackBackend] = None) -> Dict[str, Any]:
        """Test audio playback with a simple tone"""
        if not self._initialized:
            await self.initialize()
        
        # Generate a simple test tone
        import numpy as np
        
        duration = 0.5  # seconds
        sample_rate = 22050
        frequency = 440  # A4 note
        
        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        # Test playback
        test_request = PlaybackRequest(
            audio_data=wav_buffer.getvalue(),
            backend=backend or PlaybackBackend.AUTO
        )
        
        response = await self.play_audio(test_request)
        
        return {
            "success": response.success,
            "backend_used": response.backend_used,
            "error": response.error,
            "duration": duration,
            "frequency": frequency
        }


# Global audio playback service instance
_playback_service: Optional[AudioPlaybackService] = None


async def get_audio_playback_service() -> AudioPlaybackService:
    """Get global audio playback service instance"""
    global _playback_service
    if _playback_service is None:
        _playback_service = AudioPlaybackService()
        await _playback_service.initialize()
    return _playback_service