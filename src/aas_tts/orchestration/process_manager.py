"""
Applied AI Systems - Process Manager
Abstraction layer for process orchestration (Circus vs Docker)
"""
import asyncio
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass

from loguru import logger

from ..core.config import get_settings


class ProcessBackend(str, Enum):
    """Available process orchestration backends"""
    SIMPLE = "simple"  # Simple subprocess backend (default)
    CIRCUS = "circus"
    DOCKER = "docker"


@dataclass
class ProcessInfo:
    """Information about a managed process"""
    name: str
    status: str  # running, stopped, failed, etc.
    pid: Optional[int] = None
    uptime: Optional[float] = None
    memory_usage: Optional[int] = None
    cpu_usage: Optional[float] = None
    port: Optional[int] = None
    health_url: Optional[str] = None
    last_health_check: Optional[str] = None
    restart_count: int = 0


class ProcessBackendInterface(ABC):
    """Abstract interface for process orchestration backends"""
    
    @abstractmethod
    async def start_process(self, name: str, command: List[str], **kwargs) -> bool:
        """Start a process"""
        pass
    
    @abstractmethod
    async def stop_process(self, name: str) -> bool:
        """Stop a process"""
        pass
    
    @abstractmethod
    async def restart_process(self, name: str) -> bool:
        """Restart a process"""
        pass
    
    @abstractmethod
    async def get_process_info(self, name: str) -> Optional[ProcessInfo]:
        """Get process information"""
        pass
    
    @abstractmethod
    async def list_processes(self) -> List[ProcessInfo]:
        """List all managed processes"""
        pass
    
    @abstractmethod
    async def is_process_healthy(self, name: str) -> bool:
        """Check if process is healthy"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass


class ProcessManager:
    """Main process manager that abstracts backend implementation"""
    
    def __init__(self, backend: ProcessBackend = ProcessBackend.SIMPLE):
        self.settings = get_settings()
        self.backend_type = backend
        self._backend: Optional[ProcessBackendInterface] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the process manager"""
        if self._initialized:
            return
        
        logger.info(f"Initializing ProcessManager with {self.backend_type.value} backend...")
        
        try:
            if self.backend_type == ProcessBackend.SIMPLE:
                from .simple_backend import SimpleBackend
                self._backend = SimpleBackend()
            elif self.backend_type == ProcessBackend.CIRCUS:
                from .circus_backend import CircusBackend
                self._backend = CircusBackend()
            elif self.backend_type == ProcessBackend.DOCKER:
                from .docker_backend import DockerBackend
                self._backend = DockerBackend()
            else:
                raise ValueError(f"Unknown backend: {self.backend_type}")
            
            await self._backend.initialize()
            
            self._initialized = True
            logger.info("ProcessManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ProcessManager: {e}")
            raise
    
    async def start_tts_api(self, port: int = 8880, **kwargs) -> bool:
        """
        Start the TTS API server
        
        Args:
            port: Port to run the API server on
            **kwargs: Additional configuration
            
        Returns:
            True if started successfully
        """
        if not self._initialized:
            await self.initialize()
        
        # Build command to start TTS API
        command = [
            "uvicorn",
            "aas_tts.api.server:create_app",
            "--factory",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--workers", "1"
        ]
        
        # Add additional parameters
        if kwargs.get("reload", False):
            command.append("--reload")
        
        logger.info(f"Starting TTS API on port {port}...")
        
        success = await self._backend.start_process(
            name="aas-tts-api",
            command=command,
            port=port,
            health_url=f"http://localhost:{port}/health",
            **kwargs
        )
        
        if success:
            logger.info(f"TTS API started successfully on port {port}")
            
            # Wait for health check
            await self._wait_for_health("aas-tts-api", timeout=30)
        else:
            logger.error("Failed to start TTS API")
        
        return success
    
    async def stop_tts_api(self) -> bool:
        """Stop the TTS API server"""
        if not self._initialized:
            await self.initialize()
        
        logger.info("Stopping TTS API...")
        success = await self._backend.stop_process("aas-tts-api")
        
        if success:
            logger.info("TTS API stopped successfully")
        else:
            logger.error("Failed to stop TTS API")
        
        return success
    
    async def restart_tts_api(self) -> bool:
        """Restart the TTS API server"""
        if not self._initialized:
            await self.initialize()
        
        logger.info("Restarting TTS API...")
        success = await self._backend.restart_process("aas-tts-api")
        
        if success:
            logger.info("TTS API restarted successfully")
            await self._wait_for_health("aas-tts-api", timeout=30)
        else:
            logger.error("Failed to restart TTS API")
        
        return success
    
    async def get_tts_api_status(self) -> Optional[ProcessInfo]:
        """Get TTS API status"""
        if not self._initialized:
            await self.initialize()
        
        return await self._backend.get_process_info("aas-tts-api")
    
    async def is_tts_api_healthy(self) -> bool:
        """Check if TTS API is healthy"""
        if not self._initialized:
            await self.initialize()
        
        return await self._backend.is_process_healthy("aas-tts-api")
    
    async def ensure_tts_api_running(self, port: int = 8880) -> bool:
        """
        Ensure TTS API is running and healthy
        
        Args:
            port: Port for the API server
            
        Returns:
            True if API is running and healthy
        """
        if not self._initialized:
            await self.initialize()
        
        # Check if already running and healthy
        if await self.is_tts_api_healthy():
            logger.debug("TTS API is already running and healthy")
            return True
        
        # Try to start
        logger.info("TTS API not healthy, attempting to start...")
        return await self.start_tts_api(port=port)
    
    async def _wait_for_health(self, process_name: str, timeout: int = 30) -> bool:
        """Wait for a process to become healthy"""
        logger.debug(f"Waiting for {process_name} to become healthy...")
        
        for i in range(timeout):
            if await self._backend.is_process_healthy(process_name):
                logger.debug(f"{process_name} is healthy after {i+1} seconds")
                return True
            
            await asyncio.sleep(1)
        
        logger.warning(f"{process_name} did not become healthy within {timeout} seconds")
        return False
    
    async def shutdown(self) -> None:
        """Shutdown the process manager"""
        if not self._initialized:
            return
        
        logger.info("Shutting down ProcessManager...")
        
        try:
            # Stop all processes
            await self.stop_tts_api()
            
            # Cleanup backend
            if self._backend:
                await self._backend.cleanup()
            
            self._initialized = False
            logger.info("ProcessManager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during ProcessManager shutdown: {e}")


# Global process manager instance
_process_manager: Optional[ProcessManager] = None


async def get_process_manager(backend: ProcessBackend = ProcessBackend.SIMPLE) -> ProcessManager:
    """Get global process manager instance"""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager(backend=backend)
        await _process_manager.initialize()
    return _process_manager