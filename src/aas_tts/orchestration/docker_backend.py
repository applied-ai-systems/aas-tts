"""
Applied AI Systems - Docker Backend
Docker-on-Whales based process orchestration backend
"""
import asyncio
import json
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import httpx
from loguru import logger
from python_on_whales import docker, DockerClient
from python_on_whales.exceptions import DockerException

from .process_manager import ProcessBackendInterface, ProcessInfo


class DockerBackend(ProcessBackendInterface):
    """Docker-based process orchestration backend using python-on-whales"""
    
    def __init__(self):
        self.docker_client: Optional[DockerClient] = None
        self.containers: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
        self.image_name = "aas-tts:latest"
    
    async def initialize(self) -> None:
        """Initialize Docker backend"""
        if self._initialized:
            return
        
        logger.info("Initializing Docker backend...")
        
        try:
            # Initialize docker client
            self.docker_client = docker
            
            # Check if Docker is available
            await self._check_docker_available()
            
            # Ensure AAS-TTS Docker image exists
            await self._ensure_docker_image()
            
            self._initialized = True
            logger.info("Docker backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docker backend: {e}")
            raise
    
    async def start_process(self, name: str, command: List[str], **kwargs) -> bool:
        """Start a process using Docker"""
        try:
            container_name = f"aas-tts-{name}"
            
            # Store container metadata
            self.containers[name] = {
                "container_name": container_name,
                "command": command,
                "port": kwargs.get("port"),
                "health_url": kwargs.get("health_url"),
                "restart_count": 0,
                "status": "starting"
            }
            
            # Prepare Docker run parameters
            port = kwargs.get("port", 8880)
            
            run_kwargs = {
                "name": container_name,
                "detach": True,
                "remove": True,  # Auto-remove when stopped
                "publish": [(port, port)],  # Port mapping
                "envs": {
                    "AAS_TTS_SERVER_HOST": "0.0.0.0",
                    "AAS_TTS_SERVER_PORT": str(port),
                    "AAS_TTS_LOG_LEVEL": "INFO"
                }
            }
            
            # Add any additional environment variables
            if "envs" in kwargs:
                run_kwargs["envs"].update(kwargs["envs"])
            
            logger.info(f"Starting Docker container: {container_name}")
            
            # Start container
            container = await self._run_async(
                self.docker_client.run,
                self.image_name,
                command,
                **run_kwargs
            )
            
            if container:
                self.containers[name]["container_id"] = container.id
                self.containers[name]["status"] = "running"
                logger.info(f"Docker container {container_name} started successfully")
                return True
            else:
                self.containers[name]["status"] = "failed"
                return False
                
        except DockerException as e:
            logger.error(f"Docker error starting process {name}: {e}")
            if name in self.containers:
                self.containers[name]["status"] = "failed"
            return False
        except Exception as e:
            logger.error(f"Error starting process {name}: {e}")
            if name in self.containers:
                self.containers[name]["status"] = "failed"
            return False
    
    async def stop_process(self, name: str) -> bool:
        """Stop a process using Docker"""
        if name not in self.containers:
            logger.warning(f"Process {name} not found")
            return False
        
        try:
            container_name = self.containers[name]["container_name"]
            
            # Stop container
            await self._run_async(
                self.docker_client.stop,
                container_name,
                timeout=10
            )
            
            self.containers[name]["status"] = "stopped"
            logger.info(f"Docker container {container_name} stopped successfully")
            return True
            
        except DockerException as e:
            logger.error(f"Docker error stopping process {name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error stopping process {name}: {e}")
            return False
    
    async def restart_process(self, name: str) -> bool:
        """Restart a process using Docker"""
        if name not in self.containers:
            logger.warning(f"Process {name} not found")
            return False
        
        try:
            container_name = self.containers[name]["container_name"]
            
            # Restart container
            await self._run_async(
                self.docker_client.restart,
                container_name,
                timeout=10
            )
            
            self.containers[name]["restart_count"] += 1
            self.containers[name]["status"] = "running"
            logger.info(f"Docker container {container_name} restarted successfully")
            return True
            
        except DockerException as e:
            logger.error(f"Docker error restarting process {name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error restarting process {name}: {e}")
            return False
    
    async def get_process_info(self, name: str) -> Optional[ProcessInfo]:
        """Get process information from Docker"""
        if name not in self.containers:
            return None
        
        try:
            container_name = self.containers[name]["container_name"]
            container_data = self.containers[name]
            
            # Get container info
            container = await self._run_async(
                self.docker_client.inspect,
                container_name
            )
            
            if container:
                # Determine status
                status = "stopped"
                if container.state.running:
                    status = "running"
                elif container.state.restarting:
                    status = "restarting"
                elif container.state.dead:
                    status = "dead"
                
                container_data["status"] = status
                
                return ProcessInfo(
                    name=name,
                    status=status,
                    port=container_data.get("port"),
                    health_url=container_data.get("health_url"),
                    restart_count=container_data.get("restart_count", 0),
                    pid=container.state.pid,
                    uptime=None,  # Could calculate from container.state.started_at
                    memory_usage=None,  # Would need docker stats
                    cpu_usage=None,  # Would need docker stats
                    last_health_check=None
                )
            else:
                return None
                
        except DockerException as e:
            logger.error(f"Docker error getting process info for {name}: {e}")
            # Container might not exist
            self.containers[name]["status"] = "stopped"
            return ProcessInfo(
                name=name,
                status="stopped",
                port=self.containers[name].get("port"),
                restart_count=self.containers[name].get("restart_count", 0)
            )
        except Exception as e:
            logger.error(f"Error getting process info for {name}: {e}")
            return None
    
    async def list_processes(self) -> List[ProcessInfo]:
        """List all managed processes"""
        processes = []
        for name in self.containers.keys():
            info = await self.get_process_info(name)
            if info:
                processes.append(info)
        return processes
    
    async def is_process_healthy(self, name: str) -> bool:
        """Check if process is healthy via health check URL"""
        if name not in self.containers:
            return False
        
        process_info = await self.get_process_info(name)
        if not process_info or process_info.status != "running":
            return False
        
        health_url = self.containers[name].get("health_url")
        if not health_url:
            # If no health URL, just check if running
            return process_info.status == "running"
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                return response.status_code == 200
        except:
            return False
    
    async def cleanup(self) -> None:
        """Clean up Docker resources"""
        if not self._initialized:
            return
        
        logger.info("Cleaning up Docker backend...")
        
        try:
            # Stop and remove all containers
            for name in list(self.containers.keys()):
                await self.stop_process(name)
            
            self._initialized = False
            logger.info("Docker backend cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Docker cleanup: {e}")
    
    async def _check_docker_available(self) -> None:
        """Check if Docker is available"""
        try:
            await self._run_async(self.docker_client.version)
            logger.debug("Docker is available")
        except DockerException as e:
            raise RuntimeError(f"Docker is not available: {e}")
    
    async def _ensure_docker_image(self) -> None:
        """Ensure AAS-TTS Docker image exists"""
        try:
            # Check if image exists
            images = await self._run_async(
                self.docker_client.image.list,
                filters={"reference": self.image_name}
            )
            
            if images:
                logger.debug(f"Docker image {self.image_name} already exists")
                return
            
            # Build image if it doesn't exist
            logger.info(f"Building Docker image: {self.image_name}")
            
            # Create a simple Dockerfile for AAS-TTS
            dockerfile_content = '''
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Install AAS-TTS
COPY . /app
WORKDIR /app
RUN pip install .

# Expose port
EXPOSE 8880

# Default command
CMD ["aas-tts", "server", "start", "--host", "0.0.0.0", "--port", "8880"]
'''
            
            # Note: This is a simplified approach
            # In a real implementation, we'd need the project context
            logger.warning(
                "Docker image building not fully implemented. "
                "Please build the image manually with: docker build -t aas-tts:latest ."
            )
            
        except DockerException as e:
            logger.error(f"Error ensuring Docker image: {e}")
            # Don't fail initialization - image might be built manually
    
    async def _run_async(self, func, *args, **kwargs):
        """Run a docker operation asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)