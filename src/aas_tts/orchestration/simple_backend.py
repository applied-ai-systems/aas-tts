"""
Applied AI Systems - Simple Backend
Simple subprocess-based process orchestration backend
"""
import asyncio
import subprocess
import signal
import psutil
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import httpx
from loguru import logger

from .process_manager import ProcessBackendInterface, ProcessInfo


class SimpleBackend(ProcessBackendInterface):
    """Simple subprocess-based process orchestration backend"""
    
    def __init__(self):
        self.processes: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Simple backend"""
        if self._initialized:
            return
        
        logger.info("Initializing Simple backend...")
        
        try:
            self._initialized = True
            logger.info("Simple backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Simple backend: {e}")
            raise
    
    async def start_process(self, name: str, command: List[str], **kwargs) -> bool:
        """Start a process using subprocess"""
        try:
            logger.info(f"Starting process {name}: {' '.join(command)}")
            
            # Start process
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Store process metadata
            self.processes[name] = {
                "process": process,
                "command": command,
                "port": kwargs.get("port"),
                "health_url": kwargs.get("health_url"),
                "restart_count": 0,
                "status": "running",
                "pid": process.pid
            }
            
            logger.info(f"Process {name} started successfully with PID {process.pid}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting process {name}: {e}")
            if name in self.processes:
                self.processes[name]["status"] = "failed"
            return False
    
    async def stop_process(self, name: str) -> bool:
        """Stop a process"""
        if name not in self.processes:
            logger.warning(f"Process {name} not found")
            return False
        
        try:
            process = self.processes[name]["process"]
            
            if process.returncode is None:  # Process is still running
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            
            self.processes[name]["status"] = "stopped"
            logger.info(f"Process {name} stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping process {name}: {e}")
            return False
    
    async def restart_process(self, name: str) -> bool:
        """Restart a process"""
        if name not in self.processes:
            logger.warning(f"Process {name} not found")
            return False
        
        try:
            # Stop the process
            await self.stop_process(name)
            
            # Get original command and kwargs
            command = self.processes[name]["command"]
            port = self.processes[name]["port"]
            health_url = self.processes[name]["health_url"]
            
            # Start again
            success = await self.start_process(
                name=name,
                command=command,
                port=port,
                health_url=health_url
            )
            
            if success:
                self.processes[name]["restart_count"] += 1
                logger.info(f"Process {name} restarted successfully")
            
            return success
            
        except Exception as e:
            logger.error(f"Error restarting process {name}: {e}")
            return False
    
    async def get_process_info(self, name: str) -> Optional[ProcessInfo]:
        """Get process information"""
        if name not in self.processes:
            return None
        
        try:
            process_data = self.processes[name]
            process = process_data["process"]
            
            # Check if process is still running
            if process.returncode is None:
                status = "running"
            else:
                status = "stopped"
                process_data["status"] = status
            
            return ProcessInfo(
                name=name,
                status=status,
                port=process_data.get("port"),
                health_url=process_data.get("health_url"),
                restart_count=process_data.get("restart_count", 0),
                pid=process_data.get("pid"),
                uptime=None,
                memory_usage=None,
                cpu_usage=None,
                last_health_check=None
            )
            
        except Exception as e:
            logger.error(f"Error getting process info for {name}: {e}")
            return None
    
    async def list_processes(self) -> List[ProcessInfo]:
        """List all managed processes"""
        processes = []
        for name in self.processes.keys():
            info = await self.get_process_info(name)
            if info:
                processes.append(info)
        return processes
    
    async def is_process_healthy(self, name: str) -> bool:
        """Check if process is healthy via health check URL"""
        if name not in self.processes:
            return False
        
        process_info = await self.get_process_info(name)
        if not process_info or process_info.status != "running":
            return False
        
        health_url = self.processes[name].get("health_url")
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
        """Clean up Simple backend resources"""
        if not self._initialized:
            return
        
        logger.info("Cleaning up Simple backend...")
        
        try:
            # Stop all processes
            for name in list(self.processes.keys()):
                await self.stop_process(name)
            
            self._initialized = False
            logger.info("Simple backend cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Simple cleanup: {e}")