"""
Applied AI Systems - Circus Backend
Circus-based process orchestration backend
"""
import asyncio
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict

import httpx
from loguru import logger

from .process_manager import ProcessBackendInterface, ProcessInfo


class CircusBackend(ProcessBackendInterface):
    """Circus-based process orchestration backend"""
    
    def __init__(self):
        self.circus_config_path: Optional[Path] = None
        self.circus_process: Optional[subprocess.Popen] = None
        self.processes: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize Circus backend"""
        if self._initialized:
            return
        
        logger.info("Initializing Circus backend...")
        
        try:
            # Create temporary circus configuration
            self.circus_config_path = await self._create_circus_config()
            
            # Start circusd daemon
            await self._start_circusd()
            
            self._initialized = True
            logger.info("Circus backend initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Circus backend: {e}")
            raise
    
    async def start_process(self, name: str, command: List[str], **kwargs) -> bool:
        """Start a process using Circus"""
        try:
            # Store process metadata
            self.processes[name] = {
                "command": command,
                "port": kwargs.get("port"),
                "health_url": kwargs.get("health_url"),
                "restart_count": 0,
                "status": "starting"
            }
            
            # Add watcher to circus
            circus_cmd = [
                "circusctl",
                "--endpoint", self._get_circus_endpoint(),
                "add",
                name,
                *command
            ]
            
            result = await self._run_command(circus_cmd)
            if result.returncode != 0:
                logger.error(f"Failed to add circus watcher: {result.stderr}")
                return False
            
            # Start the watcher
            start_cmd = [
                "circusctl", 
                "--endpoint", self._get_circus_endpoint(),
                "start",
                name
            ]
            
            result = await self._run_command(start_cmd)
            if result.returncode == 0:
                self.processes[name]["status"] = "running"
                logger.info(f"Process {name} started successfully")
                return True
            else:
                logger.error(f"Failed to start process {name}: {result.stderr}")
                self.processes[name]["status"] = "failed"
                return False
                
        except Exception as e:
            logger.error(f"Error starting process {name}: {e}")
            if name in self.processes:
                self.processes[name]["status"] = "failed"
            return False
    
    async def stop_process(self, name: str) -> bool:
        """Stop a process using Circus"""
        if name not in self.processes:
            logger.warning(f"Process {name} not found")
            return False
        
        try:
            stop_cmd = [
                "circusctl",
                "--endpoint", self._get_circus_endpoint(),
                "stop",
                name
            ]
            
            result = await self._run_command(stop_cmd)
            if result.returncode == 0:
                self.processes[name]["status"] = "stopped"
                logger.info(f"Process {name} stopped successfully")
                return True
            else:
                logger.error(f"Failed to stop process {name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping process {name}: {e}")
            return False
    
    async def restart_process(self, name: str) -> bool:
        """Restart a process using Circus"""
        if name not in self.processes:
            logger.warning(f"Process {name} not found")
            return False
        
        try:
            restart_cmd = [
                "circusctl",
                "--endpoint", self._get_circus_endpoint(), 
                "restart",
                name
            ]
            
            result = await self._run_command(restart_cmd)
            if result.returncode == 0:
                self.processes[name]["restart_count"] += 1
                self.processes[name]["status"] = "running"
                logger.info(f"Process {name} restarted successfully")
                return True
            else:
                logger.error(f"Failed to restart process {name}: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error restarting process {name}: {e}")
            return False
    
    async def get_process_info(self, name: str) -> Optional[ProcessInfo]:
        """Get process information from Circus"""
        if name not in self.processes:
            return None
        
        try:
            # Get process stats from circus
            stats_cmd = [
                "circusctl",
                "--endpoint", self._get_circus_endpoint(),
                "stats",
                name
            ]
            
            result = await self._run_command(stats_cmd)
            
            process_data = self.processes[name]
            
            if result.returncode == 0:
                # Parse circus stats (this is a simplified version)
                status = "running"
                try:
                    stats_output = result.stdout.strip()
                    if "is not active" in stats_output:
                        status = "stopped"
                except:
                    pass
            else:
                status = "stopped"
            
            process_data["status"] = status
            
            return ProcessInfo(
                name=name,
                status=status,
                port=process_data.get("port"),
                health_url=process_data.get("health_url"),
                restart_count=process_data.get("restart_count", 0),
                pid=None,  # Would need to parse from circus stats
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
        """Clean up Circus resources"""
        if not self._initialized:
            return
        
        logger.info("Cleaning up Circus backend...")
        
        try:
            # Stop all watchers
            for name in list(self.processes.keys()):
                await self.stop_process(name)
            
            # Stop circusd if we started it
            if self.circus_process:
                try:
                    self.circus_process.terminate()
                    await asyncio.wait_for(
                        asyncio.create_task(self._wait_for_process(self.circus_process)),
                        timeout=10.0
                    )
                except asyncio.TimeoutError:
                    self.circus_process.kill()
                except:
                    pass
            
            # Clean up config file
            if self.circus_config_path and self.circus_config_path.exists():
                self.circus_config_path.unlink()
            
            self._initialized = False
            logger.info("Circus backend cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during Circus cleanup: {e}")
    
    async def _create_circus_config(self) -> Path:
        """Create Circus configuration file"""
        config = {
            "check_delay": 5,
            "endpoint": "tcp://127.0.0.1:5555",
            "pubsub_endpoint": "tcp://127.0.0.1:5556",
            "stats_endpoint": "tcp://127.0.0.1:5557",
            "watchers": []
        }
        
        # Create temporary config file
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.ini',
            prefix='circus_aas_tts_',
            delete=False
        )
        
        config_path = Path(temp_file.name)
        
        # Write circus INI format
        with open(config_path, 'w') as f:
            f.write("[circus]\n")
            f.write(f"check_delay = {config['check_delay']}\n")
            f.write(f"endpoint = {config['endpoint']}\n") 
            f.write(f"pubsub_endpoint = {config['pubsub_endpoint']}\n")
            f.write(f"stats_endpoint = {config['stats_endpoint']}\n")
            f.write("\n")
        
        logger.debug(f"Created Circus config at: {config_path}")
        return config_path
    
    async def _start_circusd(self) -> None:
        """Start the circusd daemon"""
        if not self.circus_config_path:
            raise RuntimeError("Circus config not created")
        
        logger.debug("Starting circusd daemon...")
        
        try:
            self.circus_process = subprocess.Popen(
                ["circusd", str(self.circus_config_path), "--daemon"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait a moment for circus to start
            await asyncio.sleep(2)
            
            # Check if circus is running
            if self.circus_process.poll() is not None:
                stdout, stderr = self.circus_process.communicate()
                raise RuntimeError(f"circusd failed to start: {stderr.decode()}")
            
            logger.debug("circusd started successfully")
            
        except FileNotFoundError:
            raise RuntimeError(
                "circusd not found. Install with: pip install circus"
            )
    
    def _get_circus_endpoint(self) -> str:
        """Get Circus control endpoint"""
        return "tcp://127.0.0.1:5555"
    
    async def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """Run a command asynchronously"""
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=command,
            returncode=process.returncode,
            stdout=stdout.decode(),
            stderr=stderr.decode()
        )
    
    async def _wait_for_process(self, process: subprocess.Popen) -> None:
        """Wait for a process to finish"""
        while process.poll() is None:
            await asyncio.sleep(0.1)