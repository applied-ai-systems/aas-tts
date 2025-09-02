"""
Applied AI Systems - Process Orchestration Package
Circus and Docker-on-Whales abstraction for process management
"""
from .process_manager import ProcessManager, get_process_manager
from .simple_backend import SimpleBackend
from .circus_backend import CircusBackend
from .docker_backend import DockerBackend

__all__ = [
    "ProcessManager",
    "get_process_manager",
    "SimpleBackend",
    "CircusBackend", 
    "DockerBackend",
]