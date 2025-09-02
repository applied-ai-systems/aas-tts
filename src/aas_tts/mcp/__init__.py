"""
Applied AI Systems - AAS-TTS MCP Server
FastMCP server with TTS tools and API lifecycle management
"""
from .server import create_mcp_server, MCPServer

__all__ = [
    "create_mcp_server",
    "MCPServer",
]