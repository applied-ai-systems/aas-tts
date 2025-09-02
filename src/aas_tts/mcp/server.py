"""
Applied AI Systems - AAS-TTS MCP Server
FastMCP server implementation with TTS tools and API lifecycle management
"""
import asyncio
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from mcp.server import Server
from mcp.types import Tool, Resource, ResourceTemplate
from mcp import ServerSession
from loguru import logger

from ..core.config import get_settings
from ..core.models import TTSRequest, AudioFormat, VoiceCategory
from ..services.tts_service import get_tts_service
from ..orchestration.process_manager import get_process_manager, ProcessBackend


class MCPServer:
    """AAS-TTS MCP Server with process lifecycle management"""
    
    def __init__(self):
        self.settings = get_settings()
        self.server = Server("aas-tts")
        self.process_manager = None
        self.tts_service = None
        self._api_port = 8880
        self._setup_handlers()
    
    async def initialize(self) -> None:
        """Initialize MCP server"""
        logger.info("Initializing AAS-TTS MCP server...")
        
        try:
            # Initialize process manager
            backend = ProcessBackend.SIMPLE  # Default to Simple
            if self.settings.orchestration_backend == "circus":
                backend = ProcessBackend.CIRCUS
            elif self.settings.orchestration_backend == "docker":
                backend = ProcessBackend.DOCKER
                
            self.process_manager = await get_process_manager(backend)
            
            # Initialize TTS service
            self.tts_service = await get_tts_service()
            
            # Ensure TTS API is running
            await self.process_manager.ensure_tts_api_running(port=self._api_port)
            
            logger.info("AAS-TTS MCP server initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    def _setup_handlers(self) -> None:
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available TTS tools"""
            return [
                Tool(
                    name="synthesize_speech",
                    description="Synthesize text to speech using AAS-TTS",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to synthesize"
                            },
                            "voice": {
                                "type": "string", 
                                "description": "Voice ID to use",
                                "default": "af_bella",
                                "enum": [
                                    "af_bella", "am_adam", "bf_emma", "bm_george",
                                    "jf_yuki", "hf_alpha", "hm_omega", "zf_xiaoni",
                                    "zm_yunjian"
                                ]
                            },
                            "format": {
                                "type": "string",
                                "description": "Audio format",
                                "default": "wav",
                                "enum": ["wav", "mp3", "flac", "ogg"]
                            },
                            "speed": {
                                "type": "number",
                                "description": "Speech speed multiplier",
                                "default": 1.0,
                                "minimum": 0.1,
                                "maximum": 3.0
                            },
                            "output_path": {
                                "type": "string",
                                "description": "Output file path (optional)",
                                "default": None
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="list_voices", 
                    description="List available TTS voices",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter by voice category (optional)",
                                "enum": [
                                    "af", "am", "bf", "bm", "ef", "em", "ff",
                                    "hf", "hm", "if", "im", "jf", "jm",
                                    "pf", "pm", "zf", "zm"
                                ]
                            },
                            "language": {
                                "type": "string", 
                                "description": "Filter by language (optional)",
                                "enum": [
                                    "en-US", "en-GB", "ja-JP", "zh-CN",
                                    "hi-IN", "it-IT", "pl-PL"
                                ]
                            }
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_voice_info",
                    description="Get detailed information about a specific voice",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "voice_id": {
                                "type": "string",
                                "description": "Voice ID to get information about"
                            }
                        },
                        "required": ["voice_id"]
                    }
                ),
                Tool(
                    name="check_api_health",
                    description="Check TTS API server health and status",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="manage_api_server",
                    description="Start, stop, or restart the TTS API server",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": "Action to perform",
                                "enum": ["start", "stop", "restart", "status"]
                            },
                            "port": {
                                "type": "integer",
                                "description": "Port for API server (default: 8880)",
                                "default": 8880
                            }
                        },
                        "required": ["action"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
            """Handle tool calls"""
            try:
                if name == "synthesize_speech":
                    return await self._handle_synthesize_speech(arguments)
                elif name == "list_voices":
                    return await self._handle_list_voices(arguments)
                elif name == "get_voice_info":
                    return await self._handle_get_voice_info(arguments)
                elif name == "check_api_health":
                    return await self._handle_check_api_health(arguments)
                elif name == "manage_api_server":
                    return await self._handle_manage_api_server(arguments)
                else:
                    return [{
                        "type": "text",
                        "text": f"Unknown tool: {name}"
                    }]
            except Exception as e:
                logger.error(f"Error handling tool {name}: {e}")
                return [{
                    "type": "text", 
                    "text": f"Error: {e}"
                }]
    
    async def _handle_synthesize_speech(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle text-to-speech synthesis"""
        text = args["text"]
        voice = args.get("voice", "af_bella")
        format_str = args.get("format", "wav")
        speed = args.get("speed", 1.0)
        output_path = args.get("output_path")
        
        try:
            # Create TTS request
            request = TTSRequest(
                text=text,
                voice=voice,
                format=AudioFormat(format_str),
                speed=speed
            )
            
            # Synthesize
            if output_path:
                response = await self.tts_service.synthesize_to_file(
                    request=request,
                    output_path=Path(output_path),
                    overwrite=True
                )
                
                return [{
                    "type": "text",
                    "text": f"✅ Speech synthesized successfully!\n"
                           f"📁 Output: {response.audio_path}\n"
                           f"⏱️ Processing time: {response.processing_time:.2f}s\n"
                           f"🎵 Duration: {response.audio_duration:.2f}s\n"
                           f"🎤 Voice: {response.voice}"
                }]
            else:
                response = await self.tts_service.synthesize(request)
                
                return [{
                    "type": "text",
                    "text": f"✅ Speech synthesized successfully!\n"
                           f"📊 Audio size: {len(response.audio_data)} bytes\n" 
                           f"⏱️ Processing time: {response.processing_time:.2f}s\n"
                           f"🎵 Duration: {response.audio_duration:.2f}s\n"
                           f"🎤 Voice: {response.voice}\n"
                           f"💡 Use output_path parameter to save to file"
                }]
                
        except Exception as e:
            return [{
                "type": "text",
                "text": f"❌ Synthesis failed: {e}"
            }]
    
    async def _handle_list_voices(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle voice listing"""
        category = args.get("category")
        language = args.get("language")
        
        try:
            voices = await self.tts_service.get_available_voices()
            
            # Apply filters
            if category:
                voices = [v for v in voices if v.category.value == category]
            if language:
                voices = [v for v in voices if v.language == language]
            
            if not voices:
                return [{
                    "type": "text",
                    "text": "🔍 No voices found matching criteria"
                }]
            
            # Format voice list
            voice_text = f"🎵 Available Voices ({len(voices)}):\n\n"
            
            for voice in sorted(voices, key=lambda v: v.id):
                status = "✅" if voice.available else "❌"
                voice_text += f"{status} **{voice.id}** - {voice.name}\n"
                voice_text += f"   📂 {voice.category.value} | 🌍 {voice.language} | 👤 {voice.gender}\n"
                if voice.description:
                    voice_text += f"   📄 {voice.description}\n"
                voice_text += "\n"
            
            return [{
                "type": "text",
                "text": voice_text
            }]
            
        except Exception as e:
            return [{
                "type": "text",
                "text": f"❌ Failed to list voices: {e}"
            }]
    
    async def _handle_get_voice_info(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle voice info request"""
        voice_id = args["voice_id"]
        
        try:
            voice = await self.tts_service.get_voice_info(voice_id)
            
            if not voice:
                return [{
                    "type": "text",
                    "text": f"❌ Voice '{voice_id}' not found"
                }]
            
            info_text = f"🎤 **Voice Information: {voice.id}**\n\n"
            info_text += f"📝 Name: {voice.name}\n"
            info_text += f"📂 Category: {voice.category.value}\n"
            info_text += f"🌍 Language: {voice.language}\n"
            info_text += f"👤 Gender: {voice.gender}\n"
            info_text += f"📊 Sample Rate: {voice.sample_rate} Hz\n"
            info_text += f"✅ Available: {'Yes' if voice.available else 'No'}\n"
            
            if voice.description:
                info_text += f"📄 Description: {voice.description}\n"
            
            return [{
                "type": "text",
                "text": info_text
            }]
            
        except Exception as e:
            return [{
                "type": "text",
                "text": f"❌ Failed to get voice info: {e}"
            }]
    
    async def _handle_check_api_health(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle API health check"""
        try:
            # Check if API is healthy
            is_healthy = await self.process_manager.is_tts_api_healthy()
            
            # Get detailed status
            status_info = await self.process_manager.get_tts_api_status()
            
            if is_healthy and status_info:
                health_text = f"✅ **TTS API is healthy**\n\n"
                health_text += f"📊 Status: {status_info.status}\n"
                health_text += f"🔌 Port: {status_info.port}\n"
                health_text += f"🔄 Restarts: {status_info.restart_count}\n"
                
                if status_info.health_url:
                    health_text += f"🏥 Health URL: {status_info.health_url}\n"
                
                # Test synthesis
                health_info = await self.tts_service.health_check()
                health_text += f"🎵 Voices: {health_info['voices_available']}\n"
                health_text += f"⏱️ Test synthesis: {health_info['test_synthesis_time']}s\n"
                health_text += f"💻 Device: {health_info['device']}\n"
            else:
                health_text = f"❌ **TTS API is unhealthy**\n\n"
                if status_info:
                    health_text += f"📊 Status: {status_info.status}\n"
                else:
                    health_text += "❓ API server not found or not responding\n"
                
                health_text += "\n💡 Try running: manage_api_server with action 'start'"
            
            return [{
                "type": "text", 
                "text": health_text
            }]
            
        except Exception as e:
            return [{
                "type": "text",
                "text": f"❌ Health check failed: {e}"
            }]
    
    async def _handle_manage_api_server(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle API server management"""
        action = args["action"]
        port = args.get("port", 8880)
        
        try:
            if action == "start":
                success = await self.process_manager.start_tts_api(port=port)
                if success:
                    return [{
                        "type": "text",
                        "text": f"✅ TTS API started successfully on port {port}"
                    }]
                else:
                    return [{
                        "type": "text",
                        "text": f"❌ Failed to start TTS API on port {port}"
                    }]
            
            elif action == "stop":
                success = await self.process_manager.stop_tts_api()
                if success:
                    return [{
                        "type": "text",
                        "text": "✅ TTS API stopped successfully"
                    }]
                else:
                    return [{
                        "type": "text", 
                        "text": "❌ Failed to stop TTS API"
                    }]
            
            elif action == "restart":
                success = await self.process_manager.restart_tts_api()
                if success:
                    return [{
                        "type": "text",
                        "text": "✅ TTS API restarted successfully"
                    }]
                else:
                    return [{
                        "type": "text",
                        "text": "❌ Failed to restart TTS API"
                    }]
            
            elif action == "status":
                status_info = await self.process_manager.get_tts_api_status()
                if status_info:
                    status_text = f"📊 **TTS API Status**\n\n"
                    status_text += f"Status: {status_info.status}\n"
                    status_text += f"Port: {status_info.port}\n"
                    status_text += f"Restarts: {status_info.restart_count}\n"
                    
                    return [{
                        "type": "text",
                        "text": status_text
                    }]
                else:
                    return [{
                        "type": "text",
                        "text": "❓ TTS API status unknown - server may not be running"
                    }]
            
            else:
                return [{
                    "type": "text",
                    "text": f"❌ Unknown action: {action}. Use: start, stop, restart, or status"
                }]
                
        except Exception as e:
            return [{
                "type": "text",
                "text": f"❌ Server management failed: {e}"
            }]
    
    async def run_stdio(self) -> None:
        """Run MCP server in stdio mode"""
        from mcp.server.stdio import stdio_server
        
        logger.info("Starting AAS-TTS MCP server in stdio mode...")
        
        try:
            # Initialize server
            await self.initialize()
            
            # Run stdio server
            async with stdio_server(self.server) as server:
                await server.run()
                
        except Exception as e:
            logger.error(f"MCP server error: {e}")
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup MCP server resources"""
        logger.info("Cleaning up AAS-TTS MCP server...")
        
        try:
            if self.process_manager:
                await self.process_manager.shutdown()
                
        except Exception as e:
            logger.error(f"Error during MCP server cleanup: {e}")


async def create_mcp_server() -> MCPServer:
    """Create and initialize MCP server"""
    server = MCPServer()
    await server.initialize()
    return server