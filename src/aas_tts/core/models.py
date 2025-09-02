"""
Applied AI Systems - AAS-TTS Core Models
Pydantic models for data validation and serialization
"""
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, ConfigDict
from sqlmodel import SQLModel, Field as SQLField, Column, DateTime
from sqlalchemy import func


class VoiceCategory(str, Enum):
    """Voice category enumeration"""
    AMERICAN_FEMALE = "af"
    AMERICAN_MALE = "am" 
    BRITISH_FEMALE = "bf"
    BRITISH_MALE = "bm"
    EUROPEAN_FEMALE = "ef"
    EUROPEAN_MALE = "em"
    FEMALE_FOREIGN = "ff"
    HINDI_FEMALE = "hf"
    HINDI_MALE = "hm"
    ITALIAN_FEMALE = "if"
    ITALIAN_MALE = "im"
    JAPANESE_FEMALE = "jf"
    JAPANESE_MALE = "jm"
    POLISH_FEMALE = "pf"
    POLISH_MALE = "pm"
    CHINESE_FEMALE = "zf"
    CHINESE_MALE = "zm"


class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"


class TTSEngine(str, Enum):
    """TTS engine options"""
    KOKORO = "kokoro"
    OPENAI = "openai"  # For compatibility


# Pydantic Models (API/CLI)
class VoiceInfo(BaseModel):
    """Voice information model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    id: str = Field(..., description="Voice identifier (e.g., af_bella)")
    name: str = Field(..., description="Human-readable voice name") 
    category: VoiceCategory = Field(..., description="Voice category")
    language: str = Field(..., description="Language code (e.g., en-US)")
    gender: str = Field(..., description="Voice gender")
    description: Optional[str] = Field(None, description="Voice description")
    sample_rate: int = Field(22050, description="Audio sample rate")
    available: bool = Field(True, description="Whether voice is available")

    @validator('id')
    def validate_voice_id(cls, v):
        if not v or len(v) < 2:
            raise ValueError("Voice ID must be at least 2 characters")
        return v


class TTSRequest(BaseModel):
    """TTS synthesis request model"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    # Core parameters
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice: str = Field("af_bella", description="Voice to use for synthesis")
    
    # Audio parameters
    speed: float = Field(1.0, ge=0.1, le=3.0, description="Speech speed multiplier")
    format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")
    sample_rate: Optional[int] = Field(None, ge=8000, le=48000, description="Output sample rate")
    
    # OpenAI compatibility
    model: Optional[str] = Field(None, description="Model name (for OpenAI compatibility)")
    response_format: Optional[str] = Field(None, description="Response format")
    
    # Advanced parameters
    engine: TTSEngine = Field(TTSEngine.KOKORO, description="TTS engine to use")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Generation temperature")
    normalize: bool = Field(True, description="Normalize output audio")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class TTSResponse(BaseModel):
    """TTS synthesis response model"""
    model_config = ConfigDict()
    
    # Response metadata
    request_id: UUID = Field(default_factory=uuid4, description="Unique request identifier")
    voice: str = Field(..., description="Voice used for synthesis")
    format: AudioFormat = Field(..., description="Audio format")
    
    # Audio data
    audio_data: Optional[bytes] = Field(None, description="Generated audio data")
    audio_url: Optional[str] = Field(None, description="URL to audio file")
    audio_path: Optional[Path] = Field(None, description="Local path to audio file")
    
    # Processing metadata
    text_length: int = Field(..., description="Length of input text")
    audio_duration: Optional[float] = Field(None, description="Audio duration in seconds")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    sample_rate: int = Field(22050, description="Audio sample rate")
    
    # Error handling
    success: bool = Field(True, description="Whether synthesis was successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        arbitrary_types_allowed = True


# SQLModel Models (Database)
class TTSJob(SQLModel, table=True):
    """Database model for TTS job tracking"""
    __tablename__ = "tts_jobs"
    
    # Primary key
    id: Optional[int] = SQLField(default=None, primary_key=True)
    request_id: UUID = SQLField(unique=True, index=True)
    
    # Request data
    text: str = SQLField(max_length=10000)
    voice: str = SQLField(max_length=50, index=True)
    format: str = SQLField(max_length=10)
    engine: str = SQLField(max_length=20, default="kokoro")
    
    # Parameters (JSON)
    parameters: Optional[Dict[str, Any]] = SQLField(default=None, sa_column=Column(Dict))
    
    # Results
    success: bool = SQLField(default=False)
    error_message: Optional[str] = SQLField(default=None)
    audio_path: Optional[str] = SQLField(default=None)
    audio_duration: Optional[float] = SQLField(default=None)
    processing_time: Optional[float] = SQLField(default=None)
    
    # Timestamps
    created_at: Optional[datetime] = SQLField(
        default=None, 
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    completed_at: Optional[datetime] = SQLField(
        default=None,
        sa_column=Column(DateTime(timezone=True))
    )
    
    # User tracking
    user_id: Optional[str] = SQLField(default=None, index=True)
    client_ip: Optional[str] = SQLField(default=None)


class VoiceModel(SQLModel, table=True):
    """Database model for available voices"""
    __tablename__ = "voices"
    
    # Primary key
    id: str = SQLField(primary_key=True, max_length=50)
    
    # Voice metadata
    name: str = SQLField(max_length=100)
    category: str = SQLField(max_length=10, index=True)
    language: str = SQLField(max_length=10, index=True)
    gender: str = SQLField(max_length=10)
    description: Optional[str] = SQLField(default=None)
    
    # Technical specs
    sample_rate: int = SQLField(default=22050)
    model_path: Optional[str] = SQLField(default=None)
    config_path: Optional[str] = SQLField(default=None)
    
    # Status
    available: bool = SQLField(default=True, index=True)
    version: str = SQLField(default="1.0")
    
    # Timestamps
    created_at: Optional[datetime] = SQLField(
        default=None,
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: Optional[datetime] = SQLField(
        default=None,
        sa_column=Column(DateTime(timezone=True), onupdate=func.now())
    )


# Configuration Models
class ServerConfig(BaseModel):
    """Server configuration model"""
    model_config = ConfigDict()
    
    host: str = Field("0.0.0.0", description="Server host")
    port: int = Field(8000, ge=1, le=65535, description="Server port")
    workers: int = Field(1, ge=1, le=32, description="Number of worker processes")
    reload: bool = Field(False, description="Enable auto-reload")
    log_level: str = Field("info", description="Log level")


class AudioConfig(BaseModel):
    """Audio processing configuration"""
    model_config = ConfigDict()
    
    default_sample_rate: int = Field(22050, description="Default sample rate")
    max_duration: float = Field(300.0, description="Maximum audio duration (seconds)")
    normalize_audio: bool = Field(True, description="Normalize audio output")
    temp_dir: Path = Field(Path("/tmp/aas-tts"), description="Temporary directory")
    
    class Config:
        arbitrary_types_allowed = True


class MCPConfig(BaseModel):
    """MCP server configuration"""
    model_config = ConfigDict()
    
    enabled: bool = Field(True, description="Enable MCP server")
    server_name: str = Field("aas-tts", description="MCP server name")
    description: str = Field("AAS-TTS MCP Server", description="Server description")
    version: str = Field("1.0.0", description="Server version")
    tools_enabled: bool = Field(True, description="Enable MCP tools")
    resources_enabled: bool = Field(True, description="Enable MCP resources")