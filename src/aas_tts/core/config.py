"""
Applied AI Systems - AAS-TTS Configuration
Pydantic Settings for environment-based configuration
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import ServerConfig, AudioConfig, MCPConfig


class AASSettings(BaseSettings):
    """Main AAS-TTS configuration using Pydantic Settings"""
    
    model_config = SettingsConfigDict(
        env_prefix="AAS_TTS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application Info
    app_name: str = Field("AAS-TTS", description="Application name")
    app_version: str = Field("1.0.0", description="Application version")
    debug: bool = Field(False, description="Enable debug mode")
    
    # Server Configuration
    server: ServerConfig = Field(default_factory=ServerConfig)
    
    # Audio Configuration
    audio: AudioConfig = Field(default_factory=AudioConfig)
    
    # MCP Configuration
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    
    # Model Paths
    models_dir: Path = Field(Path("models"), description="Models directory")
    voices_dir: Path = Field(Path("voices"), description="Voices directory")
    cache_dir: Path = Field(Path(".cache"), description="Cache directory")
    
    # Device Configuration
    device: str = Field("auto", description="Compute device (cpu, cuda, auto)")
    use_gpu: bool = Field(True, description="Use GPU if available")
    gpu_memory_fraction: float = Field(0.8, ge=0.1, le=1.0, description="GPU memory fraction")
    
    # Orchestration Configuration
    orchestration_backend: str = Field("simple", description="Process orchestration backend (simple, circus, docker)")
    api_port: int = Field(8880, description="Default API server port")
    
    # Audio Processing
    default_voice: str = Field("af_bella", description="Default voice")
    max_text_length: int = Field(10000, description="Maximum text length")
    audio_temp_dir: Path = Field(Path("/tmp/aas-tts"), description="Temporary audio directory")
    cleanup_temp_files: bool = Field(True, description="Clean up temporary files")
    
    # Database Configuration
    database_url: Optional[str] = Field(None, description="Database connection URL")
    database_echo: bool = Field(False, description="Echo SQL queries")
    
    # API Keys (Optional integrations)
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    
    # Security
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    cors_methods: List[str] = Field(["*"], description="CORS allowed methods")
    cors_headers: List[str] = Field(["*"], description="CORS allowed headers")
    
    # Performance
    worker_count: int = Field(1, ge=1, le=32, description="Number of worker processes")
    max_concurrent_requests: int = Field(100, description="Maximum concurrent requests")
    request_timeout: float = Field(300.0, description="Request timeout seconds")
    
    # Logging
    log_level: str = Field("INFO", description="Logging level")
    log_format: str = Field("json", description="Log format (json, text)")
    log_file: Optional[Path] = Field(None, description="Log file path")
    
    # Feature Flags
    enable_openai_compat: bool = Field(True, description="Enable OpenAI compatibility layer")
    enable_streaming: bool = Field(True, description="Enable streaming responses")
    enable_metrics: bool = Field(True, description="Enable metrics collection")
    enable_health_checks: bool = Field(True, description="Enable health check endpoints")
    
    @validator('device')
    def validate_device(cls, v):
        valid_devices = ["cpu", "cuda", "mps", "auto"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of: {valid_devices}")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v_upper
    
    @validator('models_dir', 'voices_dir', 'cache_dir', 'audio_temp_dir')
    def create_directories(cls, v):
        """Ensure directories exist"""
        if isinstance(v, str):
            v = Path(v)
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_device(self) -> str:
        """Get the actual device to use"""
        if self.device == "auto":
            try:
                import torch
                if torch.cuda.is_available() and self.use_gpu:
                    return "cuda"
                elif torch.backends.mps.is_available() and self.use_gpu:
                    return "mps"
                else:
                    return "cpu"
            except ImportError:
                return "cpu"
        return self.device
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to a specific model"""
        return self.models_dir / model_name
    
    def get_voice_path(self, voice_id: str) -> Path:
        """Get path to a specific voice"""
        return self.voices_dir / f"{voice_id}.pt"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return self.model_dump()


# Global settings instance
_settings: Optional["AASSettings"] = None


def get_settings() -> AASSettings:
    """Get global settings instance (singleton)"""
    global _settings
    if _settings is None:
        _settings = AASSettings()
    return _settings


def reload_settings() -> AASSettings:
    """Reload settings from environment"""
    global _settings
    _settings = AASSettings()
    return _settings


# CLI-specific configuration
class CLIConfig(BaseSettings):
    """CLI-specific configuration"""
    
    model_config = SettingsConfigDict(
        env_prefix="AAS_TTS_CLI_",
        case_sensitive=False
    )
    
    # Output configuration
    output_dir: Path = Field(Path("output"), description="Default output directory")
    output_format: str = Field("wav", description="Default output format")
    overwrite: bool = Field(False, description="Overwrite existing files")
    
    # CLI behavior
    verbose: bool = Field(False, description="Verbose output")
    quiet: bool = Field(False, description="Quiet mode")
    progress: bool = Field(True, description="Show progress bars")
    
    # Batch processing
    batch_size: int = Field(1, ge=1, le=100, description="Batch processing size")
    parallel: bool = Field(True, description="Enable parallel processing")
    max_workers: int = Field(4, ge=1, le=32, description="Maximum worker threads")