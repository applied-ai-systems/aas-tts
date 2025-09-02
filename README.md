# AAS-TTS: Applied AI Systems Text-to-Speech

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Typer](https://img.shields.io/badge/CLI-Typer-009639.svg)](https://typer.tiangolo.com/)

A unified, production-ready Text-to-Speech system combining Kokoro TTS with FastAPI, Typer CLI, and FastMCP server capabilities. Built with Applied AI Systems' layered architecture using Pydantic, FastAPI, FastMCP, and SQLModel.

## 🎙️ Features

- **High-Quality TTS**: Kokoro TTS models with 80+ voices across multiple languages
- **Multiple Interfaces**: CLI, REST API, and MCP server
- **Production Ready**: Comprehensive error handling, logging, and monitoring
- **Layered Architecture**: Clean separation between CLI → Service → Data layers
- **Type Safe**: Full Pydantic validation and SQLModel integration
- **Async First**: Built with async/await throughout
- **Docker Ready**: Multi-platform containers with GPU support

## 🚀 Quick Start

### Installation

```bash
# Install with uv (recommended)
uv add aas-tts

# Or with pip
pip install aas-tts

# For GPU support
uv add "aas-tts[gpu]"

# For development
uv add "aas-tts[dev]"
```

### CLI Usage

```bash
# Synthesize text to speech
aas-tts synthesize "Hello, world!" --voice af_bella --output hello.wav

# List available voices
aas-tts list-voices

# Get voice information
aas-tts info af_bella

# Check service health
aas-tts health

# Start FastAPI server
aas-tts server start --host 0.0.0.0 --port 8000

# Start MCP server
aas-tts mcp start
```

### Python API

```python
import asyncio
from aas_tts import get_tts_service, TTSRequest, AudioFormat

async def main():
    service = await get_tts_service()
    
    request = TTSRequest(
        text="Hello from AAS-TTS!",
        voice="af_bella",
        format=AudioFormat.WAV,
        speed=1.0
    )
    
    response = await service.synthesize(request)
    print(f"Generated {response.audio_duration:.2f}s of audio")

asyncio.run(main())
```

### FastAPI Server

```bash
# Start development server
aas-tts server start --reload

# Production server with multiple workers
aas-tts server start --workers 4 --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🏗️ Architecture

AAS-TTS follows Applied AI Systems' layered architecture:

```
CLI Layer (Typer)
    ↓
Service Layer (Business Logic)
    ↓
Data Layer (SQLModel + Pydantic)
```

### Core Components

- **CLI**: Typer-based command line interface
- **Services**: Audio processing, voice management, TTS orchestration
- **Models**: Pydantic models for validation, SQLModel for persistence
- **Config**: Environment-based configuration with Pydantic Settings
- **API**: FastAPI web server with OpenAI-compatible endpoints
- **MCP**: Model Context Protocol server for AI integrations

## 📦 Available Commands

### Main Commands

- `synthesize` - Convert text to speech
- `list-voices` - Show available voices
- `info` - Get voice details
- `health` - Check service status
- `version` - Show version info

### Server Commands

- `server start` - Start FastAPI server
- `server stop` - Stop running server
- `server status` - Check server status

### Voice Commands

- `voices list` - List all voices (alias for list-voices)
- `voices scan` - Scan for new voice models
- `voices download` - Download voice models

### Batch Commands

- `batch process` - Process multiple files
- `batch list` - List batch jobs
- `batch status` - Check batch status

### MCP Commands

- `mcp start` - Start MCP server
- `mcp stop` - Stop MCP server
- `mcp status` - Check MCP status

## 🎵 Voice Categories

AAS-TTS includes 80+ voices across multiple categories:

- **American**: Female (af_*), Male (am_*)
- **British**: Female (bf_*), Male (bm_*)
- **European**: Female (ef_*), Male (em_*)
- **Japanese**: Female (jf_*), Male (jm_*)
- **Chinese**: Female (zf_*), Male (zm_*)
- **Hindi**: Female (hf_*), Male (hm_*)
- **Italian**: Female (if_*), Male (im_*)
- **Polish**: Female (pf_*), Male (pm_*)

## ⚙️ Configuration

AAS-TTS uses environment variables with the `AAS_TTS_` prefix:

```bash
# Basic configuration
export AAS_TTS_DEFAULT_VOICE=af_bella
export AAS_TTS_DEBUG=true
export AAS_TTS_LOG_LEVEL=INFO

# Server configuration
export AAS_TTS_SERVER_HOST=0.0.0.0
export AAS_TTS_SERVER_PORT=8000
export AAS_TTS_SERVER_WORKERS=1

# Database configuration
export AAS_TTS_DATABASE_URL=sqlite:///aas_tts.db

# Device configuration
export AAS_TTS_DEVICE=auto  # auto, cpu, cuda, mps
export AAS_TTS_USE_GPU=true
```

## 🧪 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/applied-ai-systems/aas-tts.git
cd aas-tts

# Install with development dependencies
uv sync --all-extras

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run with coverage
pytest --cov

# Format code
black src tests
isort src tests

# Type checking
mypy src
```

### Testing

```bash
# Run all tests
pytest

# Run unit tests only
pytest -m unit

# Run integration tests
pytest -m integration

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_tts_service.py
```

## 🐳 Docker

### CPU Version

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install AAS-TTS
COPY . /app
WORKDIR /app
RUN pip install .

# Run server
EXPOSE 8000
CMD ["aas-tts", "server", "start", "--host", "0.0.0.0"]
```

### GPU Version

```dockerfile
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-pip \
    && rm -rf /var/lib/apt/lists/*

# Install AAS-TTS with GPU support
COPY . /app
WORKDIR /app
RUN pip install ".[gpu]"

EXPOSE 8000
CMD ["aas-tts", "server", "start", "--host", "0.0.0.0"]
```

## 🔌 MCP Integration

AAS-TTS includes a FastMCP server for AI agent integrations:

```bash
# Start MCP server
aas-tts mcp start

# Connect from Claude Desktop
{
  "mcpServers": {
    "aas-tts": {
      "command": "aas-tts",
      "args": ["mcp", "start"]
    }
  }
}
```

## 📊 Monitoring

### Health Checks

```bash
# CLI health check
aas-tts health

# HTTP health check
curl http://localhost:8000/health
```

### Metrics

AAS-TTS exposes Prometheus-compatible metrics:

- Request counts and latencies
- Voice usage statistics
- Audio generation metrics
- Error rates and types

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Kokoro TTS](https://github.com/hexgrad/kokoro) - High-quality neural TTS
- [Typer](https://typer.tiangolo.com/) - Modern CLI framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [SQLModel](https://sqlmodel.tiangolo.com/) - SQL databases with Python

---

**Applied AI Systems** - Building the future of AI applications