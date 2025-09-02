# AAS-TTS Examples

This directory contains example configurations and usage examples for AAS-TTS.

## MCP Client Configuration

### Claude Desktop Configuration

To use AAS-TTS with Claude Desktop, add this to your MCP configuration file:

**Location:** `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "aas-tts": {
      "command": "uvx",
      "args": ["aas-tts", "mcp", "serve", "--stdio"]
    }
  }
}
```

### Advanced Configuration

For more control over the TTS system:

```json
{
  "mcpServers": {
    "aas-tts": {
      "command": "uvx",
      "args": ["aas-tts", "mcp", "serve", "--stdio", "--backend", "circus"],
      "env": {
        "AAS_TTS_LOG_LEVEL": "INFO",
        "AAS_TTS_ORCHESTRATION_BACKEND": "circus",
        "AAS_TTS_API_PORT": "8880",
        "AAS_TTS_DEFAULT_VOICE": "af_bella",
        "AAS_TTS_USE_GPU": "true"
      }
    }
  }
}
```

### Docker Backend Configuration

To use Docker for process orchestration:

```json
{
  "mcpServers": {
    "aas-tts": {
      "command": "uvx",
      "args": ["aas-tts", "mcp", "serve", "--stdio", "--backend", "docker"],
      "env": {
        "AAS_TTS_ORCHESTRATION_BACKEND": "docker"
      }
    }
  }
}
```

## Available MCP Tools

Once configured, the following tools will be available in Claude:

### 🎙️ `synthesize_speech`
Convert text to speech with customizable options.

**Parameters:**
- `text` (required): Text to synthesize
- `voice`: Voice ID (default: af_bella)
- `format`: Audio format (wav, mp3, flac, ogg)
- `speed`: Speech speed multiplier (0.1-3.0)
- `output_path`: File path to save audio (optional)

**Example:**
```
Please synthesize "Hello from Applied AI Systems!" using the af_bella voice and save it to hello.wav
```

### 🎵 `list_voices`
List all available TTS voices with filtering options.

**Parameters:**
- `category`: Filter by voice category (af, am, bf, bm, etc.)
- `language`: Filter by language (en-US, en-GB, ja-JP, etc.)

**Example:**
```
Show me all American female voices
```

### ℹ️ `get_voice_info`
Get detailed information about a specific voice.

**Parameters:**
- `voice_id` (required): Voice identifier

**Example:**
```
Tell me about the af_bella voice
```

### 🏥 `check_api_health`
Check the health status of the TTS API server.

**Example:**
```
Check if the TTS API is healthy
```

### ⚙️ `manage_api_server`
Start, stop, restart, or check the status of the TTS API server.

**Parameters:**
- `action` (required): start, stop, restart, or status
- `port`: API server port (default: 8880)

**Example:**
```
Restart the TTS API server
```

## Installation Requirements

Before using the MCP server, ensure you have:

1. **Python 3.11+** installed
2. **uvx** (from uv) for running the command:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **AAS-TTS** package:
   ```bash
   uvx install aas-tts
   ```

## Process Orchestration

AAS-TTS uses process orchestration to manage the TTS API server automatically:

### Circus (Default)
- Lightweight Python process manager
- Automatic restart on failure
- Built-in health monitoring
- No additional dependencies

### Docker
- Container-based isolation
- Resource management
- Cross-platform consistency
- Requires Docker to be installed

## Environment Variables

Configure AAS-TTS behavior with environment variables:

- `AAS_TTS_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `AAS_TTS_ORCHESTRATION_BACKEND`: Process backend (circus, docker)
- `AAS_TTS_API_PORT`: API server port (default: 8880)
- `AAS_TTS_DEFAULT_VOICE`: Default voice for synthesis
- `AAS_TTS_USE_GPU`: Enable GPU acceleration (true/false)
- `AAS_TTS_DEVICE`: Compute device (cpu, cuda, mps, auto)

## Troubleshooting

### MCP Server Not Starting
1. Check that uvx and aas-tts are installed:
   ```bash
   uvx --help
   uvx aas-tts --version
   ```

2. Test the MCP server directly:
   ```bash
   uvx aas-tts mcp serve --stdio
   ```

3. Check logs for error messages

### API Server Issues
1. Check API health:
   ```bash
   uvx aas-tts health
   ```

2. Manually start the API:
   ```bash
   uvx aas-tts server start --port 8880
   ```

3. Test the endpoint:
   ```bash
   curl http://localhost:8880/health
   ```

### Voice Not Found
1. List available voices:
   ```bash
   uvx aas-tts list-voices
   ```

2. Check voice info:
   ```bash
   uvx aas-tts info af_bella
   ```

## Support

For issues and questions:
- GitHub Issues: https://github.com/applied-ai-systems/aas-tts/issues
- Documentation: https://github.com/applied-ai-systems/aas-tts/docs