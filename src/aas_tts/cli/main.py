"""
Applied AI Systems - AAS-TTS CLI Main Interface
Typer-based command line interface for TTS operations
"""
import asyncio
from pathlib import Path
from typing import List, Optional, Annotated
from enum import Enum

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID
from rich import print as rich_print
from rich.panel import Panel
from loguru import logger

from ..core.models import TTSRequest, AudioFormat, VoiceCategory
from ..core.config import get_settings, CLIConfig

# Initialize Typer app
app = typer.Typer(
    name="aas-tts",
    help="🎙️  AAS-TTS - Advanced Text-to-Speech System by Applied AI Systems",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True
)

# Initialize console and config
console = Console()
cli_config = CLIConfig()


class OutputFormat(str, Enum):
    """Available output formats for CLI"""
    wav = "wav"
    mp3 = "mp3" 
    flac = "flac"
    ogg = "ogg"


@app.command()
def synthesize(
    text: Annotated[str, typer.Argument(help="Text to synthesize")],
    voice: Annotated[str, typer.Option("--voice", "-v", help="Voice to use")] = "af_bella",
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output file path")] = None,
    format: Annotated[OutputFormat, typer.Option("--format", "-f", help="Audio format")] = OutputFormat.wav,
    speed: Annotated[float, typer.Option("--speed", "-s", help="Speech speed", min=0.1, max=3.0)] = 1.0,
    sample_rate: Annotated[Optional[int], typer.Option("--sample-rate", "-r", help="Sample rate")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing files")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", help="Verbose output")] = False,
):
    """
    🎙️  Synthesize text to speech
    
    Examples:
      aas-tts synthesize "Hello, world!" --voice af_bella --output hello.wav
      aas-tts synthesize "Fast speech" --speed 1.5 --format mp3
      aas-tts synthesize "Conference call test" --voice am_adam --sample-rate 16000
    """
    asyncio.run(_synthesize_async(
        text=text,
        voice=voice,
        output=output,
        format=AudioFormat(format.value),
        speed=speed,
        sample_rate=sample_rate,
        overwrite=overwrite,
        verbose=verbose
    ))


async def _synthesize_async(
    text: str,
    voice: str,
    output: Optional[Path],
    format: AudioFormat,
    speed: float,
    sample_rate: Optional[int],
    overwrite: bool,
    verbose: bool
):
    """Async synthesis implementation"""
    try:
        # Import service dynamically to avoid import issues
        from ..services.tts_service import get_tts_service
        
        # Show progress
        with Progress() as progress:
            task = progress.add_task("🎙️  Synthesizing...", total=100)
            
            # Initialize TTS service
            progress.update(task, advance=20, description="🔧 Loading TTS service...")
            service = await get_tts_service()
            
            # Create request
            progress.update(task, advance=20, description="📝 Preparing request...")
            request = TTSRequest(
                text=text,
                voice=voice,
                format=format,
                speed=speed,
                sample_rate=sample_rate
            )
            
            # Synthesize
            progress.update(task, advance=30, description="🎵 Generating audio...")
            
            if output:
                # Synthesize to file
                if not output.suffix:
                    output = output.with_suffix(f".{format.value}")
                
                response = await service.synthesize_to_file(
                    request=request,
                    output_path=output,
                    overwrite=overwrite
                )
            else:
                # Synthesize to memory
                response = await service.synthesize(request)
                
                # Auto-generate output filename
                safe_text = "".join(c for c in text[:30] if c.isalnum() or c.isspace()).strip()
                safe_text = "_".join(safe_text.split())
                output = Path(f"{safe_text}_{voice}.{format.value}")
                
                await service.audio_service.save_audio(
                    audio_data=response.audio_data,
                    output_path=output,
                    format=format
                )
                response.audio_path = output
            
            progress.update(task, advance=30, description="✅ Complete!")
        
        # Show results
        if response.success:
            rich_print(Panel.fit(
                f"✅ [green]Synthesis successful![/green]\n"
                f"📁 Output: [blue]{response.audio_path}[/blue]\n"
                f"⏱️  Processing time: [yellow]{response.processing_time:.2f}s[/yellow]" +
                (f"\n🎵 Audio duration: [cyan]{response.audio_duration:.2f}s[/cyan]" if response.audio_duration else ""),
                title="🎙️ TTS Results",
                border_style="green"
            ))
            
            if verbose:
                rich_print(f"📝 Text length: {response.text_length} characters")
                rich_print(f"🎤 Voice: {response.voice}")
                rich_print(f"📊 Sample rate: {response.sample_rate} Hz")
        else:
            rich_print(Panel.fit(
                f"❌ [red]Synthesis failed:[/red] {response.error}",
                title="🎙️ TTS Error",
                border_style="red"
            ))
            raise typer.Exit(1)
            
    except Exception as e:
        rich_print(Panel.fit(
            f"❌ [red]Error:[/red] {e}",
            title="🎙️ TTS Error", 
            border_style="red"
        ))
        if verbose:
            import traceback
            rich_print(traceback.format_exc())
        raise typer.Exit(1)


@app.command()
def list_voices(
    category: Annotated[Optional[str], typer.Option("--category", "-c", help="Filter by category")] = None,
    language: Annotated[Optional[str], typer.Option("--language", "-l", help="Filter by language")] = None,
    available_only: Annotated[bool, typer.Option("--available", help="Show only available voices")] = True,
):
    """
    🎵 List available voices
    
    Examples:
      aas-tts list-voices
      aas-tts list-voices --category af --language en-US
      aas-tts list-voices --available false
    """
    asyncio.run(_list_voices_async(category, language, available_only))


async def _list_voices_async(category: Optional[str], language: Optional[str], available_only: bool):
    """Async voice listing implementation"""
    try:
        from ..services.tts_service import get_tts_service
        
        service = await get_tts_service()
        voices = await service.get_available_voices()
        
        # Apply filters
        if category:
            voices = [v for v in voices if v.category.value == category]
        if language:
            voices = [v for v in voices if v.language == language]
        if available_only:
            voices = [v for v in voices if v.available]
        
        if not voices:
            rich_print("🔍 [yellow]No voices found matching criteria[/yellow]")
            return
        
        # Create table
        table = Table(title=f"🎵 Available Voices ({len(voices)})")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Name", style="green")
        table.add_column("Category", style="blue")
        table.add_column("Language", style="magenta")
        table.add_column("Gender", style="yellow")
        table.add_column("Status", justify="center")
        
        for voice in sorted(voices, key=lambda v: v.id):
            status = "✅" if voice.available else "❌"
            table.add_row(
                voice.id,
                voice.name,
                voice.category.value,
                voice.language,
                voice.gender,
                status
            )
        
        console.print(table)
        
    except Exception as e:
        rich_print(f"❌ [red]Error listing voices:[/red] {e}")
        raise typer.Exit(1)


@app.command()  
def info(
    voice: Annotated[str, typer.Argument(help="Voice ID to get information about")]
):
    """
    ℹ️  Get detailed information about a voice
    
    Example:
      aas-tts info af_bella
    """
    asyncio.run(_info_async(voice))


async def _info_async(voice: str):
    """Async voice info implementation"""
    try:
        from ..services.tts_service import get_tts_service
        
        service = await get_tts_service()
        voice_info = await service.get_voice_info(voice)
        
        if not voice_info:
            rich_print(f"❌ [red]Voice '{voice}' not found[/red]")
            raise typer.Exit(1)
        
        # Display voice information in a panel
        info_text = (
            f"📝 Name: [green]{voice_info.name}[/green]\n"
            f"📂 Category: [blue]{voice_info.category.value}[/blue]\n"
            f"🌍 Language: [magenta]{voice_info.language}[/magenta]\n"
            f"👤 Gender: [yellow]{voice_info.gender}[/yellow]\n"
            f"📊 Sample Rate: [cyan]{voice_info.sample_rate} Hz[/cyan]\n"
            f"✅ Available: {'Yes' if voice_info.available else 'No'}"
        )
        
        if voice_info.description:
            info_text += f"\n📄 Description: {voice_info.description}"
        
        rich_print(Panel.fit(
            info_text,
            title=f"🎤 Voice Information: {voice_info.id}",
            border_style="blue"
        ))
            
    except Exception as e:
        rich_print(f"❌ [red]Error getting voice info:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def health():
    """
    🏥 Check TTS service health
    """
    asyncio.run(_health_async())


async def _health_async():
    """Async health check implementation"""
    try:
        from ..services.tts_service import get_tts_service
        
        with console.status("[bold green]Checking service health..."):
            service = await get_tts_service()
            health_info = await service.health_check()
        
        if health_info["status"] == "healthy":
            health_text = (
                f"🎵 Voices available: [cyan]{health_info['voices_available']}[/cyan]\n"
                f"⏱️  Test synthesis: [yellow]{health_info['test_synthesis_time']}s[/yellow]\n"
                f"🎤 Default voice: [blue]{health_info['default_voice']}[/blue]\n"
                f"💻 Device: [magenta]{health_info['device']}[/magenta]"
            )
            
            rich_print(Panel.fit(
                f"✅ [green]TTS service is healthy[/green]\n{health_text}",
                title="🏥 Health Check",
                border_style="green"
            ))
        else:
            rich_print(Panel.fit(
                f"❌ [red]TTS service is unhealthy:[/red] {health_info['message']}",
                title="🏥 Health Check",
                border_style="red"
            ))
            raise typer.Exit(1)
            
    except Exception as e:
        rich_print(Panel.fit(
            f"❌ [red]Health check failed:[/red] {e}",
            title="🏥 Health Check",
            border_style="red"
        ))
        raise typer.Exit(1)


@app.command()
def version():
    """
    📋 Show version information
    """
    from .. import __version__, __author__, __email__
    from ..core.config import get_settings
    
    settings = get_settings()
    
    version_text = (
        f"🏢 {__author__}\n"
        f"📧 {__email__}\n"
        f"🐍 Python CLI with Typer\n"
        f"💻 Device: [magenta]{settings.get_device()}[/magenta]"
    )
    
    rich_print(Panel.fit(
        version_text,
        title=f"🎙️ AAS-TTS v{__version__}",
        border_style="blue"
    ))


@app.command()
def server(
    action: Annotated[str, typer.Argument(help="Action: start, stop, status")],
    host: Annotated[str, typer.Option("--host", help="Server host")] = "0.0.0.0",
    port: Annotated[int, typer.Option("--port", help="Server port")] = 8000,
    workers: Annotated[int, typer.Option("--workers", help="Number of workers")] = 1,
    reload: Annotated[bool, typer.Option("--reload", help="Enable auto-reload")] = False,
):
    """
    🚀 Server management commands
    
    Examples:
      aas-tts server start
      aas-tts server start --host 0.0.0.0 --port 8880 --workers 4
      aas-tts server stop
      aas-tts server status
    """
    if action == "start":
        asyncio.run(_start_server_async(host, port, workers, reload))
    elif action == "stop":
        rich_print("🛑 [yellow]Server stop not yet implemented[/yellow]")
    elif action == "status":
        rich_print("📊 [yellow]Server status not yet implemented[/yellow]")
    else:
        rich_print(f"❌ [red]Unknown action: {action}[/red]")
        rich_print("Valid actions: start, stop, status")
        raise typer.Exit(1)


async def _start_server_async(host: str, port: int, workers: int, reload: bool):
    """Start FastAPI server"""
    try:
        rich_print(Panel.fit(
            f"🚀 Starting AAS-TTS server...\n"
            f"🌐 Host: [blue]{host}[/blue]\n"
            f"🔌 Port: [blue]{port}[/blue]\n"
            f"👥 Workers: [blue]{workers}[/blue]\n"
            f"🔄 Reload: [blue]{reload}[/blue]",
            title="🚀 Server Starting",
            border_style="green"
        ))
        
        # Import and start server
        from ..api.server import create_app
        import uvicorn
        
        app = await create_app()
        
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
        
    except Exception as e:
        rich_print(f"❌ [red]Failed to start server:[/red] {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()