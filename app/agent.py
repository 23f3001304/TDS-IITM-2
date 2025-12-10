"""
Agent Core - The "Brain"
Pydantic AI Agent that orchestrates the quiz-solving process with tools
"""
import asyncio
import base64
import csv
import hashlib
import io
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional
from urllib.parse import urljoin, urlparse, parse_qs, unquote, quote

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from app.config import settings
from app.models import PageContent, QuizResult
from app.vision import vision
from app.sandbox import sandbox
from app.action import action


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class QuizDependencies:
    """Dependencies passed to the agent tools"""
    email: str
    secret: str
    current_url: str
    page_content: PageContent
    base_url: str = ""
    url_cache: dict = field(default_factory=dict)  # Cache for scraped URLs

    def __post_init__(self):
        parsed = urlparse(self.current_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        # Pre-cache current page content
        if self.page_content and self.page_content.text_content:
            self.url_cache[self.current_url] = self.page_content.text_content


class QuizAnswer(BaseModel):
    """Structured output from the agent"""
    answer: Any
    submission_url: str
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class QuizContext:
    """Context for a quiz solving session"""
    email: str
    secret: str
    current_url: str
    attempt_number: int = 0
    results: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a quiz solver. Answer questions directly without unnecessary tool calls.

IMPORTANT: The question is already provided to you. DO NOT scrape or download unless explicitly needed.

ONLY use tools when:
- Question asks to transcribe an AUDIO file -> transcribe_audio (or pip_install whisper then use execute_python)
- Question asks to analyze an IMAGE -> analyze_image  
- Question asks to analyze a VIDEO -> analyze_video
- Question asks to extract a ZIP/archive -> extract_zip
- Question asks to analyze CSV/data -> download_file + execute_python
- Question asks to fetch data from an API -> make_api_request (GET only)

AVAILABLE TOOLS:
- pip_install: Install Python packages (e.g., "openai-whisper pydub ffmpeg-python")
- run_shell_command: Run shell commands like ffmpeg, ffprobe
- execute_python: Run Python code (can also do pip install inside)
- download_file: Download files to local path
- transcribe_audio: Transcribe audio files
- analyze_image: OCR or analyze images
- extract_zip: Extract ZIP files

DO NOT use tools for:
- Reading the question (it's already provided)
- Scraping the current page (content is already given)
- POSTing answers (submission is automatic)

For command string questions (like "craft the command"):
- Just construct and return the command string directly
- Replace <your email> with the actual email provided
- Do NOT actually execute the command

CSV/JSON NORMALIZATION:
- snake_case means: lowercase, replace spaces with underscore, then strip extra chars
  e.g., "Joined Date" -> "joined_date", "ID" -> "id", "userName" -> "user_name"
- BUT if target keys are specified (e.g., id, name, joined, value), map to EXACTLY those keys
- For dates: use ISO-8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
- For integers: strip whitespace/commas, convert to int
- Use: import re; re.sub(r'[^a-z0-9]+', '_', col.lower()).strip('_') for snake_case
- If column like "Joined Date" maps to target "joined", just use "joined"

ANSWER FORMAT:
- Transcriptions: lowercase with spaces, include any numbers spoken
- Commands: exact string without extra quotes
- Numbers: just the number
- Text: exactly as requested
- JSON: compact format, no extra whitespace
"""

GUIDANCE_PROMPT = """You are a quiz solution strategist. Analyze the question and provide a BRIEF solution strategy.

Given a quiz question, identify:
1. What TYPE of problem is this? (command construction, data processing, API call, calculation, etc.)
2. What TOOLS are needed? (download_file, execute_python, transcribe_audio, etc.)
3. What are the KEY REQUIREMENTS? (specific format, normalization rules, exact output format)
4. Any GOTCHAS to watch for? (date formats, column name mapping, case sensitivity)

Be CONCISE - max 5 bullet points. Focus on what matters for getting the answer RIGHT.
"""

_provider = GoogleProvider(api_key=settings.google_api_key)
_model = GoogleModel("gemini-3-pro-preview", provider=_provider)  # Fast model for main solving
_flash_model = GoogleModel("gemini-2.5-flash", provider=_provider)  # Ultra-fast for guidance

# Guidance agent - lightweight, fast model, no tools
guidance_agent = Agent(
    model=_flash_model,
    output_type=str,
    system_prompt=GUIDANCE_PROMPT,
    retries=1,
)

quiz_agent = Agent(
    model=_model,
    deps_type=QuizDependencies,
    output_type=QuizAnswer,
    system_prompt=SYSTEM_PROMPT,
    retries=2,
)

# ---------------------------------------------------------------------------
# Core Tools
# ---------------------------------------------------------------------------

@quiz_agent.tool
async def scrape_url(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Scrape a URL and return its text content. Handles JavaScript-rendered pages.

    Args:
        url: The URL to scrape (can be relative or absolute)

    Returns:
        The text content of the page
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    # Check cache first
    if url in ctx.deps.url_cache:
        logger.info(f"Using cached content for: {url}")
        return ctx.deps.url_cache[url]

    logger.info(f"Scraping URL: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '').lower()
            raw_text = response.text
            
            # For JSON/API responses, return raw content (don't use Selenium)
            if 'json' in content_type or url.endswith('.json') or raw_text.strip().startswith(('{', '[')):
                logger.info(f"JSON response ({len(raw_text)} chars): {raw_text[:200]}...")
                ctx.deps.url_cache[url] = raw_text
                return raw_text

            soup = BeautifulSoup(raw_text, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()

            text = soup.get_text(separator='\n', strip=True)

            # Only use Selenium for HTML pages that appear JS-rendered (not for API/JSON)
            if len(text) < 50 and '<html' in raw_text.lower():
                logger.info(f"Content too short ({len(text)} chars), trying Selenium")
                page = await vision.extract_page_content(url)
                text = page.text_content

            logger.info(f"Scraped {len(text)} chars: {text[:200]}...")
            result = text if text else raw_text
            ctx.deps.url_cache[url] = result
            return result

    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        try:
            logger.info("Trying Selenium fallback")
            page = await vision.extract_page_content(url)
            result = page.text_content if page.text_content else "Error: Page has no content"
            ctx.deps.url_cache[url] = result
            return result
        except Exception as e2:
            return f"Error scraping URL: {e}. Selenium fallback: {e2}"


@quiz_agent.tool
async def execute_python(ctx: RunContext[QuizDependencies], code: str) -> str:
    """
    Execute Python code and return the output. Use for complex data analysis.
    Available libraries: pandas, numpy, scipy, json, csv, re, math, statistics.
    You can also use 'pip install' within the code if needed.
    The code MUST print the final answer.

    Args:
        code: Python code to execute

    Returns:
        The stdout output from the code
    """
    logger.info(f"Executing Python code:\n{code}")
    result = await sandbox.execute_code(code, timeout=120)  # Longer timeout for pip installs

    if result.success:
        output = result.stdout.strip()
        logger.info(f"Code output: {output}")
        return output if output else "Code executed but produced no output"
    else:
        logger.warning(f"Code failed: {result.stderr}")
        return f"Error: {result.stderr}"


@quiz_agent.tool
async def pip_install(ctx: RunContext[QuizDependencies], packages: str) -> str:
    """
    Install Python packages using pip.

    Args:
        packages: Space-separated list of packages to install (e.g., "openai-whisper pydub")

    Returns:
        Installation result
    """
    logger.info(f"Installing packages: {packages}")
    
    code = f'''
import subprocess
import sys

packages = "{packages}".split()
for pkg in packages:
    print(f"Installing {{pkg}}...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to install {{pkg}}: {{result.stderr}}")
    else:
        print(f"Installed {{pkg}}")
print("Done!")
'''
    
    result = await sandbox.execute_code(code, timeout=300)  # 5 min timeout for installs
    
    if result.success:
        return result.stdout.strip()
    else:
        return f"Installation failed: {result.stderr}"


@quiz_agent.tool
async def run_shell_command(ctx: RunContext[QuizDependencies], command: str) -> str:
    """
    Run a shell command (useful for ffmpeg, ffprobe, etc.).

    Args:
        command: Shell command to run

    Returns:
        Command output
    """
    logger.info(f"Running shell command: {command}")
    
    code = f'''
import subprocess
result = subprocess.run({repr(command)}, shell=True, capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print("Return code:", result.returncode)
'''
    
    result = await sandbox.execute_code(code, timeout=120)
    
    if result.success:
        return result.stdout.strip()
    else:
        return f"Command failed: {result.stderr}"


@quiz_agent.tool
async def download_file(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Download a file and return the local path.

    Args:
        url: URL of the file to download

    Returns:
        Local path where the file was saved
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Downloading file: {url}")

    try:
        path = await sandbox.download_file(url)
        logger.info(f"Downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return f"Error downloading file: {e}"


@quiz_agent.tool
async def read_file_content(ctx: RunContext[QuizDependencies], url: str, max_lines: int = 30) -> str:
    """
    Preview file content from a URL or local path. Returns first lines to understand format.

    Args:
        url: URL of the file to read OR local file path
        max_lines: Maximum number of lines to return (default 30)

    Returns:
        File preview with line count
    """
    # Handle local file paths
    if not url.startswith('http'):
        # Check if it's a local path
        from pathlib import Path
        local_path = Path(url)
        if local_path.exists():
            logger.info(f"Reading local file: {url}")
            try:
                with open(local_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                logger.error(f"Failed to read local file {url}: {e}")
                return f"Error reading local file: {e}"
        else:
            # Try as relative URL
            url = urljoin(ctx.deps.current_url, url)
            logger.info(f"Reading file preview: {url}")
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    content = response.text
            except Exception as e:
                logger.error(f"Failed to read {url}: {e}")
                return f"Error reading file: {e}"
    else:
        # Check cache first
        cache_key = f"file:{url}"
        if cache_key in ctx.deps.url_cache:
            logger.info(f"Using cached file content for: {url}")
            content = ctx.deps.url_cache[cache_key]
        else:
            logger.info(f"Reading file preview: {url}")
            try:
                async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    content = response.text
                    ctx.deps.url_cache[cache_key] = content
            except Exception as e:
                logger.error(f"Failed to read {url}: {e}")
                return f"Error reading file: {e}"

    try:
        lines = content.split('\n')
        total_lines = len(lines)
        preview = '\n'.join(lines[:max_lines])

        # Try to detect file type
        file_type = "unknown"
        if url.endswith('.csv') or ',' in lines[0]:
            file_type = "CSV"
        elif url.endswith('.json') or content.strip().startswith(('{', '[')):
            file_type = "JSON"
        elif url.endswith('.txt'):
            file_type = "TEXT"

        logger.info(f"File preview: {file_type}, {total_lines} lines")
        return f"File type: {file_type}\nTotal lines: {total_lines}\nFirst {max_lines} lines:\n{preview}"
    except Exception as e:
        logger.error(f"Failed to process {url}: {e}")
        return f"Error processing file: {e}"


# ---------------------------------------------------------------------------
# Media and Archive Tools
# ---------------------------------------------------------------------------

@quiz_agent.tool
async def transcribe_audio(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Download and transcribe an audio file using OpenAI Whisper.
    Supports mp3, wav, opus, m4a, webm, flac formats.

    Args:
        url: URL of the audio file to transcribe

    Returns:
        Transcribed text from the audio
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Transcribing audio: {url}")

    try:
        # Download the audio file
        local_path = await sandbox.download_file(url)
        # Convert to forward slashes to avoid Windows path escaping issues
        safe_path = local_path.replace('\\', '/')
        
        # Use soundfile + scipy + whisper approach (proven to work)
        code = f'''
import soundfile as sf
import numpy as np
import whisper
from scipy import signal

path = "{safe_path}"
data, samplerate = sf.read(path)

# Convert to mono if stereo
if len(data.shape) > 1:
    data = data.mean(axis=1)

# Resample to 16kHz for whisper
if samplerate != 16000:
    new_len = int(len(data) * 16000 / samplerate)
    data = signal.resample(data, new_len)

data = data.astype(np.float32)

# Load whisper model and transcribe
model = whisper.load_model("base")
result = model.transcribe(data)
print(result["text"].strip())
'''
        result = await sandbox.execute_code(code, timeout=120)
        
        if result.success and result.stdout.strip():
            transcription = result.stdout.strip()
            logger.info(f"Transcription: {transcription}")
            return transcription
        else:
            return f"Transcription failed: {result.stderr}"
            
    except Exception as e:
        logger.error(f"Audio transcription error: {e}")
        return f"Error transcribing audio: {e}"


@quiz_agent.tool
async def analyze_image(ctx: RunContext[QuizDependencies], url: str, task: str = "describe") -> str:
    """
    Download and analyze an image. Can extract text (OCR), describe content, or detect objects.

    Args:
        url: URL of the image file
        task: One of "ocr" (extract text), "describe" (describe image), "detect" (detect objects)

    Returns:
        Analysis result based on task
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Analyzing image ({task}): {url}")

    try:
        local_path = await sandbox.download_file(url)
        
        if task == "ocr":
            code = f'''
import pytesseract
from PIL import Image

try:
    img = Image.open("{local_path}")
    text = pytesseract.image_to_string(img)
    print(text.strip())
except Exception as e:
    print(f"OCR Error: {{e}}")
'''
        elif task == "describe":
            code = f'''
from PIL import Image
import os

try:
    img = Image.open("{local_path}")
    width, height = img.size
    mode = img.mode
    format_type = img.format
    print(f"Image: {{width}}x{{height}}, mode={{mode}}, format={{format_type}}")
    
    # Try to get more info
    if hasattr(img, 'info'):
        for k, v in img.info.items():
            if isinstance(v, (str, int, float)):
                print(f"{{k}}: {{v}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        else:  # detect
            code = f'''
from PIL import Image

try:
    img = Image.open("{local_path}")
    # Basic analysis
    colors = img.getcolors(maxcolors=10000)
    if colors:
        colors = sorted(colors, reverse=True)[:10]
        print("Top colors (count, rgba):")
        for count, color in colors:
            print(f"  {{count}}: {{color}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        
        result = await sandbox.execute_code(code, timeout=60)
        
        if result.success:
            return result.stdout.strip() or "No output from image analysis"
        else:
            return f"Image analysis failed: {result.stderr}"
            
    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"Error analyzing image: {e}"


@quiz_agent.tool
async def extract_zip(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Download and extract a ZIP file, returning the list of contents.

    Args:
        url: URL of the ZIP file

    Returns:
        List of files in the archive and their contents if text-based
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting ZIP: {url}")

    try:
        local_path = await sandbox.download_file(url)
        
        code = f'''
import zipfile
import os

try:
    with zipfile.ZipFile("{local_path}", 'r') as zf:
        print("=== ZIP Contents ===")
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"{{name}} ({{info.file_size}} bytes)")
        
        print("\\n=== File Contents ===")
        for name in zf.namelist():
            if not name.endswith('/'):  # Skip directories
                try:
                    content = zf.read(name)
                    # Try to decode as text
                    try:
                        text = content.decode('utf-8')
                        print(f"\\n--- {{name}} ---")
                        print(text[:2000])  # First 2000 chars
                        if len(text) > 2000:
                            print("... (truncated)")
                    except:
                        print(f"\\n--- {{name}} --- (binary file, {{len(content)}} bytes)")
                except Exception as e:
                    print(f"Error reading {{name}}: {{e}}")
except Exception as e:
    print(f"ZIP Error: {{e}}")
'''
        
        result = await sandbox.execute_code(code, timeout=60)
        
        if result.success:
            return result.stdout.strip() or "Empty ZIP file"
        else:
            return f"ZIP extraction failed: {result.stderr}"
            
    except Exception as e:
        logger.error(f"ZIP extraction error: {e}")
        return f"Error extracting ZIP: {e}"


@quiz_agent.tool
async def analyze_video(ctx: RunContext[QuizDependencies], url: str, task: str = "info") -> str:
    """
    Download and analyze a video file. Can extract info, frames, or audio.

    Args:
        url: URL of the video file
        task: One of "info" (get metadata), "frames" (extract key frames), "audio" (extract and transcribe audio)

    Returns:
        Analysis result based on task
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Analyzing video ({task}): {url}")

    try:
        local_path = await sandbox.download_file(url)
        
        if task == "info":
            code = f'''
import subprocess
import json

try:
    # Use ffprobe to get video info
    result = subprocess.run([
        'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
        "{local_path}"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        fmt = data.get('format', {{}})
        print(f"Duration: {{fmt.get('duration', 'unknown')}} seconds")
        print(f"Format: {{fmt.get('format_name', 'unknown')}}")
        print(f"Size: {{fmt.get('size', 'unknown')}} bytes")
        
        for stream in data.get('streams', []):
            codec_type = stream.get('codec_type', 'unknown')
            codec_name = stream.get('codec_name', 'unknown')
            if codec_type == 'video':
                print(f"Video: {{stream.get('width')}}x{{stream.get('height')}}, {{codec_name}}")
            elif codec_type == 'audio':
                print(f"Audio: {{codec_name}}, {{stream.get('sample_rate')}} Hz")
    else:
        print(f"ffprobe error: {{result.stderr}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        elif task == "audio":
            # Extract audio and transcribe
            code = f'''
import subprocess
import os

try:
    # Extract audio to wav
    audio_path = "{local_path}.wav"
    result = subprocess.run([
        'ffmpeg', '-i', "{local_path}", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        audio_path, '-y'
    ], capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(audio_path):
        # Try whisper
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            print(result["text"].strip().lower())
        except ImportError:
            print("Whisper not available, audio extracted to: " + audio_path)
    else:
        print(f"Audio extraction failed: {{result.stderr}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        else:  # frames
            code = f'''
import subprocess
import os

try:
    # Extract frames
    frame_dir = "{local_path}_frames"
    os.makedirs(frame_dir, exist_ok=True)
    
    result = subprocess.run([
        'ffmpeg', '-i', "{local_path}", '-vf', 'fps=1', '-frames:v', '5',
        f"{{frame_dir}}/frame_%03d.png", '-y'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        frames = os.listdir(frame_dir)
        print(f"Extracted {{len(frames)}} frames:")
        for f in sorted(frames):
            print(f"  {{frame_dir}}/{{f}}")
    else:
        print(f"Frame extraction failed: {{result.stderr}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        
        result = await sandbox.execute_code(code, timeout=120)
        
        if result.success:
            return result.stdout.strip() or "No output from video analysis"
        else:
            return f"Video analysis failed: {result.stderr}"
            
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        return f"Error analyzing video: {e}"


@quiz_agent.tool
async def extract_archive(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Download and extract any archive (zip, tar, tar.gz, 7z, rar).

    Args:
        url: URL of the archive file

    Returns:
        List of files and their contents
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting archive: {url}")

    try:
        local_path = await sandbox.download_file(url)
        
        code = f'''
import os
import tarfile
import zipfile

filepath = "{local_path}"
extract_dir = filepath + "_extracted"
os.makedirs(extract_dir, exist_ok=True)

try:
    # Try ZIP
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(extract_dir)
            print("Extracted ZIP archive")
    # Try TAR
    elif tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r:*') as tf:
            tf.extractall(extract_dir)
            print("Extracted TAR archive")
    else:
        print("Unknown archive format")
        
    # List contents
    print("\\n=== Contents ===")
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, extract_dir)
            size = os.path.getsize(full_path)
            print(f"{{rel_path}} ({{size}} bytes)")
            
            # Read text files
            if size < 10000:
                try:
                    with open(full_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        print(f"--- Content of {{rel_path}} ---")
                        print(content[:2000])
                        if len(content) > 2000:
                            print("... (truncated)")
                except:
                    pass
except Exception as e:
    print(f"Error: {{e}}")
'''
        
        result = await sandbox.execute_code(code, timeout=60)
        
        if result.success:
            return result.stdout.strip() or "Empty archive"
        else:
            return f"Archive extraction failed: {result.stderr}"
            
    except Exception as e:
        logger.error(f"Archive extraction error: {e}")
        return f"Error extracting archive: {e}"


@quiz_agent.tool
def get_page_links(ctx: RunContext[QuizDependencies]) -> list[str]:
    """Get all links from the current quiz page."""
    return ctx.deps.page_content.links


@quiz_agent.tool
def get_page_text(ctx: RunContext[QuizDependencies]) -> str:
    """Get the text content of the current quiz page."""
    return ctx.deps.page_content.text_content



@quiz_agent.tool
def compute_hash(ctx: RunContext[QuizDependencies], text: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of text. Supports md5, sha1, sha256, sha512.

    Args:
        text: Text to hash
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)

    Returns:
        Hexadecimal hash string
    """
    text_bytes = text.encode('utf-8')

    algorithms = {
        'md5': hashlib.md5,
        'sha1': hashlib.sha1,
        'sha256': hashlib.sha256,
        'sha512': hashlib.sha512,
    }

    algo = algorithms.get(algorithm.lower())
    if not algo:
        return f"Error: Unknown algorithm '{algorithm}'. Use: md5, sha1, sha256, sha512"

    result = algo(text_bytes).hexdigest()
    logger.info(f"Hash ({algorithm}): {text[:50]}... -> {result}")
    return result


@quiz_agent.tool
def encode_decode(ctx: RunContext[QuizDependencies], text: str, operation: str) -> str:
    """
    Encode or decode text using various methods.

    Args:
        text: Text to encode/decode
        operation: One of: base64_encode, base64_decode, url_encode, url_decode, hex_encode, hex_decode

    Returns:
        Encoded/decoded string
    """
    try:
        if operation == "base64_encode":
            return base64.b64encode(text.encode()).decode()
        elif operation == "base64_decode":
            return base64.b64decode(text).decode()
        elif operation == "url_encode":
            return quote(text)
        elif operation == "url_decode":
            return unquote(text)
        elif operation == "hex_encode":
            return text.encode().hex()
        elif operation == "hex_decode":
            return bytes.fromhex(text).decode()
        else:
            return f"Error: Unknown operation. Use: base64_encode/decode, url_encode/decode, hex_encode/decode"
    except Exception as e:
        return f"Error: {e}"


@quiz_agent.tool
def compute_email_code(ctx: RunContext[QuizDependencies], email: str) -> str:
    """
    Compute the secret code for an email address based on SHA-1 hash.

    Args:
        email: The email address to compute the code for

    Returns:
        The computed code as a string
    """
    sha1_hash = hashlib.sha1(email.encode()).hexdigest()
    code = int(sha1_hash[:4], 16)
    logger.info(f"Computed email code for {email}: {code}")
    return str(code)


# ---------------------------------------------------------------------------
# Text Processing Tools
# ---------------------------------------------------------------------------

@quiz_agent.tool
def extract_with_regex(ctx: RunContext[QuizDependencies], text: str, pattern: str, group: int = 0) -> str:
    """
    Extract data from text using a regex pattern.

    Args:
        text: Text to search in
        pattern: Regex pattern to match
        group: Which capture group to return (0 = full match)

    Returns:
        All matches joined by newlines, or error message
    """
    try:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not matches:
            return "No matches found"

        # Handle groups
        if isinstance(matches[0], tuple):
            results = [m[group] if group < len(m) else m[0] for m in matches]
        else:
            results = matches

        logger.info(f"Regex found {len(results)} matches")
        return '\n'.join(str(r) for r in results)
    except Exception as e:
        return f"Regex error: {e}"


@quiz_agent.tool
def extract_numbers(ctx: RunContext[QuizDependencies], text: str, as_float: bool = True) -> str:
    """
    Extract all numbers from text.

    Args:
        text: Text to extract numbers from
        as_float: If True, parse as floats. If False, integers only.

    Returns:
        List of numbers found, one per line
    """
    if as_float:
        pattern = r'-?\d+\.?\d*'
        numbers = [float(n) for n in re.findall(pattern, text)]
    else:
        pattern = r'-?\d+'
        numbers = [int(n) for n in re.findall(pattern, text)]

    logger.info(f"Extracted {len(numbers)} numbers")
    return '\n'.join(str(n) for n in numbers)


@quiz_agent.tool
def count_occurrences(ctx: RunContext[QuizDependencies], text: str, pattern: str, case_sensitive: bool = False) -> str:
    """
    Count occurrences of a pattern in text.

    Args:
        text: Text to search in
        pattern: Pattern to count (can be regex)
        case_sensitive: Whether to match case

    Returns:
        Count as string
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        matches = re.findall(pattern, text, flags)
        count = len(matches)
        logger.info(f"Found {count} occurrences of '{pattern}'")
        return str(count)
    except Exception as e:
        return f"Error: {e}"


@quiz_agent.tool
def parse_json(ctx: RunContext[QuizDependencies], json_text: str, query_path: str = "") -> str:
    """
    Parse JSON and optionally extract a value using dot notation path.

    Args:
        json_text: JSON string to parse
        query_path: Optional path like "data.items.0.name" or "users[0].email"

    Returns:
        Parsed JSON or extracted value
    """
    try:
        data = json.loads(json_text)

        if not query_path:
            return json.dumps(data, indent=2)

        # Navigate the path
        current = data
        for key in query_path.replace('[', '.').replace(']', '').split('.'):
            if not key:
                continue
            if isinstance(current, list):
                current = current[int(key)]
            elif isinstance(current, dict):
                current = current.get(key, current.get(int(key) if key.isdigit() else key))
            else:
                return f"Cannot navigate further at '{key}'"

        if isinstance(current, (dict, list)):
            return json.dumps(current, indent=2)
        return str(current)

    except json.JSONDecodeError as e:
        return f"Invalid JSON: {e}"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Math and Statistics Tools
# ---------------------------------------------------------------------------

@quiz_agent.tool
def do_math(ctx: RunContext[QuizDependencies], expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, exp, abs, round, floor, ceil

    Args:
        expression: Math expression like "sqrt(16) + 2**3" or "sum([1,2,3,4])"

    Returns:
        Result as string
    """
    # Safe math functions
    safe_dict = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
        'pow': pow,
        'sum': sum,
        'min': min,
        'max': max,
        'len': len,
        'pi': math.pi,
        'e': math.e,
        'mean': statistics.mean,
        'median': statistics.median,
        'stdev': statistics.stdev,
    }

    try:
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        logger.info(f"Math: {expression} = {result}")
        return str(result)
    except Exception as e:
        return f"Math error: {e}"


# ---------------------------------------------------------------------------
# Date and Time Tools
# ---------------------------------------------------------------------------

@quiz_agent.tool
def get_date_info(ctx: RunContext[QuizDependencies], date_string: str = "", operation: str = "parse") -> str:
    """
    Parse and manipulate dates.

    Args:
        date_string: Date string to parse (empty = current date)
        operation: parse, day_of_week, days_until, days_since, add_days:N, format:FORMAT

    Returns:
        Date information
    """
    try:
        # Parse date or use now
        if not date_string:
            dt = datetime.now()
        else:
            # Try common formats
            formats = [
                '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y',
                '%Y-%m-%d %H:%M:%S', '%d-%m-%Y',
                '%B %d, %Y', '%b %d, %Y',
            ]
            dt = None
            for fmt in formats:
                try:
                    dt = datetime.strptime(date_string, fmt)
                    break
                except ValueError:
                    continue
            if not dt:
                return f"Could not parse date: {date_string}"

        if operation == "parse":
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        elif operation == "day_of_week":
            return dt.strftime('%A')
        elif operation == "days_until":
            delta = dt - datetime.now()
            return str(delta.days)
        elif operation == "days_since":
            delta = datetime.now() - dt
            return str(delta.days)
        elif operation.startswith("add_days:"):
            days = int(operation.split(':')[1])
            new_dt = dt + timedelta(days=days)
            return new_dt.strftime('%Y-%m-%d')
        elif operation.startswith("format:"):
            fmt = operation.split(':', 1)[1]
            return dt.strftime(fmt)
        else:
            return f"Unknown operation: {operation}"

    except Exception as e:
        return f"Date error: {e}"


# ---------------------------------------------------------------------------
# HTTP and API Tools
# ---------------------------------------------------------------------------

@quiz_agent.tool
async def make_api_request(
    ctx: RunContext[QuizDependencies],
    url: str,
    method: str = "GET",
    body: str = "",
    headers: str = ""
) -> str:
    """
    Make an HTTP request to an API endpoint.

    Args:
        url: API URL (can be relative)
        method: HTTP method (GET, POST, PUT, DELETE)
        body: Request body (JSON string for POST/PUT)
        headers: Optional headers as JSON string

    Returns:
        Response body
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    # Cache GET requests only
    cache_key = f"api:{method}:{url}"
    if method.upper() == "GET" and cache_key in ctx.deps.url_cache:
        logger.info(f"Using cached API response for: {url}")
        return ctx.deps.url_cache[cache_key]

    logger.info(f"API request: {method} {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            req_headers = json.loads(headers) if headers else {}
            req_body = json.loads(body) if body else None

            response = await client.request(
                method=method.upper(),
                url=url,
                json=req_body if req_body else None,
                headers=req_headers
            )

            logger.info(f"API response: {response.status_code}")
            result = response.text
            
            # Cache successful GET requests
            if method.upper() == "GET" and response.status_code == 200:
                ctx.deps.url_cache[cache_key] = result
            
            return result

    except Exception as e:
        return f"API error: {e}"


@quiz_agent.tool
async def analyze_csv_data(
    ctx: RunContext[QuizDependencies],
    url: str,
    operation: str,
    column: str = "",
    filter_expr: str = ""
) -> str:
    """
    Quick CSV analysis without writing Python code.

    Args:
        url: URL of CSV file
        operation: One of: sum, count, mean, max, min, unique, filter_count, filter_sum
        column: Column name or index (0-based) for the operation
        filter_expr: Filter expression like ">100" or "==value" or "contains:text"

    Returns:
        Analysis result
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            reader = csv.reader(io.StringIO(response.text))
            rows = list(reader)

            if not rows:
                return "Empty CSV"

            first_row = rows[0]
            has_header = not all(cell.replace('.', '').replace('-', '').isdigit() for cell in first_row if cell)

            if has_header:
                headers = first_row
                data_rows = rows[1:]
            else:
                headers = [str(i) for i in range(len(first_row))]
                data_rows = rows

            # Find column index
            col_idx = 0
            if column:
                if column.isdigit():
                    col_idx = int(column)
                elif column in headers:
                    col_idx = headers.index(column)

            values = []
            for row in data_rows:
                if col_idx < len(row) and row[col_idx]:
                    val = row[col_idx].strip()
                    try:
                        values.append(float(val))
                    except ValueError:
                        values.append(val)

            if filter_expr:
                filtered = []
                for v in values:
                    try:
                        if filter_expr.startswith('>='):
                            if float(v) >= float(filter_expr[2:]):
                                filtered.append(v)
                        elif filter_expr.startswith('<='):
                            if float(v) <= float(filter_expr[2:]):
                                filtered.append(v)
                        elif filter_expr.startswith('>'):
                            if float(v) > float(filter_expr[1:]):
                                filtered.append(v)
                        elif filter_expr.startswith('<'):
                            if float(v) < float(filter_expr[1:]):
                                filtered.append(v)
                        elif filter_expr.startswith('=='):
                            if str(v) == filter_expr[2:]:
                                filtered.append(v)
                        elif filter_expr.startswith('!='):
                            if str(v) != filter_expr[2:]:
                                filtered.append(v)
                        elif filter_expr.startswith('contains:'):
                            if filter_expr[9:] in str(v):
                                filtered.append(v)
                    except (ValueError, TypeError):
                        pass
                values = filtered

            numeric_values = [v for v in values if isinstance(v, (int, float))]

            if operation == "sum":
                result = sum(numeric_values)
            elif operation == "count":
                result = len(values)
            elif operation == "mean":
                result = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            elif operation == "max":
                result = max(numeric_values) if numeric_values else None
            elif operation == "min":
                result = min(numeric_values) if numeric_values else None
            elif operation == "unique":
                result = len(set(values))
            elif operation == "filter_count":
                result = len(values)
            elif operation == "filter_sum":
                result = sum(numeric_values)
            elif operation == "list":
                return '\n'.join(str(v) for v in values[:50])
            else:
                return f"Unknown operation: {operation}"

            logger.info(f"CSV analysis: {operation} = {result}")
            return str(result)

    except Exception as e:
        logger.error(f"CSV analysis error: {e}")
        return f"Error: {e}"


class QuizSolver:
    """Main quiz solver using Pydantic AI agent"""

    async def solve_quiz(self, context: QuizContext) -> list[QuizResult]:
        """Main entry point - solves a quiz chain starting from the initial URL."""
        logger.info(f"Starting quiz solving for {context.email}")
        logger.info(f"Initial URL: {context.current_url}")

        while True:
            try:
                result = await self._solve_single_question(context)
                context.results.append(result)

                if result.correct:
                    if result.next_url:
                        logger.info(f"Correct! Moving to next question: {result.next_url}")
                        context.current_url = result.next_url
                        context.attempt_number = 0
                    else:
                        logger.info("Quiz completed successfully!")
                        break
                else:
                    # Wrong answer - check if we should retry or move on
                    reason = (result.message or "").lower()
                    is_delay_timeout = "delay" in reason and "180" in reason
                    
                    if result.next_url:
                        # Has next URL - decide whether to retry or move on
                        if is_delay_timeout:
                            logger.info(f"Delay timeout, moving to next: {result.next_url}")
                            context.current_url = result.next_url
                            context.attempt_number = 0
                        else:
                            # Wrong but not timeout - retry
                            context.attempt_number += 1
                            if context.attempt_number >= settings.max_retries_per_question:
                                logger.warning(f"Max retries reached, moving to next: {result.next_url}")
                                context.current_url = result.next_url
                                context.attempt_number = 0
                            else:
                                logger.info(f"Wrong answer, retrying (attempt {context.attempt_number + 1}/{settings.max_retries_per_question})")
                                context.results.pop()
                                continue
                    else:
                        # No next URL
                        if is_delay_timeout:
                            logger.info("Delay timeout and no next URL, quiz ended")
                            break
                        # Retry without next URL
                        context.attempt_number += 1
                        if context.attempt_number >= settings.max_retries_per_question:
                            logger.warning(f"Max retries reached, no next URL, quiz ended")
                            break
                        logger.info(f"Wrong answer, retrying (attempt {context.attempt_number + 1}/{settings.max_retries_per_question})")
                        context.results.pop()
                        continue

            except Exception as e:
                logger.error(f"Error solving question: {e}", exc_info=True)
                # On exception, try emergency submission to get next_url
                try:
                    emergency_result = await self._emergency_submit(context)
                    context.results.append(emergency_result)
                    if emergency_result.next_url:
                        logger.info(f"Emergency submit got next URL: {emergency_result.next_url}")
                        context.current_url = emergency_result.next_url
                        context.attempt_number = 0
                        continue
                except Exception as e2:
                    logger.error(f"Emergency submit also failed: {e2}")
                
                context.results.append(QuizResult(
                    url=context.current_url,
                    answer=None,
                    correct=False,
                    message=str(e)
                ))
                context.attempt_number += 1
                if context.attempt_number >= settings.max_retries_per_question:
                    logger.warning(f"Max retries reached on error, stopping")
                    break

        return context.results

    async def _emergency_submit(self, context: QuizContext) -> QuizResult:
        """Emergency submission to try to get a next_url when agent fails."""
        url = context.current_url
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        # Submit a placeholder answer to hopefully get next_url
        response = await action.submit_answer(
            endpoint=f"{base_url}/submit",
            email=context.email,
            secret=context.secret,
            url=url,
            answer="error_fallback"
        )
        
        return QuizResult(
            url=url,
            answer="error_fallback",
            correct=response.correct,
            message=response.message or response.reason,
            next_url=response.url
        )

    async def _solve_single_question(self, context: QuizContext) -> QuizResult:
        url = context.current_url
        logger.info(f"Solving question: {url}")
        
        # Fast page extraction - single attempt, no retries for speed
        page = await vision.extract_page_content(url)
        if not page or not page.text_content:
            # One retry on failure
            page = await vision.extract_page_content(url)
        
        if not page:
            raise RuntimeError(f"Failed to extract page content from {url}")

        logger.info(f"Page text: {page.text_content[:300]}...")
        logger.info(f"Links: {page.links}")

        deps = QuizDependencies(
            email=context.email,
            secret=context.secret,
            current_url=url,
            page_content=page
        )

        # Get guidance (fast, uses flash model)
        guidance = await self._get_solution_guidance(page)
        logger.info(f"Guidance: {guidance[:200] if guidance else 'none'}")

        # Step 2: Build prompt with guidance and solve
        prompt = self._build_prompt(url, page, deps, guidance)

        try:
            result = await quiz_agent.run(prompt, deps=deps)
            agent_answer = result.output
            
            # Post-process the answer
            final_answer = self._postprocess_answer(agent_answer.answer, page.text_content)

            logger.info(f"Agent result: answer={final_answer}, submission_url={agent_answer.submission_url}")

            submission_endpoint = self._resolve_submission_url(agent_answer, page, deps)

            response = await action.submit_answer(
                endpoint=submission_endpoint,
                email=context.email,
                secret=context.secret,
                url=url,
                answer=final_answer
            )

            return QuizResult(
                url=url,
                answer=final_answer,
                correct=response.correct,
                message=response.message or response.reason,
                next_url=response.url
            )

        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            return await self._fallback_solve(context, page, deps)

    def _postprocess_answer(self, answer: Any, page_text: str) -> Any:
        """Post-process the answer to fix common issues."""
        if not isinstance(answer, str):
            return answer
        
        # Fix: Remove unnecessary quotes around URLs in command strings
        # e.g., 'uv http get "https://..." -H' -> 'uv http get https://... -H'
        if answer.startswith('uv http get "') or answer.startswith("uv http get '"):
            # Remove quotes around the URL
            answer = re.sub(r'^(uv http get )["\']([^"\']+)["\'](.*)$', r'\1\2\3', answer)
            logger.info(f"Postprocess: removed quotes from uv command -> {answer}")
        
        return answer

    async def _get_solution_guidance(self, page: PageContent) -> str:
        """Get solution guidance from the guidance agent before solving."""
        try:
            guidance_prompt = f"""Analyze this quiz question and provide a brief solution strategy:

QUESTION:
{page.text_content}

AVAILABLE FILES/LINKS:
{page.links if page.links else "(none)"}

Provide 3-5 bullet points on:
- Problem type and approach
- Tools needed (if any)
- Key requirements/format rules
- Potential gotchas
"""
            result = await guidance_agent.run(guidance_prompt)
            return result.output
        except Exception as e:
            logger.warning(f"Guidance agent failed: {e}")
            return ""

    def _build_prompt(self, url: str, page: PageContent, deps: QuizDependencies, guidance: str = "") -> str:
        links_formatted = (
            "\n".join([f"  - {link}" for link in page.links])
            if page.links else "  (none)"
        )

        guidance_section = ""
        if guidance:
            guidance_section = f"""
SOLUTION STRATEGY (follow this guidance):
{guidance}

"""

        return f"""Answer this question. DO NOT use tools unless the question requires downloading/analyzing a file.

Email: {deps.email}
{guidance_section}
QUESTION:
{page.text_content}

AVAILABLE FILES (only use if question asks to download/analyze):
{links_formatted}

Return your answer directly. For command questions, just return the command string with {deps.email} replacing <your email>.
Set submission_url to: {deps.base_url}/submit
"""

    def _resolve_submission_url(
        self,
        agent_answer: QuizAnswer,
        page: PageContent,
        deps: QuizDependencies
    ) -> str:
        """Resolve the submission URL from agent answer or fallbacks."""
        submission_endpoint = None
        
        # Priority 1: Check if page has a specific submission endpoint (from vision)
        if page.submission_endpoint and 'submit' in page.submission_endpoint.lower():
            submission_endpoint = page.submission_endpoint
            logger.info(f"Using vision submission endpoint: {submission_endpoint}")
        
        # Priority 2: Check links for submit endpoint
        if not submission_endpoint and page.links:
            for link in page.links:
                if 'submit' in link.lower():
                    submission_endpoint = link
                    logger.info(f"Using submit link from page: {submission_endpoint}")
                    break
        
        # Always use /submit endpoint - the "url = X" in questions refers to payload field, not endpoint
        if not submission_endpoint:
            submission_endpoint = f"{deps.base_url}/submit"
            logger.info(f"Using /submit endpoint: {submission_endpoint}")
        
        return submission_endpoint

    async def _fallback_solve(
        self,
        context: QuizContext,
        page: PageContent,
        deps: QuizDependencies
    ) -> QuizResult:
        """Fallback solver when AI agent fails."""
        logger.info("Using fallback solver")

        url = context.current_url
        text_lower = page.text_content.lower()
        submission_endpoint = page.submission_endpoint or f"{deps.base_url}/submit"
        answer = None

        # Try various patterns
        if "anything you want" in text_lower or "any answer" in text_lower:
            answer = "test_answer"
        elif "scrape" in text_lower:
            answer = await self._handle_scrape_fallback(page, url)
        elif "email" in text_lower and "code" in text_lower:
            # Email code pattern
            sha1_hash = hashlib.sha1(context.email.encode()).hexdigest()
            answer = str(int(sha1_hash[:4], 16))
        elif "hash" in text_lower:
            # Try to find what to hash
            match = re.search(r'hash\s+of\s+["\']?(\w+)["\']?', text_lower)
            if match:
                answer = hashlib.sha256(match.group(1).encode()).hexdigest()

        if not answer:
            answer = "unknown"

        logger.info(f"Fallback answer: {answer}")

        response = await action.submit_answer(
            endpoint=submission_endpoint,
            email=context.email,
            secret=context.secret,
            url=url,
            answer=answer
        )

        return QuizResult(
            url=url,
            answer=answer,
            correct=response.correct,
            message=response.message or response.reason,
            next_url=response.url
        )

    async def _handle_scrape_fallback(self, page: PageContent, current_url: str) -> Optional[str]:
        """Handle scraping in fallback mode."""
        scrape_urls = [u for u in page.links if 'submit' not in u.lower() and u != current_url]

        if not scrape_urls:
            return None

        for scrape_url in scrape_urls[:3]:  # Try up to 3 URLs
            try:
                scraped_page = await vision.extract_page_content(scrape_url)
                scraped_text = scraped_page.text_content
                logger.info(f"Fallback scraped: {scraped_text[:200]}")

                # Look for secret code patterns
                patterns = [
                    r'secret\s+code\s+is\s+(\d+)',
                    r'code\s*[:\-=]\s*(\d+)',
                    r'answer\s*[:\-=]\s*(\d+)',
                    r'result\s*[:\-=]\s*(\d+)',
                ]

                for pattern in patterns:
                    match = re.search(pattern, scraped_text, re.IGNORECASE)
                    if match:
                        answer = match.group(1)
                        logger.info(f"Extracted code: {answer}")
                        return answer

                # Try to find any prominent number
                numbers = re.findall(r'\b(\d{4,6})\b', scraped_text)
                if numbers:
                    answer = numbers[0]
                    logger.info(f"Found number as fallback: {answer}")
                    return answer

            except Exception as e:
                logger.error(f"Fallback scrape failed for {scrape_url}: {e}")
                continue

        return None


# ---------------------------------------------------------------------------
# Module Exports
# ---------------------------------------------------------------------------

agent = QuizSolver()
QuizAgent = QuizSolver  # Alias for backwards compatibility
