"""
Media and Archive Tools
- Audio transcription
- Image analysis
- Video analysis
- Archive extraction
"""
from urllib.parse import urljoin

from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.sandbox import sandbox


@quiz_agent.tool
async def transcribe_audio(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Download and transcribe an audio file using Google Speech Recognition.
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
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')
        
        code = f'''
import speech_recognition as sr
from pydub import AudioSegment
import os

path = "{safe_path}"

try:
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    
    wav_path = path.rsplit(".", 1)[0] + "_converted.wav"
    audio.export(wav_path, format="wav")
    
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
    
    text = recognizer.recognize_google(audio_data)
    print(text)
    
    if os.path.exists(wav_path):
        os.remove(wav_path)
except Exception as e:
    print(f"Error: {{e}}")
'''
        result = await sandbox.execute_code(code, timeout=120)
        
        if result.success and result.stdout.strip() and not result.stdout.strip().startswith("Error:"):
            transcription = result.stdout.strip()
            logger.info(f"Transcription: {transcription}")
            return transcription
        else:
            error_msg = result.stderr or result.stdout
            logger.warning(f"Speech recognition failed: {error_msg}")
            return f"Transcription failed: {error_msg}"
            
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

try:
    img = Image.open("{local_path}")
    width, height = img.size
    mode = img.mode
    format_type = img.format
    print(f"Image: {{width}}x{{height}}, mode={{mode}}, format={{format_type}}")
    
    if hasattr(img, 'info'):
        for k, v in img.info.items():
            if isinstance(v, (str, int, float)):
                print(f"{{k}}: {{v}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        else:
            code = f'''
from PIL import Image

try:
    img = Image.open("{local_path}")
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

try:
    with zipfile.ZipFile("{local_path}", 'r') as zf:
        print("=== ZIP Contents ===")
        for name in zf.namelist():
            info = zf.getinfo(name)
            print(f"{{name}} ({{info.file_size}} bytes)")
        
        print("\\n=== File Contents ===")
        for name in zf.namelist():
            if not name.endswith('/'):
                try:
                    content = zf.read(name)
                    try:
                        text = content.decode('utf-8')
                        print(f"\\n--- {{name}} ---")
                        print(text[:2000])
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
            code = f'''
import subprocess
import os
import speech_recognition as sr

try:
    audio_path = "{local_path}.wav"
    result = subprocess.run([
        'ffmpeg', '-i', "{local_path}", '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        audio_path, '-y'
    ], capture_output=True, text=True)
    
    if result.returncode == 0 and os.path.exists(audio_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print(text.lower())
        os.remove(audio_path)
    else:
        print(f"Audio extraction failed: {{result.stderr}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        else:
            code = f'''
import subprocess
import os

try:
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
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(extract_dir)
            print("Extracted ZIP archive")
    elif tarfile.is_tarfile(filepath):
        with tarfile.open(filepath, 'r:*') as tf:
            tf.extractall(extract_dir)
            print("Extracted TAR archive")
    else:
        print("Unknown archive format")
        
    print("\\n=== Contents ===")
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, extract_dir)
            size = os.path.getsize(full_path)
            print(f"{{rel_path}} ({{size}} bytes)")
            
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
