"""
Video and Advanced Multimedia Tools
- Video frame extraction
- Audio analysis
- Subtitle extraction
- Media metadata
"""
from urllib.parse import urljoin

from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.sandbox import sandbox


@quiz_agent.tool
async def get_video_info(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Get detailed information about a video file.

    Args:
        url: URL of video file

    Returns:
        Video metadata (duration, resolution, codec, etc.)
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Getting video info: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import subprocess
import json

result = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", "{safe_path}"],
    capture_output=True, text=True
)

if result.returncode != 0:
    print(f"Error: {{result.stderr}}")
else:
    data = json.loads(result.stdout)
    
    fmt = data.get("format", {{}})
    print(f"Format: {{fmt.get('format_name', 'unknown')}}")
    print(f"Duration: {{float(fmt.get('duration', 0)):.2f}} seconds")
    print(f"Size: {{int(fmt.get('size', 0)) / 1024 / 1024:.2f}} MB")
    print(f"Bit rate: {{int(fmt.get('bit_rate', 0)) / 1000:.0f}} kbps")
    
    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "video":
            print(f"\\nVideo Stream:")
            print(f"  Codec: {{stream.get('codec_name')}}")
            print(f"  Resolution: {{stream.get('width')}}x{{stream.get('height')}}")
            print(f"  FPS: {{eval(stream.get('r_frame_rate', '0/1')):.2f}}")
        elif codec_type == "audio":
            print(f"\\nAudio Stream:")
            print(f"  Codec: {{stream.get('codec_name')}}")
            print(f"  Sample rate: {{stream.get('sample_rate')}} Hz")
            print(f"  Channels: {{stream.get('channels')}}")
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Video info error: {result.stderr}"

    except Exception as e:
        return f"Error getting video info: {e}"


@quiz_agent.tool
async def extract_video_frames(
    ctx: RunContext[QuizDependencies],
    url: str,
    timestamps: str = "",
    count: int = 5
) -> str:
    """
    Extract frames from a video at specific timestamps or evenly spaced.

    Args:
        url: URL of video file
        timestamps: Comma-separated timestamps in seconds (e.g., "0,5,10") or empty for even spacing
        count: Number of frames to extract if timestamps not specified

    Returns:
        Paths to extracted frames
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting frames from: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import subprocess
import json
import os

video_path = "{safe_path}"
timestamps = "{timestamps}"
count = {count}

# Get duration
result = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
    capture_output=True, text=True
)
duration = float(json.loads(result.stdout)["format"]["duration"])

if timestamps:
    times = [float(t.strip()) for t in timestamps.split(",")]
else:
    times = [duration * i / (count + 1) for i in range(1, count + 1)]

output_paths = []
for i, t in enumerate(times):
    output = f"/tmp/frame_{{i}}_{{t:.2f}}.jpg"
    subprocess.run([
        "ffmpeg", "-y", "-ss", str(t), "-i", video_path,
        "-vframes", "1", "-q:v", "2", output
    ], capture_output=True)
    if os.path.exists(output):
        output_paths.append(output)
        print(f"Frame at {{t:.2f}}s: {{output}}")

print(f"\\nExtracted {{len(output_paths)}} frames")
'''
        result = await sandbox.execute_code(code, timeout=120)
        return result.stdout.strip() if result.success else f"Frame extraction error: {result.stderr}"

    except Exception as e:
        return f"Error extracting frames: {e}"


@quiz_agent.tool
async def extract_audio_from_video(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract audio track from a video file.

    Args:
        url: URL of video file

    Returns:
        Path to extracted audio file
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting audio from video: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import subprocess
import os

video_path = "{safe_path}"
audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"

result = subprocess.run([
    "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", audio_path
], capture_output=True, text=True)

if os.path.exists(audio_path):
    size = os.path.getsize(audio_path) / 1024 / 1024
    print(f"Audio extracted: {{audio_path}}")
    print(f"Size: {{size:.2f}} MB")
else:
    print(f"Error: {{result.stderr}}")
'''
        result = await sandbox.execute_code(code, timeout=120)
        return result.stdout.strip() if result.success else f"Audio extraction error: {result.stderr}"

    except Exception as e:
        return f"Error extracting audio: {e}"


@quiz_agent.tool
async def transcribe_video(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract and transcribe audio from a video file.

    Args:
        url: URL of video file

    Returns:
        Transcribed text from video audio
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Transcribing video: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import subprocess
import os
import speech_recognition as sr
from pydub import AudioSegment

video_path = "{safe_path}"
audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"

# Extract audio
subprocess.run([
    "ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path
], capture_output=True)

if not os.path.exists(audio_path):
    print("Error: Could not extract audio")
else:
    # Transcribe
    recognizer = sr.Recognizer()
    
    audio = AudioSegment.from_wav(audio_path)
    duration = len(audio) / 1000  # seconds
    
    # Split into chunks for long audio
    chunk_length = 30000  # 30 seconds
    chunks = [audio[i:i+chunk_length] for i in range(0, len(audio), chunk_length)]
    
    full_text = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"/tmp/chunk_{{i}}.wav"
        chunk.export(chunk_path, format="wav")
        
        with sr.AudioFile(chunk_path) as source:
            audio_data = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio_data)
            full_text.append(text)
            print(f"Chunk {{i+1}}/{{len(chunks)}}: {{text[:100]}}...")
        except sr.UnknownValueError:
            print(f"Chunk {{i+1}}: (inaudible)")
        except Exception as e:
            print(f"Chunk {{i+1}} error: {{e}}")
        
        os.remove(chunk_path)
    
    print("\\n=== Full Transcription ===")
    print(" ".join(full_text))
'''
        result = await sandbox.execute_code(code, timeout=300)
        return result.stdout.strip() if result.success else f"Video transcription error: {result.stderr}"

    except Exception as e:
        return f"Error transcribing video: {e}"


@quiz_agent.tool
async def analyze_audio(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Analyze audio file properties (duration, frequency, amplitude).

    Args:
        url: URL of audio file

    Returns:
        Audio analysis results
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Analyzing audio: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import subprocess
import json
from pydub import AudioSegment
import numpy as np

path = "{safe_path}"

# Get metadata with ffprobe
result = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", path],
    capture_output=True, text=True
)

if result.returncode == 0:
    data = json.loads(result.stdout)
    fmt = data.get("format", {{}})
    
    print("=== Audio Metadata ===")
    print(f"Format: {{fmt.get('format_name', 'unknown')}}")
    print(f"Duration: {{float(fmt.get('duration', 0)):.2f}} seconds")
    print(f"Size: {{int(fmt.get('size', 0)) / 1024:.2f}} KB")
    print(f"Bit rate: {{int(fmt.get('bit_rate', 0)) / 1000:.0f}} kbps")
    
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            print(f"\\nCodec: {{stream.get('codec_name')}}")
            print(f"Sample rate: {{stream.get('sample_rate')}} Hz")
            print(f"Channels: {{stream.get('channels')}}")
            print(f"Bits per sample: {{stream.get('bits_per_sample', 'N/A')}}")

# Load and analyze with pydub
try:
    audio = AudioSegment.from_file(path)
    samples = np.array(audio.get_array_of_samples())
    
    print("\\n=== Amplitude Analysis ===")
    print(f"Max amplitude: {{np.max(np.abs(samples))}}")
    print(f"Mean amplitude: {{np.mean(np.abs(samples)):.2f}}")
    print(f"RMS: {{np.sqrt(np.mean(samples**2)):.2f}}")
    
    # Detect silence
    silence_threshold = np.max(np.abs(samples)) * 0.01
    silent_samples = np.sum(np.abs(samples) < silence_threshold)
    silence_pct = (silent_samples / len(samples)) * 100
    print(f"\\nSilence: {{silence_pct:.1f}}% of audio")
    
except Exception as e:
    print(f"Detailed analysis error: {{e}}")
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Audio analysis error: {result.stderr}"

    except Exception as e:
        return f"Error analyzing audio: {e}"


@quiz_agent.tool
async def extract_subtitles(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract subtitles/captions from a video file if available.

    Args:
        url: URL of video file

    Returns:
        Extracted subtitle text or error if no subtitles
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting subtitles from: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import subprocess
import os

video_path = "{safe_path}"
srt_path = video_path.rsplit(".", 1)[0] + ".srt"

# Check for subtitle streams
probe = subprocess.run(
    ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
    capture_output=True, text=True
)

import json
data = json.loads(probe.stdout)
subtitle_streams = [s for s in data.get("streams", []) if s.get("codec_type") == "subtitle"]

if not subtitle_streams:
    print("No embedded subtitles found in video")
else:
    print(f"Found {{len(subtitle_streams)}} subtitle stream(s)")
    
    # Extract first subtitle stream
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path, "-map", "0:s:0", srt_path
    ], capture_output=True, text=True)
    
    if os.path.exists(srt_path):
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        print("\\n=== Subtitles ===")
        print(content[:5000])
        if len(content) > 5000:
            print("... (truncated)")
    else:
        print(f"Could not extract subtitles: {{result.stderr}}")
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Subtitle extraction error: {result.stderr}"

    except Exception as e:
        return f"Error extracting subtitles: {e}"


@quiz_agent.tool
async def convert_media(
    ctx: RunContext[QuizDependencies],
    url: str,
    output_format: str
) -> str:
    """
    Convert media file to a different format.

    Args:
        url: URL of media file
        output_format: Target format (mp3, wav, mp4, avi, gif, etc.)

    Returns:
        Path to converted file
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Converting {url} to {output_format}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import subprocess
import os

input_path = "{safe_path}"
output_format = "{output_format}"
output_path = input_path.rsplit(".", 1)[0] + f".{{output_format}}"

# Build ffmpeg command based on format
if output_format in ["mp3", "wav", "ogg", "flac", "aac"]:
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", output_path]
elif output_format == "gif":
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vf", "fps=10,scale=320:-1", output_path]
else:
    cmd = ["ffmpeg", "-y", "-i", input_path, output_path]

result = subprocess.run(cmd, capture_output=True, text=True)

if os.path.exists(output_path):
    size = os.path.getsize(output_path) / 1024
    print(f"Converted: {{output_path}}")
    print(f"Size: {{size:.2f}} KB")
else:
    print(f"Conversion failed: {{result.stderr}}")
'''
        result = await sandbox.execute_code(code, timeout=120)
        return result.stdout.strip() if result.success else f"Conversion error: {result.stderr}"

    except Exception as e:
        return f"Error converting media: {e}"


@quiz_agent.tool
async def get_image_exif(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract EXIF metadata from an image.

    Args:
        url: URL of image file

    Returns:
        EXIF metadata (camera, date, GPS, settings, etc.)
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting EXIF from: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

img = Image.open("{safe_path}")

print(f"Format: {{img.format}}")
print(f"Size: {{img.size}}")
print(f"Mode: {{img.mode}}")

exif = img._getexif()

if exif:
    print("\\n=== EXIF Data ===")
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        
        # Handle bytes
        if isinstance(value, bytes):
            try:
                value = value.decode()
            except:
                value = f"<binary {{len(value)}} bytes>"
        
        # Limit long values
        str_val = str(value)
        if len(str_val) > 100:
            str_val = str_val[:100] + "..."
        
        print(f"{{tag}}: {{str_val}}")
        
        # Decode GPS info
        if tag == "GPSInfo":
            for gps_tag_id, gps_value in value.items():
                gps_tag = GPSTAGS.get(gps_tag_id, gps_tag_id)
                print(f"  GPS.{{gps_tag}}: {{gps_value}}")
else:
    print("\\nNo EXIF data found")
'''
        result = await sandbox.execute_code(code, timeout=30)
        return result.stdout.strip() if result.success else f"EXIF extraction error: {result.stderr}"

    except Exception as e:
        return f"Error extracting EXIF: {e}"
