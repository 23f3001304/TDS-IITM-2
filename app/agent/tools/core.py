"""
Core Agent Tools
- URL scraping
- Python execution
- File operations
"""
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.vision import vision
from app.sandbox import sandbox


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
            
            if 'json' in content_type or url.endswith('.json') or raw_text.strip().startswith(('{', '[')):
                logger.info(f"JSON response ({len(raw_text)} chars): {raw_text[:200]}...")
                ctx.deps.url_cache[url] = raw_text
                return raw_text

            soup = BeautifulSoup(raw_text, 'html.parser')
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()

            text = soup.get_text(separator='\n', strip=True)

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
    The code MUST print the final answer.

    Args:
        code: Python code to execute

    Returns:
        The stdout output from the code
    """
    logger.info(f"Executing Python code:\n{code}")
    result = await sandbox.execute_code(code, timeout=120)

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
        packages: Space-separated list of packages to install

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
    
    result = await sandbox.execute_code(code, timeout=300)
    
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
    Preview file content from a URL or local path.

    Args:
        url: URL of the file to read OR local file path
        max_lines: Maximum number of lines to return (default 30)

    Returns:
        File preview with line count
    """
    from pathlib import Path
    
    if not url.startswith('http'):
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


@quiz_agent.tool
def get_page_links(ctx: RunContext[QuizDependencies]) -> list[str]:
    """Get all links from the current quiz page."""
    return ctx.deps.page_content.links


@quiz_agent.tool
def get_page_text(ctx: RunContext[QuizDependencies]) -> str:
    """Get the text content of the current quiz page."""
    return ctx.deps.page_content.text_content
