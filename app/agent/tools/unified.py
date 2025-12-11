"""
Unified Advanced Tools - Consolidated functionality following OOP principles.

This module combines and consolidates tools from:
- web.py (HTML parsing, links, tables, forms)
- data.py (JSON, XML, CSV, Excel)
- files.py (PDF, archives, binary analysis)
- video.py (video/audio metadata)
- analysis.py + crypto.py (merged into analyze tool)

Design Principles:
- Single Responsibility: Each tool does one thing well
- DRY: Common patterns extracted to helper classes
- Composable: Tools can be chained via execute_python
"""
import csv
import io
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Final, List, Optional, Tuple, Union
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.sandbox import sandbox


# =============================================================================
# Constants
# =============================================================================
HTTP_TIMEOUT: Final[int] = 30
BINARY_TIMEOUT: Final[int] = 60
MAX_TABLE_ROWS: Final[int] = 20
MAX_TABLES: Final[int] = 3
MAX_LINKS: Final[int] = 50
MAX_TEXT_LENGTH: Final[int] = 2000
MAX_CSV_ROWS: Final[int] = 50
MAX_XML_ELEMENTS: Final[int] = 50


# =============================================================================
# Helper Classes (OOP Foundation)
# =============================================================================

class ContentFetcher:
    """Handles URL fetching with caching and error handling."""
    
    _cache: ClassVar[Dict[str, Tuple[str, str]]] = {}
    
    @classmethod
    async def fetch(cls, url: str, base_url: str = "") -> tuple[str, str]:
        """Fetch content from URL, returns (content, content_type)."""
        if not url.startswith('http'):
            url = urljoin(base_url, url)
        
        if url in cls._cache:
            return cls._cache[url]
        
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get('content-type', '')
            cls._cache[url] = (response.text, content_type)
            return response.text, content_type
    
    @classmethod
    async def fetch_binary(cls, url: str, base_url: str = "") -> bytes:
        """Fetch binary content."""
        if not url.startswith('http'):
            url = urljoin(base_url, url)
        
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content


@dataclass
class ParseResult:
    """Standardized result from parsing operations."""
    success: bool
    data: Any
    error: str = ""
    
    def __str__(self) -> str:
        if not self.success:
            return f"Error: {self.error}"
        if isinstance(self.data, (dict, list)):
            return json.dumps(self.data, indent=2)
        return str(self.data)


# =============================================================================
# WEB TOOLS (Consolidated from web.py)
# =============================================================================

@quiz_agent.tool
async def parse_webpage(
    ctx: RunContext[QuizDependencies],
    url: str,
    extract: str = "all"
) -> str:
    """
    Parse webpage and extract various elements.
    
    Args:
        url: URL to parse
        extract: What to extract - "links", "tables", "forms", "text", "meta", "all"
    
    Returns:
        Extracted content based on type requested
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)
    
    logger.info(f"Parsing webpage: {url}, extract: {extract}")
    
    try:
        content, _ = await ContentFetcher.fetch(url)
        soup = BeautifulSoup(content, 'html.parser')
        results = []
        
        if extract in ("links", "all"):
            links = []
            for a in soup.find_all('a', href=True):
                href = urljoin(url, a['href'])
                text = a.get_text(strip=True)[:50]
                links.append(f"{href} [{text}]" if text else href)
            if links:
                results.append(f"=== LINKS ({len(links)}) ===\n" + '\n'.join(links[:50]))
        
        if extract in ("tables", "all"):
            tables = soup.find_all('table')
            for i, table in enumerate(tables[:3]):  # Max 3 tables
                rows = []
                for tr in table.find_all('tr')[:20]:  # Max 20 rows
                    cells = [td.get_text(strip=True)[:50] for td in tr.find_all(['td', 'th'])]
                    rows.append(' | '.join(cells))
                if rows:
                    results.append(f"=== TABLE {i} ===\n" + '\n'.join(rows))
        
        if extract in ("forms", "all"):
            forms = soup.find_all('form')
            for i, form in enumerate(forms):
                action = form.get('action', '')
                method = form.get('method', 'GET')
                inputs = []
                for inp in form.find_all(['input', 'select', 'textarea']):
                    name = inp.get('name', '')
                    itype = inp.get('type', 'text')
                    if name:
                        inputs.append(f"  {name} ({itype})")
                if inputs:
                    results.append(f"=== FORM {i}: {method} {action} ===\n" + '\n'.join(inputs))
        
        if extract in ("text", "all"):
            # Remove script/style
            for tag in soup(['script', 'style']):
                tag.decompose()
            text = soup.get_text(separator=' ', strip=True)[:2000]
            results.append(f"=== TEXT ===\n{text}")
        
        if extract in ("meta", "all"):
            meta = []
            title = soup.find('title')
            if title:
                meta.append(f"Title: {title.string}")
            for m in soup.find_all('meta'):
                name = m.get('name') or m.get('property', '')
                content = m.get('content', '')
                if name and content:
                    meta.append(f"{name}: {content[:100]}")
            if meta:
                results.append(f"=== META ===\n" + '\n'.join(meta[:10]))
        
        return '\n\n'.join(results) if results else "No content extracted"
        
    except Exception as e:
        return f"Parse error: {e}"


# =============================================================================
# DATA TOOLS (Consolidated from data.py)
# =============================================================================

@quiz_agent.tool
async def process_data(
    ctx: RunContext[QuizDependencies],
    url: str,
    operation: str,
    params: str = ""
) -> str:
    """
    Process JSON, XML, or CSV data with various operations.
    
    Args:
        url: URL of data file
        operation: Operation to perform:
            - "query" + params=".path.to.value" for JSON path query
            - "filter" + params="column==value" for CSV filtering
            - "sum/count/mean/max/min" + params="column" for aggregation
            - "xpath" + params="//element" for XML
            - "pivot" + params="index,column,value" for pivot table
        params: Operation-specific parameters
    
    Returns:
        Processed data result
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)
    
    logger.info(f"Processing data: {url}, operation: {operation}")
    
    try:
        content, content_type = await ContentFetcher.fetch(url)
        
        # Detect data type
        is_json = 'json' in content_type or content.strip().startswith(('{', '['))
        is_xml = 'xml' in content_type or content.strip().startswith('<?xml')
        is_csv = 'csv' in content_type or (not is_json and not is_xml)
        
        if is_json:
            data = json.loads(content)
            
            if operation == "query":
                # JQ-like path query
                parts = re.findall(r'\.(\w+)|\[(\d+)\]|\[\*\]', params)
                result = data
                for key, index, wildcard in parts:
                    if key and isinstance(result, dict):
                        result = result.get(key)
                    elif index and isinstance(result, list):
                        result = result[int(index)]
                    elif isinstance(result, list):
                        result = [item.get(key) if key and isinstance(item, dict) else item for item in result]
                return json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
            
            elif operation in ("sum", "count", "mean", "max", "min"):
                # Aggregate operation on array
                if isinstance(data, list):
                    if params:
                        values = [item.get(params) for item in data if isinstance(item, dict)]
                        values = [v for v in values if isinstance(v, (int, float))]
                    else:
                        values = [v for v in data if isinstance(v, (int, float))]
                    
                    if operation == "sum":
                        return str(sum(values))
                    elif operation == "count":
                        return str(len(values))
                    elif operation == "mean":
                        return str(sum(values) / len(values)) if values else "0"
                    elif operation == "max":
                        return str(max(values)) if values else "N/A"
                    elif operation == "min":
                        return str(min(values)) if values else "N/A"
            
            return json.dumps(data, indent=2)
        
        elif is_csv:
            lines = content.strip().split('\n')
            reader = csv.DictReader(io.StringIO(content))
            rows = list(reader)
            
            if not rows:
                return "Empty CSV"
            
            if operation == "filter" and params:
                # Filter: column==value or column>value
                match = re.match(r'(\w+)\s*(==|!=|>|<|>=|<=)\s*(.+)', params)
                if match:
                    col, op, val = match.groups()
                    filtered = []
                    for row in rows:
                        cell = row.get(col, '')
                        try:
                            cell_num = float(cell) if cell else 0
                            val_num = float(val)
                            if op == '==' and cell == val: filtered.append(row)
                            elif op == '!=' and cell != val: filtered.append(row)
                            elif op == '>' and cell_num > val_num: filtered.append(row)
                            elif op == '<' and cell_num < val_num: filtered.append(row)
                            elif op == '>=' and cell_num >= val_num: filtered.append(row)
                            elif op == '<=' and cell_num <= val_num: filtered.append(row)
                        except:
                            if op == '==' and cell == val: filtered.append(row)
                            elif op == '!=' and cell != val: filtered.append(row)
                    rows = filtered
            
            elif operation in ("sum", "count", "mean", "max", "min") and params:
                values = []
                for row in rows:
                    try:
                        values.append(float(row.get(params, 0)))
                    except:
                        pass
                
                if operation == "sum":
                    return str(sum(values))
                elif operation == "count":
                    return str(len(values))
                elif operation == "mean":
                    return str(sum(values) / len(values)) if values else "0"
                elif operation == "max":
                    return str(max(values)) if values else "N/A"
                elif operation == "min":
                    return str(min(values)) if values else "N/A"
            
            # Return CSV preview
            output = io.StringIO()
            if rows:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows[:50])  # Max 50 rows
            return f"Columns: {list(rows[0].keys()) if rows else []}\nRows: {len(rows)}\n\n{output.getvalue()}"
        
        elif is_xml:
            from xml.etree import ElementTree as ET
            root = ET.fromstring(content)
            
            if operation == "xpath" and params:
                elements = root.findall(params)
                results = []
                for el in elements[:50]:
                    if el.text:
                        results.append(el.text.strip())
                    else:
                        results.append(ET.tostring(el, encoding='unicode')[:200])
                return '\n'.join(results) if results else "No matches"
            
            return ET.tostring(root, encoding='unicode')[:3000]
        
        return content[:2000]
        
    except Exception as e:
        return f"Data processing error: {e}"


# =============================================================================
# FILE TOOLS (Consolidated from files.py)
# =============================================================================

@quiz_agent.tool
async def process_document(
    ctx: RunContext[QuizDependencies],
    url: str,
    operation: str = "text"
) -> str:
    """
    Process PDF and document files.
    
    Args:
        url: URL of document file
        operation: "text" (extract text), "tables" (extract tables), "info" (metadata)
    
    Returns:
        Extracted content
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)
    
    logger.info(f"Processing document: {url}, operation: {operation}")
    
    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')
        
        if operation == "text":
            code = f'''
import pdfplumber
try:
    with pdfplumber.open("{safe_path}") as pdf:
        print(f"Pages: {{len(pdf.pages)}}")
        for i, page in enumerate(pdf.pages[:10]):
            text = page.extract_text()
            if text:
                print(f"\\n=== Page {{i+1}} ===")
                print(text[:2000])
except Exception as e:
    print(f"Error: {{e}}")
'''
        elif operation == "tables":
            code = f'''
import pdfplumber
try:
    with pdfplumber.open("{safe_path}") as pdf:
        for i, page in enumerate(pdf.pages[:5]):
            tables = page.extract_tables()
            for j, table in enumerate(tables[:3]):
                print(f"\\n=== Page {{i+1}} Table {{j+1}} ===")
                for row in table[:20]:
                    print(" | ".join(str(c or "") for c in row))
except Exception as e:
    print(f"Error: {{e}}")
'''
        else:  # info
            code = f'''
import pdfplumber
try:
    with pdfplumber.open("{safe_path}") as pdf:
        print(f"Pages: {{len(pdf.pages)}}")
        if pdf.metadata:
            for k, v in pdf.metadata.items():
                print(f"{{k}}: {{v}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        
        result = await sandbox.execute_code(code, timeout=120)
        return result.stdout.strip() if result.success else f"Error: {result.stderr}"
        
    except Exception as e:
        return f"Document processing error: {e}"


@quiz_agent.tool
async def process_archive(
    ctx: RunContext[QuizDependencies],
    url: str,
    operation: str = "list",
    filename: str = ""
) -> str:
    """
    Process archive files (zip, tar, etc).
    
    Args:
        url: URL of archive file
        operation: "list" (list contents), "extract" (extract specific file)
        filename: File to extract (for extract operation)
    
    Returns:
        Archive contents or extracted file content
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)
    
    logger.info(f"Processing archive: {url}, operation: {operation}")
    
    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')
        
        if operation == "list":
            code = f'''
import zipfile
import tarfile
import os

path = "{safe_path}"
if zipfile.is_zipfile(path):
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist()[:100]:
            print(f"{{info.filename}} ({{info.file_size}} bytes)")
elif tarfile.is_tarfile(path):
    with tarfile.open(path) as tf:
        for member in tf.getmembers()[:100]:
            print(f"{{member.name}} ({{member.size}} bytes)")
else:
    print("Unknown archive format")
'''
        elif operation == "extract" and filename:
            code = f'''
import zipfile
import tarfile

path = "{safe_path}"
target = "{filename}"
if zipfile.is_zipfile(path):
    with zipfile.ZipFile(path) as zf:
        try:
            content = zf.read(target)
            try:
                print(content.decode('utf-8'))
            except:
                print(f"Binary file: {{len(content)}} bytes")
                print(content[:500].hex())
        except KeyError:
            print(f"File not found: {{target}}")
elif tarfile.is_tarfile(path):
    with tarfile.open(path) as tf:
        try:
            member = tf.getmember(target)
            f = tf.extractfile(member)
            if f:
                content = f.read()
                try:
                    print(content.decode('utf-8'))
                except:
                    print(f"Binary: {{len(content)}} bytes")
        except KeyError:
            print(f"File not found: {{target}}")
'''
        else:
            return "Specify operation: list or extract with filename"
        
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Error: {result.stderr}"
        
    except Exception as e:
        return f"Archive processing error: {e}"


# =============================================================================
# MEDIA TOOLS (Consolidated from video.py)
# =============================================================================

@quiz_agent.tool
async def analyze_media(
    ctx: RunContext[QuizDependencies],
    url: str,
    operation: str = "info"
) -> str:
    """
    Analyze video/audio files for metadata.
    
    Args:
        url: URL of media file
        operation: "info" (metadata), "duration", "frames" (for video)
    
    Returns:
        Media information
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)
    
    logger.info(f"Analyzing media: {url}, operation: {operation}")
    
    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')
        
        code = f'''
import subprocess
import json

try:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", 
         "-show_format", "-show_streams", "{safe_path}"],
        capture_output=True, text=True, timeout=30
    )
    
    if result.returncode != 0:
        print(f"ffprobe error: {{result.stderr}}")
    else:
        data = json.loads(result.stdout)
        fmt = data.get("format", {{}})
        
        print(f"Format: {{fmt.get('format_name', 'unknown')}}")
        print(f"Duration: {{float(fmt.get('duration', 0)):.2f}} seconds")
        print(f"Size: {{int(fmt.get('size', 0)) / 1024:.1f}} KB")
        
        for stream in data.get("streams", []):
            stype = stream.get("codec_type")
            if stype == "video":
                print(f"\\nVideo: {{stream.get('codec_name')}} {{stream.get('width')}}x{{stream.get('height')}}")
            elif stype == "audio":
                print(f"\\nAudio: {{stream.get('codec_name')}} {{stream.get('sample_rate')}}Hz")
except FileNotFoundError:
    print("ffprobe not found - install ffmpeg")
except Exception as e:
    print(f"Error: {{e}}")
'''
        
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Error: {result.stderr}"
        
    except Exception as e:
        return f"Media analysis error: {e}"


# =============================================================================
# ANALYSIS TOOLS (Consolidated from analysis.py + crypto.py)  
# =============================================================================

@quiz_agent.tool
def analyze_text(
    ctx: RunContext[QuizDependencies],
    text: str,
    operation: str
) -> str:
    """
    Analyze text with various operations (patterns, encoding, crypto).
    
    Args:
        text: Text to analyze
        operation: Operation to perform:
            - "pattern": Detect number/string patterns
            - "frequency": Character/word frequency
            - "encoding": Detect and try decoding (base64, hex, etc)
            - "rot13": Apply ROT13
            - "reverse": Reverse text
            - "stats": Text statistics
    
    Returns:
        Analysis result
    """
    logger.info(f"Analyzing text, operation: {operation}, length: {len(text)}")
    
    try:
        if operation == "pattern":
            # Try to detect patterns in numbers
            numbers = re.findall(r'-?\d+\.?\d*', text)
            if numbers:
                nums = [float(n) if '.' in n else int(n) for n in numbers]
                if len(nums) >= 3:
                    # Check arithmetic progression
                    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
                    if len(set(diffs)) == 1:
                        return f"Arithmetic progression with difference {diffs[0]}, next: {nums[-1] + diffs[0]}"
                    
                    # Check geometric progression
                    if all(nums[i] != 0 for i in range(len(nums)-1)):
                        ratios = [nums[i+1] / nums[i] for i in range(len(nums)-1)]
                        if len(set(round(r, 6) for r in ratios)) == 1:
                            return f"Geometric progression with ratio {ratios[0]}, next: {nums[-1] * ratios[0]}"
                    
                    # Check Fibonacci-like
                    if all(nums[i] + nums[i+1] == nums[i+2] for i in range(len(nums)-2)):
                        return f"Fibonacci-like sequence, next: {nums[-2] + nums[-1]}"
                
                return f"Numbers found: {nums}"
            return "No clear pattern detected"
        
        elif operation == "frequency":
            from collections import Counter
            char_freq = Counter(text.lower())
            word_freq = Counter(re.findall(r'\w+', text.lower()))
            
            result = "Character frequency (top 10):\n"
            for char, count in char_freq.most_common(10):
                result += f"  '{char}': {count}\n"
            
            result += "\nWord frequency (top 10):\n"
            for word, count in word_freq.most_common(10):
                result += f"  '{word}': {count}\n"
            
            return result
        
        elif operation == "encoding":
            import base64
            results = []
            
            # Try base64
            try:
                decoded = base64.b64decode(text + '==').decode('utf-8')
                if decoded.isprintable():
                    results.append(f"Base64: {decoded}")
            except:
                pass
            
            # Try hex
            try:
                decoded = bytes.fromhex(text).decode('utf-8')
                if decoded.isprintable():
                    results.append(f"Hex: {decoded}")
            except:
                pass
            
            # Try URL decode
            from urllib.parse import unquote
            decoded = unquote(text)
            if decoded != text:
                results.append(f"URL: {decoded}")
            
            return '\n'.join(results) if results else "No encoding detected"
        
        elif operation == "rot13":
            result = []
            for char in text:
                if char.isalpha():
                    base = ord('A') if char.isupper() else ord('a')
                    result.append(chr((ord(char) - base + 13) % 26 + base))
                else:
                    result.append(char)
            return ''.join(result)
        
        elif operation == "reverse":
            return text[::-1]
        
        elif operation == "stats":
            words = text.split()
            lines = text.split('\n')
            return f"Characters: {len(text)}\nWords: {len(words)}\nLines: {len(lines)}\nUnique words: {len(set(words))}"
        
        return f"Unknown operation: {operation}"
        
    except Exception as e:
        return f"Analysis error: {e}"


@quiz_agent.tool
def compute_math(
    ctx: RunContext[QuizDependencies],
    expression: str,
    operation: str = "eval"
) -> str:
    """
    Compute mathematical operations.
    
    Args:
        expression: Mathematical expression or number
        operation: 
            - "eval": Evaluate expression (default)
            - "factor": Prime factorization
            - "gcd": GCD of comma-separated numbers
            - "lcm": LCM of comma-separated numbers
            - "base": Convert number base (e.g., "255,10,16" = 255 from base 10 to 16)
    
    Returns:
        Computation result
    """
    logger.info(f"Computing: {expression}, operation: {operation}")
    
    try:
        if operation == "eval":
            # Safe eval for math expressions
            import math
            allowed = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len, 'pow': pow, 'int': int, 'float': float,
                'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'log10': math.log10, 'exp': math.exp,
                'pi': math.pi, 'e': math.e, 'floor': math.floor, 'ceil': math.ceil
            }
            result = eval(expression, {"__builtins__": {}}, allowed)
            return str(result)
        
        elif operation == "factor":
            n = int(expression)
            factors = []
            d = 2
            while d * d <= n:
                while n % d == 0:
                    factors.append(d)
                    n //= d
                d += 1
            if n > 1:
                factors.append(n)
            return f"Factors: {factors}"
        
        elif operation == "gcd":
            import math
            numbers = [int(x.strip()) for x in expression.split(',')]
            result = numbers[0]
            for n in numbers[1:]:
                result = math.gcd(result, n)
            return str(result)
        
        elif operation == "lcm":
            import math
            numbers = [int(x.strip()) for x in expression.split(',')]
            result = numbers[0]
            for n in numbers[1:]:
                result = result * n // math.gcd(result, n)
            return str(result)
        
        elif operation == "base":
            parts = expression.split(',')
            if len(parts) == 3:
                num, from_base, to_base = parts
                decimal = int(num.strip(), int(from_base))
                to_base = int(to_base)
                if to_base == 10:
                    return str(decimal)
                elif to_base == 2:
                    return bin(decimal)[2:]
                elif to_base == 8:
                    return oct(decimal)[2:]
                elif to_base == 16:
                    return hex(decimal)[2:].upper()
            return "Format: number,from_base,to_base"
        
        return f"Unknown operation: {operation}"
        
    except Exception as e:
        return f"Math error: {e}"
