"""
File Manipulation and Document Processing Tools
- PDF extraction
- Document analysis
- Binary file analysis
- File comparison
"""
import os
from urllib.parse import urljoin

from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.sandbox import sandbox


@quiz_agent.tool
async def extract_pdf_text(ctx: RunContext[QuizDependencies], url: str, page_range: str = "") -> str:
    """
    Extract text content from a PDF file.

    Args:
        url: URL of PDF file
        page_range: Page range like "1-5" or "1,3,5" (empty = all pages)

    Returns:
        Extracted text from PDF
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting PDF text: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import pdfplumber

with pdfplumber.open("{safe_path}") as pdf:
    page_range = "{page_range}"
    
    if page_range:
        if "-" in page_range:
            start, end = map(int, page_range.split("-"))
            pages = range(start-1, min(end, len(pdf.pages)))
        else:
            pages = [int(p)-1 for p in page_range.split(",")]
    else:
        pages = range(len(pdf.pages))
    
    print(f"Total pages: {{len(pdf.pages)}}")
    
    for i in pages:
        if 0 <= i < len(pdf.pages):
            page = pdf.pages[i]
            text = page.extract_text()
            print(f"\\n=== Page {{i+1}} ===")
            print(text if text else "(No text extracted)")
'''
        result = await sandbox.execute_code(code, timeout=120)
        return result.stdout.strip() if result.success else f"PDF error: {result.stderr}"

    except Exception as e:
        return f"Error extracting PDF: {e}"


@quiz_agent.tool
async def extract_pdf_tables(ctx: RunContext[QuizDependencies], url: str, page_num: int = 1) -> str:
    """
    Extract tables from a PDF file.

    Args:
        url: URL of PDF file
        page_num: Page number to extract tables from (1-based)

    Returns:
        Table data in CSV format
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting PDF tables: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import pdfplumber

with pdfplumber.open("{safe_path}") as pdf:
    page_num = {page_num}
    
    if page_num > len(pdf.pages):
        print(f"Page {{page_num}} not found. PDF has {{len(pdf.pages)}} pages.")
    else:
        page = pdf.pages[page_num - 1]
        tables = page.extract_tables()
        
        if not tables:
            print("No tables found on this page")
        else:
            for i, table in enumerate(tables):
                print(f"=== Table {{i+1}} ===")
                for row in table:
                    clean_row = [str(cell).replace('\\n', ' ') if cell else '' for cell in row]
                    print(','.join(clean_row))
                print()
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"PDF table error: {result.stderr}"

    except Exception as e:
        return f"Error extracting PDF tables: {e}"


@quiz_agent.tool
async def analyze_file_binary(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Analyze a binary file's structure and metadata.

    Args:
        url: URL of file to analyze

    Returns:
        File type, size, and structure information
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Analyzing binary file: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import os
import struct

path = "{safe_path}"
size = os.path.getsize(path)

print(f"File size: {{size}} bytes")

with open(path, 'rb') as f:
    header = f.read(16)
    
# Magic number detection
magic_types = {{
    b'\\x89PNG': 'PNG image',
    b'\\xff\\xd8\\xff': 'JPEG image',
    b'GIF8': 'GIF image',
    b'PK\\x03\\x04': 'ZIP archive',
    b'PK\\x05\\x06': 'ZIP archive (empty)',
    b'%PDF': 'PDF document',
    b'{{\\\\rtf': 'RTF document',
    b'\\x50\\x4b\\x03\\x04': 'Office document (DOCX/XLSX)',
    b'Rar!': 'RAR archive',
    b'\\x1f\\x8b': 'GZIP archive',
    b'BZh': 'BZIP2 archive',
    b'\\x7fELF': 'ELF executable',
    b'MZ': 'Windows executable',
    b'ID3': 'MP3 audio',
    b'ftyp': 'MP4/MOV video',
    b'RIFF': 'WAV/AVI file',
    b'OggS': 'OGG audio/video',
}}

detected = "Unknown"
for magic, filetype in magic_types.items():
    if header.startswith(magic):
        detected = filetype
        break

print(f"File type: {{detected}}")
print(f"First 32 bytes (hex): {{header[:32].hex()}}")
print(f"First 32 bytes (ascii): {{repr(header[:32])}}")
'''
        result = await sandbox.execute_code(code, timeout=30)
        return result.stdout.strip() if result.success else f"Binary analysis error: {result.stderr}"

    except Exception as e:
        return f"Error analyzing file: {e}"


@quiz_agent.tool
async def list_archive_contents(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    List contents of an archive file (ZIP, TAR, RAR, 7z).

    Args:
        url: URL of archive file

    Returns:
        List of files in archive with sizes
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Listing archive: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import zipfile
import tarfile
import os

path = "{safe_path}"

try:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            print("Archive type: ZIP")
            print(f"Files: {{len(zf.namelist())}}")
            print()
            for info in zf.infolist():
                size = info.file_size
                compressed = info.compress_size
                ratio = (1 - compressed/size) * 100 if size > 0 else 0
                print(f"{{info.filename}}: {{size}} bytes ({{ratio:.1f}}% compressed)")
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            print("Archive type: TAR")
            members = tf.getmembers()
            print(f"Files: {{len(members)}}")
            print()
            for m in members:
                print(f"{{m.name}}: {{m.size}} bytes")
    else:
        print("Unknown archive format")
        print(f"File extension: {{os.path.splitext(path)[1]}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        result = await sandbox.execute_code(code, timeout=30)
        return result.stdout.strip() if result.success else f"Archive listing error: {result.stderr}"

    except Exception as e:
        return f"Error listing archive: {e}"


@quiz_agent.tool
async def extract_file_from_archive(
    ctx: RunContext[QuizDependencies],
    archive_url: str,
    file_path: str
) -> str:
    """
    Extract and read a specific file from an archive.

    Args:
        archive_url: URL of archive file
        file_path: Path of file within archive to extract

    Returns:
        Contents of the extracted file
    """
    if not archive_url.startswith('http'):
        archive_url = urljoin(ctx.deps.current_url, archive_url)

    logger.info(f"Extracting {file_path} from {archive_url}")

    try:
        local_path = await sandbox.download_file(archive_url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import zipfile
import tarfile

path = "{safe_path}"
target = "{file_path}"

try:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            content = zf.read(target)
            try:
                print(content.decode('utf-8'))
            except:
                print(f"Binary file: {{len(content)}} bytes")
                print(f"First 100 bytes (hex): {{content[:100].hex()}}")
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            member = tf.getmember(target)
            f = tf.extractfile(member)
            content = f.read()
            try:
                print(content.decode('utf-8'))
            except:
                print(f"Binary file: {{len(content)}} bytes")
except KeyError:
    print(f"File not found in archive: {{target}}")
except Exception as e:
    print(f"Error: {{e}}")
'''
        result = await sandbox.execute_code(code, timeout=30)
        return result.stdout.strip() if result.success else f"Extraction error: {result.stderr}"

    except Exception as e:
        return f"Error extracting file: {e}"


@quiz_agent.tool
async def diff_files(ctx: RunContext[QuizDependencies], url1: str, url2: str) -> str:
    """
    Compare two text files and show differences.

    Args:
        url1: URL of first file
        url2: URL of second file

    Returns:
        Unified diff output
    """
    logger.info(f"Comparing files: {url1} vs {url2}")

    try:
        path1 = await sandbox.download_file(url1 if url1.startswith('http') else urljoin(ctx.deps.current_url, url1))
        path2 = await sandbox.download_file(url2 if url2.startswith('http') else urljoin(ctx.deps.current_url, url2))
        
        safe_path1 = path1.replace('\\', '/')
        safe_path2 = path2.replace('\\', '/')

        code = f'''
import difflib

with open("{safe_path1}") as f1:
    lines1 = f1.readlines()
with open("{safe_path2}") as f2:
    lines2 = f2.readlines()

diff = difflib.unified_diff(lines1, lines2, 
                            fromfile="file1", 
                            tofile="file2",
                            lineterm='')

for line in diff:
    print(line)
'''
        result = await sandbox.execute_code(code, timeout=30)
        output = result.stdout.strip()
        return output if output else "Files are identical" if result.success else f"Diff error: {result.stderr}"

    except Exception as e:
        return f"Error comparing files: {e}"


@quiz_agent.tool
async def get_file_stats(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Get statistics about a text file (line count, word count, character count).

    Args:
        url: URL of text file

    Returns:
        File statistics
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Getting file stats: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import os
import collections

path = "{safe_path}"
file_size = os.path.getsize(path)

with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

lines = content.split('\\n')
words = content.split()
chars = len(content)
chars_no_space = len(content.replace(' ', '').replace('\\n', '').replace('\\t', ''))

# Additional stats
unique_words = len(set(w.lower() for w in words))
avg_word_len = sum(len(w) for w in words) / len(words) if words else 0
avg_line_len = sum(len(l) for l in lines) / len(lines) if lines else 0

# Most common words
word_counts = collections.Counter(w.lower() for w in words)
common = word_counts.most_common(10)

print(f"File size: {{file_size}} bytes")
print(f"Lines: {{len(lines)}}")
print(f"Words: {{len(words)}}")
print(f"Unique words: {{unique_words}}")
print(f"Characters: {{chars}}")
print(f"Characters (no whitespace): {{chars_no_space}}")
print(f"Avg word length: {{avg_word_len:.1f}}")
print(f"Avg line length: {{avg_line_len:.1f}}")
print(f"\\nMost common words:")
for word, count in common:
    print(f"  {{word}}: {{count}}")
'''
        result = await sandbox.execute_code(code, timeout=30)
        return result.stdout.strip() if result.success else f"Stats error: {result.stderr}"

    except Exception as e:
        return f"Error getting stats: {e}"


@quiz_agent.tool
async def search_in_file(
    ctx: RunContext[QuizDependencies],
    url: str,
    pattern: str,
    context_lines: int = 2
) -> str:
    """
    Search for a pattern in a file and return matching lines with context.

    Args:
        url: URL of file to search
        pattern: Regex pattern to search for
        context_lines: Number of lines of context before/after match

    Returns:
        Matching lines with line numbers and context
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Searching for '{pattern}' in {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import re

with open("{safe_path}", 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

pattern = re.compile(r"{pattern}", re.IGNORECASE)
context = {context_lines}
matches = []

for i, line in enumerate(lines):
    if pattern.search(line):
        start = max(0, i - context)
        end = min(len(lines), i + context + 1)
        matches.append((i+1, start, end))

shown_lines = set()
for line_num, start, end in matches:
    for j in range(start, end):
        if j not in shown_lines:
            prefix = ">>>" if j == line_num - 1 else "   "
            print(f"{{prefix}} {{j+1}}: {{lines[j].rstrip()}}")
            shown_lines.add(j)
    print("---")

print(f"\\nTotal matches: {{len(matches)}}")
'''
        result = await sandbox.execute_code(code, timeout=30)
        return result.stdout.strip() if result.success else f"Search error: {result.stderr}"

    except Exception as e:
        return f"Error searching file: {e}"
