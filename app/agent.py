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

    def __post_init__(self):
        parsed = urlparse(self.current_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"


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
    start_time: float
    attempt_number: int = 0
    results: list = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    @property
    def remaining_seconds(self) -> float:
        return settings.quiz_timeout_seconds - self.elapsed_seconds

    @property
    def is_timed_out(self) -> bool:
        return self.remaining_seconds <= 0


# ---------------------------------------------------------------------------
# Agent Configuration
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Senior Data Engineer solving quiz questions. You have access to many tools.

CRITICAL RULES:
1. ONLY scrape URLs from the "Links on Page" list. DO NOT guess URLs.
2. Read the question VERY carefully - pay attention to exact wording.
3. If a question mentions a file, download and analyze it properly.

AVAILABLE TOOLS:
Core:
- scrape_url: Fetch content from a URL (handles JS-rendered pages)
- execute_python: Run Python code for complex data analysis
- download_file: Download files (CSV, JSON, Excel, PDF, etc.)
- read_file_content: Preview file content before full analysis
- get_page_links / get_page_text: Access current page data

Hash & Encoding:
- compute_hash: Compute MD5, SHA1, SHA256 hash of text
- encode_decode: Base64/URL encode/decode operations

Text Processing:
- extract_with_regex: Extract data using regex patterns
- extract_numbers: Extract all numbers from text
- count_occurrences: Count pattern occurrences in text
- parse_json: Parse and query JSON data

Math & Data:
- do_math: Perform mathematical calculations
- analyze_csv_data: Quick CSV analysis without Python code

Other:
- make_api_request: Make HTTP GET/POST requests
- get_date_info: Parse and manipulate dates
- compute_email_code: Compute code based on email hash

STRATEGY:
1. Understand the question completely
2. Identify the submission URL from links (look for 'submit')
3. Choose the right tool(s) for the task
4. Double-check your answer matches what was asked (COUNT vs SUM, etc.)
5. Return answer and complete submission URL

COMMON PATTERNS:
- "sum of values above X" -> filter > X, then SUM
- "count of items where" -> filter, then COUNT
- "hash of" -> use compute_hash tool
- "extract/find pattern" -> use extract_with_regex
- "scrape page for secret" -> scrape_url then extract data
"""

_provider = GoogleProvider(api_key=settings.google_api_key)
_model = GoogleModel("gemini-3-pro-preview", provider=_provider)

quiz_agent = Agent(
    model=_model,
    deps_type=QuizDependencies,
    output_type=QuizAnswer,
    system_prompt=SYSTEM_PROMPT,
    retries=3,
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

    logger.info(f"Scraping URL: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()

            text = soup.get_text(separator='\n', strip=True)

            # If content is too short, might be JS-rendered
            if len(text) < 50:
                logger.info(f"Content too short ({len(text)} chars), trying Selenium")
                page = await vision.extract_page_content(url)
                text = page.text_content

            logger.info(f"Scraped {len(text)} chars: {text[:300]}...")
            return text if text else "Page returned no text content"

    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        try:
            logger.info("Trying Selenium fallback")
            page = await vision.extract_page_content(url)
            return page.text_content if page.text_content else "Error: Page has no content"
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
    result = await sandbox.execute_code(code)

    if result.success:
        output = result.stdout.strip()
        logger.info(f"Code output: {output}")
        return output if output else "Code executed but produced no output"
    else:
        logger.warning(f"Code failed: {result.stderr}")
        return f"Error: {result.stderr}"


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
    Preview file content from a URL. Returns first lines to understand format.

    Args:
        url: URL of the file to read
        max_lines: Maximum number of lines to return (default 30)

    Returns:
        File preview with line count
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Reading file preview: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            content = response.text
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
        logger.error(f"Failed to read {url}: {e}")
        return f"Error reading file: {e}"


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
            return response.text

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

        while not context.is_timed_out:
            try:
                result = await self._solve_single_question(context)
                context.results.append(result)

                if result.correct and result.next_url:
                    logger.info(f"Correct! Moving to next question: {result.next_url}")
                    context.current_url = result.next_url
                    context.attempt_number = 0
                elif result.correct:
                    logger.info("Quiz completed successfully!")
                    break
                else:
                    context.attempt_number += 1
                    if context.attempt_number >= settings.max_retries_per_question:
                        logger.warning(f"Max retries reached for {context.current_url}")
                        break
                    logger.info(f"Retrying (attempt {context.attempt_number + 1})")

            except Exception as e:
                logger.error(f"Error solving question: {e}", exc_info=True)
                context.results.append(QuizResult(
                    url=context.current_url,
                    answer=None,
                    correct=False,
                    message=str(e)
                ))
                break

        if context.is_timed_out:
            logger.warning("Quiz timed out!")

        return context.results

    async def _solve_single_question(self, context: QuizContext) -> QuizResult:
        url = context.current_url
        logger.info(f"Solving question: {url}")
        page = None
        for attempt in range(3):
            try:
                page = await vision.extract_page_content(url)
                if page and page.text_content:
                    break
            except Exception as e:
                logger.warning(f"Page extraction attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)

        if not page:
            raise RuntimeError(f"Failed to extract page content from {url}")

        logger.info(f"Page text: {page.text_content[:500]}...")
        logger.info(f"Links: {page.links}")
        logger.info(f"Vision submission endpoint: {page.submission_endpoint}")

        deps = QuizDependencies(
            email=context.email,
            secret=context.secret,
            current_url=url,
            page_content=page
        )

        prompt = self._build_prompt(url, page, deps)

        try:
            result = await quiz_agent.run(prompt, deps=deps)
            agent_answer = result.output

            logger.info(f"Agent result: answer={agent_answer.answer}, submission_url={agent_answer.submission_url}")

            submission_endpoint = self._resolve_submission_url(agent_answer, page, deps)

            response = await action.submit_answer(
                endpoint=submission_endpoint,
                email=context.email,
                secret=context.secret,
                url=url,
                answer=agent_answer.answer
            )

            return QuizResult(
                url=url,
                answer=agent_answer.answer,
                correct=response.correct,
                message=response.message or response.reason,
                next_url=response.url
            )

        except Exception as e:
            logger.error(f"Agent failed: {e}", exc_info=True)
            return await self._fallback_solve(context, page, deps)

    def _build_prompt(self, url: str, page: PageContent, deps: QuizDependencies) -> str:
        links_formatted = (
            "\n".join([f"  - {link}" for link in page.links])
            if page.links else "  (no links found)"
        )

        return f"""Solve this quiz question.

=== CURRENT PAGE INFO ===
URL: {url}
Base URL: {deps.base_url}
Email: {deps.email}

=== PAGE CONTENT ===
{page.text_content}

=== AVAILABLE LINKS (ONLY use these for scraping) ===
{links_formatted}

=== INSTRUCTIONS ===
1. Read the page content above to understand what is being asked
2. Find the submission URL from the links list (look for 'submit')
3. If you need to scrape another page, use ONLY a URL from the links list above
4. DO NOT guess or make up URLs - only use what's in the links list
5. If working with a CSV file and a cutoff value:
   - Download the file and read it
   - Filter values greater than the cutoff
   - Calculate the SUM of those values (NOT the count)
6. Return your answer and the complete submission URL

IMPORTANT:
- The submission URL must be absolute like: {deps.base_url}/submit
- When a cutoff is given with data, the answer is usually the SUM of values above the cutoff
- Read the question carefully for COUNT vs SUM vs other operations
"""

    def _resolve_submission_url(
        self,
        agent_answer: QuizAnswer,
        page: PageContent,
        deps: QuizDependencies
    ) -> str:
        """Resolve the submission URL from agent answer or fallbacks."""
        submission_endpoint = agent_answer.submission_url

        if not submission_endpoint or not submission_endpoint.startswith('http'):
            if page.submission_endpoint:
                submission_endpoint = page.submission_endpoint
            else:
                # Try to find submit link in page links
                for link in page.links:
                    if 'submit' in link.lower():
                        submission_endpoint = link
                        break
                else:
                    submission_endpoint = f"{deps.base_url}/submit"
            logger.info(f"Using fallback submission endpoint: {submission_endpoint}")

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
