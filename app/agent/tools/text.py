"""
Text Processing and Encoding Tools
"""
import base64
import hashlib
import json
import re
from urllib.parse import quote, unquote

from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent


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
