"""
Encoding, Cryptography, and Security Tools
- Various encoding schemes
- Hash functions
- Cryptographic operations
- Security analysis
"""
import base64
import binascii
import hashlib
import hmac
import json
import re
import struct
from urllib.parse import quote, unquote

from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.sandbox import sandbox


@quiz_agent.tool
def decode_base64_variants(ctx: RunContext[QuizDependencies], data: str, variant: str = "standard") -> str:
    """
    Decode various Base64 variants.

    Args:
        data: Base64 encoded string
        variant: standard, urlsafe, base32, base16, ascii85

    Returns:
        Decoded string
    """
    try:
        if variant == "standard":
            # Add padding if needed
            padding = 4 - (len(data) % 4)
            if padding != 4:
                data += '=' * padding
            return base64.b64decode(data).decode('utf-8', errors='replace')
        elif variant == "urlsafe":
            padding = 4 - (len(data) % 4)
            if padding != 4:
                data += '=' * padding
            return base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
        elif variant == "base32":
            padding = 8 - (len(data) % 8)
            if padding != 8:
                data += '=' * padding
            return base64.b32decode(data).decode('utf-8', errors='replace')
        elif variant == "base16":
            return base64.b16decode(data.upper()).decode('utf-8', errors='replace')
        elif variant == "ascii85":
            return base64.a85decode(data).decode('utf-8', errors='replace')
        else:
            return f"Unknown variant: {variant}"
    except Exception as e:
        return f"Decode error: {e}"


@quiz_agent.tool
def encode_base64_variants(ctx: RunContext[QuizDependencies], data: str, variant: str = "standard") -> str:
    """
    Encode to various Base64 variants.

    Args:
        data: String to encode
        variant: standard, urlsafe, base32, base16, ascii85

    Returns:
        Encoded string
    """
    try:
        data_bytes = data.encode('utf-8')
        if variant == "standard":
            return base64.b64encode(data_bytes).decode()
        elif variant == "urlsafe":
            return base64.urlsafe_b64encode(data_bytes).decode()
        elif variant == "base32":
            return base64.b32encode(data_bytes).decode()
        elif variant == "base16":
            return base64.b16encode(data_bytes).decode()
        elif variant == "ascii85":
            return base64.a85encode(data_bytes).decode()
        else:
            return f"Unknown variant: {variant}"
    except Exception as e:
        return f"Encode error: {e}"


@quiz_agent.tool
def rot_cipher(ctx: RunContext[QuizDependencies], text: str, shift: int = 13) -> str:
    """
    Apply ROT cipher (Caesar cipher) to text.

    Args:
        text: Text to encode/decode
        shift: Number of positions to shift (default 13 for ROT13)

    Returns:
        Transformed text
    """
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            shifted = (ord(char) - base + shift) % 26 + base
            result.append(chr(shifted))
        else:
            result.append(char)
    return ''.join(result)


@quiz_agent.tool
def xor_bytes(ctx: RunContext[QuizDependencies], data: str, key: str, data_format: str = "hex") -> str:
    """
    XOR data with a key.

    Args:
        data: Data to XOR (hex string or plain text based on data_format)
        key: Key to XOR with (hex or text)
        data_format: "hex" or "text" for input data format

    Returns:
        XOR result as hex string
    """
    try:
        if data_format == "hex":
            data_bytes = bytes.fromhex(data)
        else:
            data_bytes = data.encode()

        # Try key as hex first, then as text
        try:
            key_bytes = bytes.fromhex(key)
        except ValueError:
            key_bytes = key.encode()

        result = bytes(d ^ key_bytes[i % len(key_bytes)] for i, d in enumerate(data_bytes))
        
        # Return both hex and attempted text decode
        hex_result = result.hex()
        try:
            text_result = result.decode('utf-8')
            return f"Hex: {hex_result}\nText: {text_result}"
        except:
            return f"Hex: {hex_result}"

    except Exception as e:
        return f"XOR error: {e}"


@quiz_agent.tool
def compute_hmac(ctx: RunContext[QuizDependencies], message: str, key: str, algorithm: str = "sha256") -> str:
    """
    Compute HMAC of a message.

    Args:
        message: Message to authenticate
        key: Secret key
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)

    Returns:
        HMAC as hex string
    """
    try:
        algo_map = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512,
        }
        
        algo = algo_map.get(algorithm.lower())
        if not algo:
            return f"Unknown algorithm: {algorithm}"

        h = hmac.new(key.encode(), message.encode(), algo)
        return h.hexdigest()

    except Exception as e:
        return f"HMAC error: {e}"


@quiz_agent.tool
def hash_file(ctx: RunContext[QuizDependencies], hex_data: str, algorithm: str = "sha256") -> str:
    """
    Compute hash of binary data (provided as hex).

    Args:
        hex_data: Data as hex string
        algorithm: Hash algorithm (md5, sha1, sha256, sha512)

    Returns:
        Hash as hex string
    """
    try:
        data = bytes.fromhex(hex_data)
        
        algorithms = {
            'md5': hashlib.md5,
            'sha1': hashlib.sha1,
            'sha256': hashlib.sha256,
            'sha512': hashlib.sha512,
        }

        algo = algorithms.get(algorithm.lower())
        if not algo:
            return f"Unknown algorithm: {algorithm}"

        return algo(data).hexdigest()

    except Exception as e:
        return f"Hash error: {e}"


@quiz_agent.tool
def morse_code(ctx: RunContext[QuizDependencies], text: str, operation: str = "encode") -> str:
    """
    Encode or decode Morse code.

    Args:
        text: Text to encode or Morse code to decode (use . and - for dots/dashes)
        operation: "encode" or "decode"

    Returns:
        Encoded/decoded result
    """
    morse_dict = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.',
        'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..',
        'M': '--', 'N': '-.', 'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.',
        'S': '...', 'T': '-', 'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-',
        'Y': '-.--', 'Z': '--..', '0': '-----', '1': '.----', '2': '..---',
        '3': '...--', '4': '....-', '5': '.....', '6': '-....', '7': '--...',
        '8': '---..', '9': '----.', '.': '.-.-.-', ',': '--..--', '?': '..--..',
        ' ': '/'
    }
    
    reverse_dict = {v: k for k, v in morse_dict.items()}

    try:
        if operation == "encode":
            result = []
            for char in text.upper():
                if char in morse_dict:
                    result.append(morse_dict[char])
            return ' '.join(result)
        else:
            # Decode - handle various separators
            text = text.replace('  ', ' / ')  # Double space = word boundary
            words = text.split(' / ')
            decoded_words = []
            for word in words:
                chars = word.strip().split(' ')
                decoded = ''.join(reverse_dict.get(c, '?') for c in chars if c)
                decoded_words.append(decoded)
            return ' '.join(decoded_words)
    except Exception as e:
        return f"Morse code error: {e}"


@quiz_agent.tool
def binary_text(ctx: RunContext[QuizDependencies], data: str, operation: str = "decode") -> str:
    """
    Convert between binary and text.

    Args:
        data: Binary string (for decode) or text (for encode)
        operation: "encode" (text to binary) or "decode" (binary to text)

    Returns:
        Converted result
    """
    try:
        if operation == "encode":
            return ' '.join(format(ord(c), '08b') for c in data)
        else:
            # Clean up input
            data = re.sub(r'[^01]', ' ', data)
            bytes_list = data.split()
            chars = [chr(int(b, 2)) for b in bytes_list if len(b) == 8]
            return ''.join(chars)
    except Exception as e:
        return f"Binary conversion error: {e}"


@quiz_agent.tool
def analyze_encoding(ctx: RunContext[QuizDependencies], data: str) -> str:
    """
    Try to detect and decode various encodings automatically.

    Args:
        data: Encoded data to analyze

    Returns:
        Detected encoding and decoded result
    """
    results = []
    data = data.strip()

    # Check for Base64
    try:
        if re.match(r'^[A-Za-z0-9+/=]+$', data) and len(data) % 4 == 0:
            decoded = base64.b64decode(data).decode('utf-8')
            if decoded.isprintable() or '\n' in decoded:
                results.append(f"Base64: {decoded}")
    except:
        pass

    # Check for URL encoding
    if '%' in data:
        try:
            decoded = unquote(data)
            if decoded != data:
                results.append(f"URL encoded: {decoded}")
        except:
            pass

    # Check for Hex
    try:
        if re.match(r'^[0-9a-fA-F]+$', data) and len(data) % 2 == 0:
            decoded = bytes.fromhex(data).decode('utf-8', errors='replace')
            if any(c.isprintable() for c in decoded):
                results.append(f"Hex: {decoded}")
    except:
        pass

    # Check for binary
    if re.match(r'^[01\s]+$', data):
        try:
            clean = data.replace(' ', '')
            if len(clean) % 8 == 0:
                chars = [chr(int(clean[i:i+8], 2)) for i in range(0, len(clean), 8)]
                decoded = ''.join(chars)
                if decoded.isprintable():
                    results.append(f"Binary: {decoded}")
        except:
            pass

    # Check for ROT13
    rot13_decoded = rot_cipher(None, data, 13)
    if rot13_decoded != data:
        # Check if result looks more like English
        common_words = ['the', 'and', 'is', 'in', 'to', 'of']
        if any(word in rot13_decoded.lower() for word in common_words):
            results.append(f"ROT13: {rot13_decoded}")

    # Check for Morse
    if set(data.strip()) <= set('.-/ '):
        try:
            decoded = morse_code(None, data, "decode")
            if decoded and '?' not in decoded:
                results.append(f"Morse: {decoded}")
        except:
            pass

    if results:
        return '\n'.join(results)
    return "Could not detect encoding. Try specific decode functions."


@quiz_agent.tool
async def jwt_decode(ctx: RunContext[QuizDependencies], token: str) -> str:
    """
    Decode a JWT token (without verification).

    Args:
        token: JWT token string

    Returns:
        Decoded header and payload
    """
    try:
        parts = token.split('.')
        if len(parts) != 3:
            return "Invalid JWT format (expected 3 parts)"

        def decode_part(part):
            # Add padding
            padding = 4 - (len(part) % 4)
            if padding != 4:
                part += '=' * padding
            decoded = base64.urlsafe_b64decode(part)
            return json.loads(decoded)

        header = decode_part(parts[0])
        payload = decode_part(parts[1])

        result = [
            "=== JWT Header ===",
            json.dumps(header, indent=2),
            "\n=== JWT Payload ===",
            json.dumps(payload, indent=2),
        ]

        # Check for common claims
        if 'exp' in payload:
            from datetime import datetime
            exp = datetime.fromtimestamp(payload['exp'])
            result.append(f"\nExpires: {exp}")
        if 'iat' in payload:
            from datetime import datetime
            iat = datetime.fromtimestamp(payload['iat'])
            result.append(f"Issued at: {iat}")

        return '\n'.join(result)

    except Exception as e:
        return f"JWT decode error: {e}"


@quiz_agent.tool
def unicode_info(ctx: RunContext[QuizDependencies], text: str) -> str:
    """
    Get Unicode information about characters.

    Args:
        text: Text to analyze

    Returns:
        Unicode codepoints and names
    """
    import unicodedata
    
    result = []
    for char in text[:50]:  # Limit to 50 chars
        try:
            name = unicodedata.name(char, "UNKNOWN")
            codepoint = ord(char)
            result.append(f"'{char}' U+{codepoint:04X} ({name})")
        except:
            result.append(f"'{char}' U+{ord(char):04X}")
    
    return '\n'.join(result)


@quiz_agent.tool
def number_base_convert(
    ctx: RunContext[QuizDependencies],
    number: str,
    from_base: int,
    to_base: int
) -> str:
    """
    Convert a number between different bases (2, 8, 10, 16, etc).

    Args:
        number: Number string to convert
        from_base: Source base (2-36)
        to_base: Target base (2-36)

    Returns:
        Converted number
    """
    try:
        # Parse from source base
        decimal = int(number, from_base)
        
        # Convert to target base
        if to_base == 10:
            return str(decimal)
        elif to_base == 2:
            return bin(decimal)[2:]
        elif to_base == 8:
            return oct(decimal)[2:]
        elif to_base == 16:
            return hex(decimal)[2:].upper()
        else:
            # Generic base conversion
            if decimal == 0:
                return "0"
            digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            result = ""
            while decimal > 0:
                result = digits[decimal % to_base] + result
                decimal //= to_base
            return result
    except Exception as e:
        return f"Conversion error: {e}"


@quiz_agent.tool
def atbash_cipher(ctx: RunContext[QuizDependencies], text: str) -> str:
    """
    Apply Atbash cipher (reverse alphabet substitution).

    Args:
        text: Text to encode/decode

    Returns:
        Transformed text
    """
    result = []
    for char in text:
        if char.isalpha():
            if char.isupper():
                result.append(chr(ord('Z') - (ord(char) - ord('A'))))
            else:
                result.append(chr(ord('z') - (ord(char) - ord('a'))))
        else:
            result.append(char)
    return ''.join(result)


@quiz_agent.tool
async def crack_simple_cipher(ctx: RunContext[QuizDependencies], ciphertext: str) -> str:
    """
    Try to crack simple substitution ciphers by trying common methods.

    Args:
        ciphertext: Encrypted text to crack

    Returns:
        Possible decryptions
    """
    logger.info(f"Attempting to crack cipher: {ciphertext[:50]}...")
    
    results = []
    
    # Try ROT1-25
    for shift in range(1, 26):
        decoded = rot_cipher(ctx, ciphertext, shift)
        # Simple heuristic - check for common English words
        common = ['the', 'and', 'is', 'in', 'to', 'it', 'of', 'for', 'that']
        score = sum(1 for word in common if word in decoded.lower())
        if score >= 2:
            results.append(f"ROT{shift}: {decoded}")
    
    # Try Atbash
    atbash_decoded = atbash_cipher(ctx, ciphertext)
    common = ['the', 'and', 'is', 'in', 'to', 'it', 'of', 'for', 'that']
    if sum(1 for word in common if word in atbash_decoded.lower()) >= 2:
        results.append(f"Atbash: {atbash_decoded}")
    
    if results:
        return '\n'.join(results)
    
    # If nothing found, return frequency analysis
    freq = {}
    for c in ciphertext.upper():
        if c.isalpha():
            freq[c] = freq.get(c, 0) + 1
    
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
    
    return f"No automatic solution found.\nLetter frequency: {sorted_freq[:10]}\n(English: E T A O I N S H R D)"
