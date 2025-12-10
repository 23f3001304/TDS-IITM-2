"""
Tools Package - Register all tools with the quiz agent

This module imports all tool modules to register them with the quiz_agent.
Tools are automatically registered via decorators when imported.

Tool Categories:
- core: URL scraping, Python execution, shell commands, file download
- media: Audio transcription, image analysis, archive extraction
- text: Hashing, encoding/decoding, regex extraction
- math_tools: Mathematical expressions, date operations
- http: API requests, CSV analysis
- web: HTML parsing, link/table/form extraction
- data: JSON/XML parsing, Excel, pivot tables, data joins
- files: PDF extraction, binary analysis, file diff
- crypto: Base64 variants, ciphers, JWT decode
- analysis: Pattern recognition, statistics, equation solving
- video: Video metadata, frame extraction, subtitle extraction
"""
# Core tools - essential functionality
from app.agent.tools import core
from app.agent.tools import media
from app.agent.tools import text
from app.agent.tools import math_tools
from app.agent.tools import http

# Advanced web tools
from app.agent.tools import web

# Data processing tools
from app.agent.tools import data

# File manipulation tools
from app.agent.tools import files

# Encoding and cryptography tools
from app.agent.tools import crypto

# Analysis and problem-solving tools
from app.agent.tools import analysis

# Video and multimedia tools
from app.agent.tools import video
