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
- video: Video metadata, frame extraction, subtitle extraction

GEMINI TOOL LIMIT: ~57-58 tools maximum
Current config uses ~57 tools. Do not enable more without testing.

Disabled due to Gemini limit:
- crypto: 14 tools (encoding/ciphers) - use encode_decode from text instead
- analysis: 12 tools (patterns/stats) - use execute_python instead
"""
# Core tools - essential functionality (24 tools)
from app.agent.tools import core
from app.agent.tools import media
from app.agent.tools import text
from app.agent.tools import math_tools
from app.agent.tools import http

# Extended tools - stay under 58 tool limit (33 more tools)
from app.agent.tools import web      # +8 = 32
from app.agent.tools import data     # +9 = 41  
from app.agent.tools import files    # +8 = 49
from app.agent.tools import video    # +8 = 57

# DISABLED - exceeds Gemini limit
# from app.agent.tools import crypto    # +14 = 71 (TOO MANY)
# from app.agent.tools import analysis  # +12 = 69 (TOO MANY)
