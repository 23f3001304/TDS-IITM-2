"""
Tools Package - Register all tools with the quiz agent

This module imports all tool modules to register them with the quiz_agent.
Tools are automatically registered via decorators when imported.

TOOL INVENTORY (~48 tools - optimized for Gemini's ~58 tool limit):

CORE (8 tools):
- scrape_url: Fetch webpage content
- execute_python: Run Python code
- pip_install: Install Python packages
- run_shell_command: Execute shell commands
- download_file: Download files
- read_file_content: Read local files
- get_page_links: Get links from current page
- get_page_text: Get text from current page

MEDIA (5 tools):
- transcribe_audio: Speech-to-text
- analyze_image: OCR, describe, detect objects
- analyze_video: Video analysis
- extract_zip: Extract zip files
- extract_archive: Extract any archive

TEXT (7 tools):
- compute_hash: MD5, SHA256, etc.
- encode_decode: Base64, URL, hex
- compute_email_code: Email hash code
- extract_with_regex: Regex extraction
- extract_numbers: Find numbers in text
- count_occurrences: Count pattern matches
- parse_json: Parse JSON data

MATH (2 tools):
- do_math: Evaluate math expressions
- get_date_info: Date calculations

HTTP (2 tools):
- make_api_request: HTTP requests
- analyze_csv_data: Quick CSV analysis

UNIFIED (7 tools - consolidated from web/data/files/video/analysis):
- parse_webpage: Extract links, tables, forms, text, meta from HTML
- process_data: JSON/XML/CSV query, filter, aggregate
- process_document: PDF text/tables extraction
- process_archive: List/extract archive contents
- analyze_media: Video/audio metadata
- analyze_text: Pattern detection, frequency, encoding detection
- compute_math: Advanced math, factorization, base conversion

DESIGN PRINCIPLES:
1. Single Responsibility: Each tool does one thing well
2. Consolidated: Related functions merged (e.g., 8 web tools -> 1 parse_webpage)
3. Parameterized: Use 'operation' parameter to select behavior
4. Safe: All tools have proper error handling
"""
# Core tools - essential functionality
from app.agent.tools import core      # 8 tools
from app.agent.tools import media     # 5 tools  
from app.agent.tools import text      # 7 tools
from app.agent.tools import math_tools  # 2 tools
from app.agent.tools import http      # 2 tools

# Unified advanced tools - consolidated from web/data/files/video/analysis
from app.agent.tools import unified   # 7 tools

# TOTAL: ~31 core + 7 unified = ~38 tools (well under 58 limit)

# DISABLED - functionality now in unified.py:
# from app.agent.tools import web      # Merged into unified.parse_webpage
# from app.agent.tools import data     # Merged into unified.process_data
# from app.agent.tools import files    # Merged into unified.process_document/process_archive
# from app.agent.tools import video    # Merged into unified.analyze_media
# from app.agent.tools import crypto   # Merged into unified.analyze_text
# from app.agent.tools import analysis # Merged into unified.analyze_text/compute_math
