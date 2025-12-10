"""
Advanced Web and Browser Tools
- HTML parsing
- Link extraction
- Form detection
- Table extraction
- JavaScript analysis
"""
import re
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup
from loguru import logger
from pydantic_ai import RunContext
import httpx

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent


@quiz_agent.tool
async def extract_links(ctx: RunContext[QuizDependencies], url: str, filter_pattern: str = "") -> str:
    """
    Extract all links from a webpage.

    Args:
        url: URL to extract links from
        filter_pattern: Optional regex to filter links (e.g., "\.pdf$" for PDF links)

    Returns:
        List of links found, one per line
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting links from: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            links = []

            for a in soup.find_all('a', href=True):
                href = a['href']
                full_url = urljoin(url, href)
                text = a.get_text(strip=True)[:50]
                
                if filter_pattern:
                    if re.search(filter_pattern, full_url, re.IGNORECASE):
                        links.append(f"{full_url} [{text}]" if text else full_url)
                else:
                    links.append(f"{full_url} [{text}]" if text else full_url)

            logger.info(f"Found {len(links)} links")
            return '\n'.join(links) if links else "No links found"

    except Exception as e:
        return f"Error extracting links: {e}"


@quiz_agent.tool
async def extract_tables(ctx: RunContext[QuizDependencies], url: str, table_index: int = 0) -> str:
    """
    Extract HTML tables from a webpage.

    Args:
        url: URL containing tables
        table_index: Which table to extract (0-based). Use -1 for all tables.

    Returns:
        Table data in CSV-like format
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting tables from: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            tables = soup.find_all('table')

            if not tables:
                return "No tables found on page"

            result = []
            tables_to_process = tables if table_index == -1 else [tables[table_index]] if table_index < len(tables) else []

            for i, table in enumerate(tables_to_process):
                if table_index == -1:
                    result.append(f"=== Table {i} ===")

                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['th', 'td'])
                    cell_text = [cell.get_text(strip=True) for cell in cells]
                    result.append(','.join(cell_text))

            return '\n'.join(result) if result else "No table data found"

    except IndexError:
        return f"Table index {table_index} out of range (found {len(tables)} tables)"
    except Exception as e:
        return f"Error extracting tables: {e}"


@quiz_agent.tool
async def extract_forms(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract form information from a webpage.

    Args:
        url: URL to analyze for forms

    Returns:
        Form details including action, method, and input fields
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting forms from: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            forms = soup.find_all('form')

            if not forms:
                return "No forms found on page"

            result = []
            for i, form in enumerate(forms):
                result.append(f"=== Form {i} ===")
                result.append(f"Action: {form.get('action', 'none')}")
                result.append(f"Method: {form.get('method', 'GET')}")
                result.append("Fields:")

                for inp in form.find_all(['input', 'select', 'textarea']):
                    name = inp.get('name', 'unnamed')
                    inp_type = inp.get('type', inp.name)
                    value = inp.get('value', '')
                    placeholder = inp.get('placeholder', '')
                    result.append(f"  - {name} (type={inp_type}, value={value}, placeholder={placeholder})")

            return '\n'.join(result)

    except Exception as e:
        return f"Error extracting forms: {e}"


@quiz_agent.tool
async def extract_scripts(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract JavaScript code and data from a webpage.

    Args:
        url: URL to analyze

    Returns:
        JavaScript content including inline scripts and data embedded in page
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting scripts from: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            scripts = soup.find_all('script')

            result = []
            for i, script in enumerate(scripts):
                src = script.get('src', '')
                if src:
                    result.append(f"=== External Script {i}: {src} ===")
                else:
                    content = script.string
                    if content and content.strip():
                        # Look for embedded data
                        json_matches = re.findall(r'(?:var\s+\w+\s*=\s*|const\s+\w+\s*=\s*|let\s+\w+\s*=\s*)(\{[^}]+\}|\[[^\]]+\])', content)
                        result.append(f"=== Inline Script {i} ===")
                        result.append(content[:2000])
                        if len(content) > 2000:
                            result.append("... (truncated)")

            return '\n'.join(result) if result else "No scripts found"

    except Exception as e:
        return f"Error extracting scripts: {e}"


@quiz_agent.tool
async def extract_meta(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract metadata from a webpage (title, meta tags, OpenGraph, etc).

    Args:
        url: URL to analyze

    Returns:
        Metadata information
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting metadata from: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            result = []

            # Title
            title = soup.find('title')
            if title:
                result.append(f"Title: {title.get_text(strip=True)}")

            # Meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', ''))
                content = meta.get('content', '')
                if name and content:
                    result.append(f"{name}: {content}")

            # Headings
            for i in range(1, 4):
                for h in soup.find_all(f'h{i}'):
                    result.append(f"H{i}: {h.get_text(strip=True)[:100]}")

            return '\n'.join(result) if result else "No metadata found"

    except Exception as e:
        return f"Error extracting metadata: {e}"


@quiz_agent.tool
def parse_url(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Parse a URL into its components.

    Args:
        url: URL to parse

    Returns:
        URL components (scheme, host, path, query params, etc.)
    """
    try:
        parsed = urlparse(url)
        result = [
            f"Scheme: {parsed.scheme}",
            f"Host: {parsed.netloc}",
            f"Path: {parsed.path}",
            f"Query: {parsed.query}",
            f"Fragment: {parsed.fragment}",
        ]

        if parsed.query:
            params = parse_qs(parsed.query)
            result.append("Query Parameters:")
            for key, values in params.items():
                result.append(f"  {key}: {', '.join(values)}")

        return '\n'.join(result)
    except Exception as e:
        return f"Error parsing URL: {e}"


@quiz_agent.tool
async def extract_images(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Extract image URLs from a webpage.

    Args:
        url: URL to extract images from

    Returns:
        List of image URLs with alt text
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Extracting images from: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            images = []

            for img in soup.find_all('img'):
                src = img.get('src', '')
                if src:
                    full_url = urljoin(url, src)
                    alt = img.get('alt', '')[:50]
                    images.append(f"{full_url} [alt: {alt}]" if alt else full_url)

            logger.info(f"Found {len(images)} images")
            return '\n'.join(images) if images else "No images found"

    except Exception as e:
        return f"Error extracting images: {e}"


@quiz_agent.tool
async def get_page_structure(ctx: RunContext[QuizDependencies], url: str) -> str:
    """
    Get the DOM structure of a webpage (useful for understanding complex layouts).

    Args:
        url: URL to analyze

    Returns:
        Simplified DOM tree
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Getting page structure: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style
            for element in soup(['script', 'style', 'noscript']):
                element.decompose()

            def get_structure(element, depth=0, max_depth=4):
                if depth > max_depth:
                    return []
                
                lines = []
                indent = "  " * depth
                
                if hasattr(element, 'name') and element.name:
                    attrs = []
                    if element.get('id'):
                        attrs.append(f"#{element['id']}")
                    if element.get('class'):
                        attrs.append(f".{'.'.join(element['class'][:2])}")
                    
                    attr_str = ''.join(attrs)
                    text = element.get_text(strip=True)[:30] if element.string else ""
                    
                    lines.append(f"{indent}<{element.name}{attr_str}> {text}")
                    
                    for child in element.children:
                        if hasattr(child, 'name'):
                            lines.extend(get_structure(child, depth + 1, max_depth))
                
                return lines

            body = soup.find('body')
            if body:
                structure = get_structure(body)
                return '\n'.join(structure[:100])  # Limit output
            return "No body element found"

    except Exception as e:
        return f"Error getting page structure: {e}"
