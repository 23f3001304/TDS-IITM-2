"""
HTTP and API Tools
"""
import csv
import io
import json
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent


@quiz_agent.tool
async def make_api_request(
    ctx: RunContext[QuizDependencies],
    url: str,
    method: str = "GET",
    body: str = "",
    headers: str = ""
) -> str:
    """
    Make an HTTP request to an API endpoint. Use ONLY for GET requests to fetch data.
    DO NOT use this for submitting quiz answers - answers are submitted automatically.

    Args:
        url: API URL (can be relative)
        method: HTTP method (GET only - POST/PUT/DELETE not recommended)
        body: Request body (JSON string for POST/PUT)
        headers: Optional headers as JSON string

    Returns:
        Response body
    """
    if '/submit' in url.lower():
        logger.warning(f"Blocked submission attempt via make_api_request: {url}")
        return "ERROR: Do not use make_api_request for submissions. Just return the answer and it will be submitted automatically."
    
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    cache_key = f"api:{method}:{url}"
    if method.upper() == "GET" and cache_key in ctx.deps.url_cache:
        logger.info(f"Using cached API response for: {url}")
        return ctx.deps.url_cache[cache_key]

    logger.info(f"API request: {method} {url}")
    if body:
        logger.info(f"API request body: {body}")

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

            logger.info(f"API response status: {response.status_code}")
            logger.info(f"API response body: {response.text}")
            result = response.text
            
            if method.upper() == "GET" and response.status_code == 200:
                ctx.deps.url_cache[cache_key] = result
            
            return result

    except Exception as e:
        logger.error(f"API request error: {e}")
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
