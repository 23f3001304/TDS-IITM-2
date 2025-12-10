"""
Advanced Data Processing Tools
- JSON manipulation
- XML parsing
- CSV advanced operations
- Excel file handling
- Data transformation
"""
import csv
import io
import json
import re
from collections import Counter
from urllib.parse import urljoin

import httpx
from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.sandbox import sandbox


@quiz_agent.tool
async def query_json(ctx: RunContext[QuizDependencies], url: str, jq_path: str) -> str:
    """
    Query JSON data using a JQ-like path expression.

    Args:
        url: URL of JSON file or data
        jq_path: Path expression like ".data.items[0].name" or ".results[*].value"

    Returns:
        Extracted data
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Querying JSON: {url} with path: {jq_path}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        def navigate(obj, path):
            parts = re.findall(r'\.(\w+)|\[(\d+)\]|\[\*\]', path)
            
            for key, index, _ in parts:
                if key:
                    if isinstance(obj, dict):
                        obj = obj.get(key)
                    elif isinstance(obj, list):
                        obj = [item.get(key) if isinstance(item, dict) else None for item in obj]
                elif index:
                    if isinstance(obj, list):
                        obj = obj[int(index)]
                elif _ == '':  # [*] wildcard
                    if not isinstance(obj, list):
                        return obj
                    continue
                
                if obj is None:
                    return None
            
            return obj

        result = navigate(data, jq_path)
        
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2)
        return str(result)

    except Exception as e:
        return f"Error querying JSON: {e}"


@quiz_agent.tool
async def parse_xml(ctx: RunContext[QuizDependencies], url: str, xpath: str = "") -> str:
    """
    Parse XML data and optionally extract using XPath-like expressions.

    Args:
        url: URL of XML file
        xpath: Simple XPath like "//item/name" or "root/data"

    Returns:
        Extracted XML data or full structure
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Parsing XML: {url}")

    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

        code = f'''
from lxml import etree

xml_content = """{response.text}"""
root = etree.fromstring(xml_content.encode())

xpath_expr = "{xpath}"

if xpath_expr:
    results = root.xpath(xpath_expr)
    for r in results:
        if hasattr(r, 'text'):
            print(r.text if r.text else etree.tostring(r, encoding='unicode'))
        else:
            print(str(r))
else:
    print(etree.tostring(root, pretty_print=True, encoding='unicode')[:3000])
'''
        result = await sandbox.execute_code(code, timeout=30)
        return result.stdout.strip() if result.success else f"XML parse error: {result.stderr}"

    except Exception as e:
        return f"Error parsing XML: {e}"


@quiz_agent.tool
async def analyze_excel(ctx: RunContext[QuizDependencies], url: str, sheet: str = "", operation: str = "info") -> str:
    """
    Analyze Excel files (.xlsx).

    Args:
        url: URL of Excel file
        sheet: Sheet name (empty = first sheet)
        operation: One of: info, head, columns, stats, unique:column_name, sum:column_name

    Returns:
        Analysis result
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Analyzing Excel: {url} (op={operation})")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import pandas as pd

df = pd.read_excel("{safe_path}", sheet_name="{sheet}" if "{sheet}" else 0)
operation = "{operation}"

if operation == "info":
    print(f"Shape: {{df.shape}}")
    print(f"Columns: {{list(df.columns)}}")
    print(f"\\nData types:\\n{{df.dtypes}}")
elif operation == "head":
    print(df.head(20).to_string())
elif operation == "columns":
    for col in df.columns:
        print(f"{{col}}: {{df[col].dtype}}")
elif operation == "stats":
    print(df.describe().to_string())
elif operation.startswith("unique:"):
    col = operation.split(":", 1)[1]
    print(df[col].unique().tolist())
elif operation.startswith("sum:"):
    col = operation.split(":", 1)[1]
    print(df[col].sum())
elif operation.startswith("count:"):
    col = operation.split(":", 1)[1]
    print(df[col].value_counts().to_string())
else:
    print(df.to_string())
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Excel error: {result.stderr}"

    except Exception as e:
        return f"Error analyzing Excel: {e}"


@quiz_agent.tool
async def pivot_csv(
    ctx: RunContext[QuizDependencies],
    url: str,
    index_col: str,
    values_col: str,
    agg_func: str = "sum"
) -> str:
    """
    Create a pivot table from CSV data.

    Args:
        url: URL of CSV file
        index_col: Column to use as index (rows)
        values_col: Column to aggregate
        agg_func: Aggregation function (sum, mean, count, max, min)

    Returns:
        Pivot table result
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Creating pivot from: {url}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import pandas as pd

df = pd.read_csv("{safe_path}")
pivot = df.groupby("{index_col}")["{values_col}"].agg("{agg_func}")
print(pivot.to_string())
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Pivot error: {result.stderr}"

    except Exception as e:
        return f"Error creating pivot: {e}"


@quiz_agent.tool
async def join_datasets(
    ctx: RunContext[QuizDependencies],
    url1: str,
    url2: str,
    key_col: str,
    how: str = "inner"
) -> str:
    """
    Join two CSV/JSON datasets on a common key.

    Args:
        url1: URL of first dataset
        url2: URL of second dataset
        key_col: Column name to join on
        how: Join type (inner, left, right, outer)

    Returns:
        First 30 rows of joined data
    """
    logger.info(f"Joining datasets on {key_col}")

    try:
        path1 = await sandbox.download_file(url1 if url1.startswith('http') else urljoin(ctx.deps.current_url, url1))
        path2 = await sandbox.download_file(url2 if url2.startswith('http') else urljoin(ctx.deps.current_url, url2))
        
        safe_path1 = path1.replace('\\', '/')
        safe_path2 = path2.replace('\\', '/')

        code = f'''
import pandas as pd

def read_file(path):
    if path.endswith('.json'):
        return pd.read_json(path)
    else:
        return pd.read_csv(path)

df1 = read_file("{safe_path1}")
df2 = read_file("{safe_path2}")

result = df1.merge(df2, on="{key_col}", how="{how}")
print(f"Joined shape: {{result.shape}}")
print(result.head(30).to_string())
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Join error: {result.stderr}"

    except Exception as e:
        return f"Error joining datasets: {e}"


@quiz_agent.tool
def transform_data(ctx: RunContext[QuizDependencies], data: str, transformation: str) -> str:
    """
    Transform text/data using common operations.

    Args:
        data: Input data (text or JSON string)
        transformation: One of: sort, sort_desc, reverse, unique, count, flatten, 
                       split:delimiter, join:delimiter, filter:pattern

    Returns:
        Transformed data
    """
    try:
        lines = data.strip().split('\n')

        if transformation == "sort":
            return '\n'.join(sorted(lines))
        elif transformation == "sort_desc":
            return '\n'.join(sorted(lines, reverse=True))
        elif transformation == "reverse":
            return '\n'.join(reversed(lines))
        elif transformation == "unique":
            seen = set()
            unique = []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    unique.append(line)
            return '\n'.join(unique)
        elif transformation == "count":
            counts = Counter(lines)
            return '\n'.join(f"{v}: {k}" for k, v in counts.most_common())
        elif transformation == "flatten":
            try:
                obj = json.loads(data)
                def flatten(obj, prefix=''):
                    items = []
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            new_key = f"{prefix}.{k}" if prefix else k
                            items.extend(flatten(v, new_key))
                    elif isinstance(obj, list):
                        for i, v in enumerate(obj):
                            items.extend(flatten(v, f"{prefix}[{i}]"))
                    else:
                        items.append(f"{prefix}: {obj}")
                    return items
                return '\n'.join(flatten(obj))
            except json.JSONDecodeError:
                return "Data is not valid JSON for flattening"
        elif transformation.startswith("split:"):
            delim = transformation.split(":", 1)[1]
            result = []
            for line in lines:
                result.extend(line.split(delim))
            return '\n'.join(result)
        elif transformation.startswith("join:"):
            delim = transformation.split(":", 1)[1]
            return delim.join(lines)
        elif transformation.startswith("filter:"):
            pattern = transformation.split(":", 1)[1]
            return '\n'.join(line for line in lines if re.search(pattern, line))
        else:
            return f"Unknown transformation: {transformation}"

    except Exception as e:
        return f"Transform error: {e}"


@quiz_agent.tool
async def aggregate_column(
    ctx: RunContext[QuizDependencies],
    url: str,
    column: str,
    group_by: str = "",
    operation: str = "sum"
) -> str:
    """
    Aggregate a column in CSV/Excel data.

    Args:
        url: URL of data file
        column: Column to aggregate
        group_by: Optional column to group by (empty = aggregate all)
        operation: sum, mean, median, std, min, max, count

    Returns:
        Aggregation result
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Aggregating {column} by {group_by} with {operation}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import pandas as pd

path = "{safe_path}"
if path.endswith('.xlsx') or path.endswith('.xls'):
    df = pd.read_excel(path)
else:
    df = pd.read_csv(path)

column = "{column}"
group_by = "{group_by}"
operation = "{operation}"

if group_by:
    result = df.groupby(group_by)[column].agg(operation)
    print(result.to_string())
else:
    if operation == "sum":
        print(df[column].sum())
    elif operation == "mean":
        print(df[column].mean())
    elif operation == "median":
        print(df[column].median())
    elif operation == "std":
        print(df[column].std())
    elif operation == "min":
        print(df[column].min())
    elif operation == "max":
        print(df[column].max())
    elif operation == "count":
        print(df[column].count())
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Aggregation error: {result.stderr}"

    except Exception as e:
        return f"Error aggregating: {e}"


@quiz_agent.tool
async def filter_data(
    ctx: RunContext[QuizDependencies],
    url: str,
    condition: str,
    columns: str = ""
) -> str:
    """
    Filter data based on conditions.

    Args:
        url: URL of CSV/Excel file
        condition: Pandas query condition like "age > 30" or "status == 'active'"
        columns: Comma-separated columns to return (empty = all)

    Returns:
        Filtered rows
    """
    if not url.startswith('http'):
        url = urljoin(ctx.deps.current_url, url)

    logger.info(f"Filtering data with condition: {condition}")

    try:
        local_path = await sandbox.download_file(url)
        safe_path = local_path.replace('\\', '/')

        code = f'''
import pandas as pd

path = "{safe_path}"
if path.endswith('.xlsx') or path.endswith('.xls'):
    df = pd.read_excel(path)
else:
    df = pd.read_csv(path)

filtered = df.query("{condition}")

columns = "{columns}"
if columns:
    cols = [c.strip() for c in columns.split(',')]
    filtered = filtered[cols]

print(f"Found {{len(filtered)}} matching rows")
print(filtered.to_string())
'''
        result = await sandbox.execute_code(code, timeout=60)
        return result.stdout.strip() if result.success else f"Filter error: {result.stderr}"

    except Exception as e:
        return f"Error filtering: {e}"


@quiz_agent.tool
def compare_values(ctx: RunContext[QuizDependencies], list1: str, list2: str) -> str:
    """
    Compare two lists/sets of values.

    Args:
        list1: First list (newline or comma separated)
        list2: Second list (newline or comma separated)

    Returns:
        Set operations: intersection, only in list1, only in list2
    """
    def parse_list(s):
        if '\n' in s:
            return set(line.strip() for line in s.split('\n') if line.strip())
        return set(item.strip() for item in s.split(',') if item.strip())

    set1 = parse_list(list1)
    set2 = parse_list(list2)

    intersection = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1

    result = [
        f"List 1 count: {len(set1)}",
        f"List 2 count: {len(set2)}",
        f"\nIn both ({len(intersection)}):",
        *list(intersection)[:20],
        f"\nOnly in list 1 ({len(only1)}):",
        *list(only1)[:20],
        f"\nOnly in list 2 ({len(only2)}):",
        *list(only2)[:20],
    ]

    return '\n'.join(str(r) for r in result)
