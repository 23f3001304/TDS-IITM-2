"""
Math, Statistics, and Date/Time Tools
"""
import math
import statistics
from datetime import datetime, timedelta

from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent


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
        if not date_string:
            dt = datetime.now()
        else:
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
