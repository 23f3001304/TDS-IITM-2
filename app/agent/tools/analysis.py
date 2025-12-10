"""
Advanced Analysis and Problem-Solving Tools
- Pattern recognition
- String analysis
- Logic puzzles
- Sequence analysis
"""
import itertools
import math
import re
from collections import Counter
from functools import reduce

from loguru import logger
from pydantic_ai import RunContext

from app.agent.models import QuizDependencies
from app.agent.prompts import quiz_agent
from app.sandbox import sandbox


@quiz_agent.tool
def find_pattern(ctx: RunContext[QuizDependencies], sequence: str) -> str:
    """
    Analyze a sequence of numbers to find patterns.

    Args:
        sequence: Comma or space separated numbers

    Returns:
        Detected patterns and next predicted values
    """
    # Parse numbers
    nums = [float(x.strip()) for x in re.split(r'[,\s]+', sequence.strip()) if x.strip()]
    
    if len(nums) < 3:
        return "Need at least 3 numbers to find pattern"
    
    results = []
    
    # Check arithmetic sequence
    diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
    if len(set(diffs)) == 1:
        d = diffs[0]
        next_val = nums[-1] + d
        results.append(f"Arithmetic sequence (d={d}): next = {next_val}")
    
    # Check geometric sequence
    if all(n != 0 for n in nums[:-1]):
        ratios = [nums[i+1] / nums[i] for i in range(len(nums)-1)]
        if len(set(round(r, 6) for r in ratios)) == 1:
            r = ratios[0]
            next_val = nums[-1] * r
            results.append(f"Geometric sequence (r={r}): next = {next_val}")
    
    # Check quadratic (second differences constant)
    if len(nums) >= 4:
        second_diffs = [diffs[i+1] - diffs[i] for i in range(len(diffs)-1)]
        if len(set(second_diffs)) == 1:
            d2 = second_diffs[0]
            next_diff = diffs[-1] + d2
            next_val = nums[-1] + next_diff
            results.append(f"Quadratic sequence (2nd diff={d2}): next = {next_val}")
    
    # Check Fibonacci-like
    is_fib = True
    for i in range(2, len(nums)):
        if nums[i] != nums[i-1] + nums[i-2]:
            is_fib = False
            break
    if is_fib:
        next_val = nums[-1] + nums[-2]
        results.append(f"Fibonacci-like sequence: next = {next_val}")
    
    # Check powers
    for base in [2, 3, 10]:
        powers = [base ** i for i in range(len(nums))]
        if nums == powers:
            results.append(f"Powers of {base}: next = {base ** len(nums)}")
    
    # Check factorials
    factorials = [math.factorial(i) for i in range(len(nums))]
    if nums == factorials:
        results.append(f"Factorials: next = {math.factorial(len(nums))}")
    
    # Check primes
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    primes = []
    n = 2
    while len(primes) < len(nums) + 1:
        if is_prime(n):
            primes.append(n)
        n += 1
    
    if nums == primes[:len(nums)]:
        results.append(f"Prime numbers: next = {primes[len(nums)]}")
    
    # Check squares/cubes
    squares = [i**2 for i in range(1, len(nums)+2)]
    if nums == squares[:len(nums)]:
        results.append(f"Perfect squares: next = {squares[len(nums)]}")
    
    cubes = [i**3 for i in range(1, len(nums)+2)]
    if nums == cubes[:len(nums)]:
        results.append(f"Perfect cubes: next = {cubes[len(nums)]}")
    
    if results:
        return '\n'.join(results)
    
    # If no pattern found, show differences
    return f"No standard pattern detected.\nFirst differences: {diffs}\nSecond differences: {second_diffs if len(nums) >= 4 else 'N/A'}"


@quiz_agent.tool
def analyze_string_pattern(ctx: RunContext[QuizDependencies], strings: str) -> str:
    """
    Analyze a list of strings to find patterns.

    Args:
        strings: Newline or comma separated strings to analyze

    Returns:
        Common patterns, prefixes, suffixes, etc.
    """
    # Parse strings
    items = [s.strip() for s in re.split(r'[\n,]', strings.strip()) if s.strip()]
    
    if len(items) < 2:
        return "Need at least 2 strings to find patterns"
    
    results = []
    
    # Common prefix
    prefix = items[0]
    for s in items[1:]:
        while not s.startswith(prefix) and prefix:
            prefix = prefix[:-1]
    if prefix:
        results.append(f"Common prefix: '{prefix}'")
    
    # Common suffix
    suffix = items[0]
    for s in items[1:]:
        while not s.endswith(suffix) and suffix:
            suffix = suffix[1:]
    if suffix:
        results.append(f"Common suffix: '{suffix}'")
    
    # Length pattern
    lengths = [len(s) for s in items]
    if len(set(lengths)) == 1:
        results.append(f"All strings have length {lengths[0]}")
    else:
        results.append(f"Lengths: {lengths}")
    
    # Character set analysis
    all_chars = set(''.join(items))
    if all_chars <= set('0123456789'):
        results.append("Contains only digits")
    elif all_chars <= set('abcdefghijklmnopqrstuvwxyz'):
        results.append("Contains only lowercase letters")
    elif all_chars <= set('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        results.append("Contains only uppercase letters")
    elif all_chars <= set('0123456789abcdefABCDEF'):
        results.append("Looks like hexadecimal")
    
    # Check if sorted
    if items == sorted(items):
        results.append("Strings are sorted ascending")
    elif items == sorted(items, reverse=True):
        results.append("Strings are sorted descending")
    
    # Check for sequence pattern in variable parts
    if prefix:
        suffixes = [s[len(prefix):] for s in items]
        try:
            nums = [int(s) for s in suffixes]
            pattern_result = find_pattern(ctx, ','.join(map(str, nums)))
            results.append(f"Variable part analysis: {pattern_result}")
        except:
            pass
    
    return '\n'.join(results) if results else "No obvious patterns found"


@quiz_agent.tool
def generate_permutations(ctx: RunContext[QuizDependencies], items: str, length: int = 0) -> str:
    """
    Generate permutations of items.

    Args:
        items: Comma or space separated items (or characters in a word)
        length: Length of permutations (0 = all items)

    Returns:
        List of permutations
    """
    # Parse items
    if ',' in items:
        elements = [x.strip() for x in items.split(',')]
    elif ' ' in items:
        elements = items.split()
    else:
        elements = list(items)
    
    r = length if length > 0 else len(elements)
    
    perms = list(itertools.permutations(elements, r))
    
    if len(perms) > 100:
        return f"Too many permutations ({len(perms)}). Showing first 100:\n" + \
               '\n'.join(' '.join(map(str, p)) for p in perms[:100])
    
    return f"Permutations ({len(perms)} total):\n" + '\n'.join(' '.join(map(str, p)) for p in perms)


@quiz_agent.tool
def generate_combinations(ctx: RunContext[QuizDependencies], items: str, length: int) -> str:
    """
    Generate combinations of items.

    Args:
        items: Comma or space separated items
        length: Size of each combination

    Returns:
        List of combinations
    """
    if ',' in items:
        elements = [x.strip() for x in items.split(',')]
    elif ' ' in items:
        elements = items.split()
    else:
        elements = list(items)
    
    combs = list(itertools.combinations(elements, length))
    
    if len(combs) > 100:
        return f"Too many combinations ({len(combs)}). Showing first 100:\n" + \
               '\n'.join(' '.join(map(str, c)) for c in combs[:100])
    
    return f"Combinations ({len(combs)} total):\n" + '\n'.join(' '.join(map(str, c)) for c in combs)


@quiz_agent.tool
def calculate_statistics(ctx: RunContext[QuizDependencies], data: str) -> str:
    """
    Calculate comprehensive statistics for numeric data.

    Args:
        data: Comma or newline separated numbers

    Returns:
        Statistical summary
    """
    import statistics
    
    nums = [float(x.strip()) for x in re.split(r'[,\n\s]+', data.strip()) if x.strip()]
    
    if not nums:
        return "No numbers found"
    
    n = len(nums)
    results = [
        f"Count: {n}",
        f"Sum: {sum(nums)}",
        f"Mean: {statistics.mean(nums):.6f}",
        f"Median: {statistics.median(nums):.6f}",
    ]
    
    if n > 1:
        results.extend([
            f"Std Dev: {statistics.stdev(nums):.6f}",
            f"Variance: {statistics.variance(nums):.6f}",
        ])
    
    results.extend([
        f"Min: {min(nums)}",
        f"Max: {max(nums)}",
        f"Range: {max(nums) - min(nums)}",
    ])
    
    # Quartiles
    sorted_nums = sorted(nums)
    if n >= 4:
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        results.extend([
            f"Q1 (25%): {sorted_nums[q1_idx]}",
            f"Q3 (75%): {sorted_nums[q3_idx]}",
            f"IQR: {sorted_nums[q3_idx] - sorted_nums[q1_idx]}",
        ])
    
    # Mode
    try:
        mode = statistics.mode(nums)
        results.append(f"Mode: {mode}")
    except:
        results.append("Mode: No unique mode")
    
    return '\n'.join(results)


@quiz_agent.tool
def solve_equation(ctx: RunContext[QuizDependencies], equation: str, variable: str = "x") -> str:
    """
    Solve simple algebraic equations.

    Args:
        equation: Equation like "2x + 5 = 15" or "x^2 - 4 = 0"
        variable: Variable to solve for (default 'x')

    Returns:
        Solution(s)
    """
    code = f'''
from sympy import symbols, Eq, solve, sympify
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

x = symbols("{variable}")

# Parse equation
equation = "{equation}"
equation = equation.replace("^", "**")

if "=" in equation:
    left, right = equation.split("=")
    transformations = standard_transformations + (implicit_multiplication_application,)
    left_expr = parse_expr(left.strip(), local_dict={{"{variable}": x}}, transformations=transformations)
    right_expr = parse_expr(right.strip(), local_dict={{"{variable}": x}}, transformations=transformations)
    eq = Eq(left_expr, right_expr)
else:
    transformations = standard_transformations + (implicit_multiplication_application,)
    eq = parse_expr(equation, local_dict={{"{variable}": x}}, transformations=transformations)

solutions = solve(eq, x)
print(f"Solutions for {variable}:")
for sol in solutions:
    print(f"  {variable} = {{sol}}")
'''
    
    logger.info(f"Solving equation: {equation}")
    result = sandbox.run_sync(code, timeout=30)
    
    if result.success:
        return result.stdout.strip()
    return f"Could not solve: {result.stderr}"


@quiz_agent.tool
def prime_factorization(ctx: RunContext[QuizDependencies], number: int) -> str:
    """
    Find prime factorization of a number.

    Args:
        number: Number to factorize

    Returns:
        Prime factors and their powers
    """
    if number < 2:
        return f"{number} has no prime factorization"
    
    factors = {}
    n = number
    d = 2
    
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    
    factor_str = ' × '.join(f"{p}^{e}" if e > 1 else str(p) for p, e in sorted(factors.items()))
    
    results = [
        f"{number} = {factor_str}",
        f"Prime factors: {sorted(factors.keys())}",
        f"Number of divisors: {reduce(lambda a, b: a * (b + 1), factors.values(), 1)}",
    ]
    
    # List all divisors if not too many
    divisors = [1]
    for p, e in factors.items():
        divisors = [d * (p ** i) for d in divisors for i in range(e + 1)]
    divisors = sorted(set(divisors))
    
    if len(divisors) <= 50:
        results.append(f"All divisors: {divisors}")
    
    return '\n'.join(results)


@quiz_agent.tool
def gcd_lcm(ctx: RunContext[QuizDependencies], numbers: str) -> str:
    """
    Calculate GCD and LCM of numbers.

    Args:
        numbers: Comma or space separated numbers

    Returns:
        GCD, LCM, and factorizations
    """
    nums = [int(x.strip()) for x in re.split(r'[,\s]+', numbers.strip()) if x.strip()]
    
    if len(nums) < 2:
        return "Need at least 2 numbers"
    
    gcd = reduce(math.gcd, nums)
    
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)
    
    lcm_result = reduce(lcm, nums)
    
    return f"GCD({', '.join(map(str, nums))}) = {gcd}\nLCM({', '.join(map(str, nums))}) = {lcm_result}"


@quiz_agent.tool
def string_distance(ctx: RunContext[QuizDependencies], str1: str, str2: str) -> str:
    """
    Calculate various string distances/similarities.

    Args:
        str1: First string
        str2: Second string

    Returns:
        Levenshtein distance, similarity ratio, common subsequence
    """
    # Levenshtein distance
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    lev_distance = dp[m][n]
    
    # Longest common subsequence
    dp_lcs = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp_lcs[i][j] = dp_lcs[i-1][j-1] + 1
            else:
                dp_lcs[i][j] = max(dp_lcs[i-1][j], dp_lcs[i][j-1])
    
    lcs_length = dp_lcs[m][n]
    
    # Similarity ratio
    similarity = (1 - lev_distance / max(m, n)) * 100 if max(m, n) > 0 else 100
    
    results = [
        f"String 1: '{str1}' (length {m})",
        f"String 2: '{str2}' (length {n})",
        f"Levenshtein distance: {lev_distance}",
        f"Similarity: {similarity:.1f}%",
        f"Longest common subsequence length: {lcs_length}",
    ]
    
    # Common characters
    common = set(str1) & set(str2)
    results.append(f"Common characters: {sorted(common)}")
    
    return '\n'.join(results)


@quiz_agent.tool
def analyze_frequency(ctx: RunContext[QuizDependencies], text: str, unit: str = "char") -> str:
    """
    Analyze frequency distribution of characters, words, or n-grams.

    Args:
        text: Text to analyze
        unit: "char", "word", "bigram", "trigram"

    Returns:
        Frequency distribution
    """
    if unit == "char":
        items = list(text.lower())
        items = [c for c in items if c.isalnum()]
    elif unit == "word":
        items = re.findall(r'\b\w+\b', text.lower())
    elif unit == "bigram":
        words = re.findall(r'\b\w+\b', text.lower())
        items = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
    elif unit == "trigram":
        words = re.findall(r'\b\w+\b', text.lower())
        items = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
    else:
        return f"Unknown unit: {unit}"
    
    freq = Counter(items)
    total = sum(freq.values())
    
    results = [f"Total {unit}s: {total}", f"Unique {unit}s: {len(freq)}", "", "Top 20:"]
    
    for item, count in freq.most_common(20):
        pct = (count / total) * 100
        results.append(f"  '{item}': {count} ({pct:.1f}%)")
    
    return '\n'.join(results)


@quiz_agent.tool
def validate_format(ctx: RunContext[QuizDependencies], value: str, format_type: str) -> str:
    """
    Validate if a value matches a specific format.

    Args:
        value: Value to validate
        format_type: email, url, ip, uuid, date, phone, credit_card, hex, json

    Returns:
        Validation result with details
    """
    patterns = {
        'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
        'url': r'^https?://[^\s<>"{}|\\^`\[\]]+$',
        'ip': r'^(\d{1,3}\.){3}\d{1,3}$',
        'ipv6': r'^([0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$',
        'uuid': r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$',
        'date': r'^\d{4}-\d{2}-\d{2}$',
        'phone': r'^[\d\s\-\+\(\)]{7,20}$',
        'hex': r'^(0x)?[0-9a-fA-F]+$',
    }
    
    if format_type == 'json':
        try:
            import json
            json.loads(value)
            return f"Valid JSON\nParsed type: {type(json.loads(value)).__name__}"
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"
    
    if format_type == 'credit_card':
        # Luhn algorithm
        digits = [int(d) for d in value if d.isdigit()]
        if len(digits) < 13 or len(digits) > 19:
            return "Invalid credit card: wrong length"
        
        checksum = 0
        for i, d in enumerate(reversed(digits)):
            if i % 2 == 1:
                d *= 2
                if d > 9:
                    d -= 9
            checksum += d
        
        if checksum % 10 == 0:
            return "Valid credit card (Luhn check passed)"
        return "Invalid credit card (Luhn check failed)"
    
    pattern = patterns.get(format_type)
    if not pattern:
        return f"Unknown format: {format_type}. Available: {list(patterns.keys())}"
    
    if re.match(pattern, value):
        if format_type == 'ip':
            parts = value.split('.')
            if all(0 <= int(p) <= 255 for p in parts):
                return f"Valid {format_type}: {value}"
            return f"Invalid {format_type}: octets must be 0-255"
        return f"Valid {format_type}: {value}"
    
    return f"Invalid {format_type}: {value}"


@quiz_agent.tool
async def brute_force_answer(
    ctx: RunContext[QuizDependencies],
    template: str,
    options: str,
    validation: str = ""
) -> str:
    """
    Generate all possible answers from a template with placeholders.

    Args:
        template: Template with {0}, {1} placeholders
        options: Pipe-separated options for each placeholder (e.g., "a,b,c|1,2,3")
        validation: Optional regex to filter valid results

    Returns:
        All possible combinations
    """
    option_lists = [opt.split(',') for opt in options.split('|')]
    
    results = []
    for combo in itertools.product(*option_lists):
        try:
            result = template.format(*combo)
            if not validation or re.match(validation, result):
                results.append(result)
        except:
            continue
    
    if len(results) > 500:
        return f"Too many combinations ({len(results)}). Showing first 500:\n" + '\n'.join(results[:500])
    
    return f"Possible answers ({len(results)}):\n" + '\n'.join(results)
