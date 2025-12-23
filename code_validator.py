#!/usr/bin/env python3
"""
Code Validator - Validates and tests generated code.

Features:
- Syntax validation
- Security checks
- Execution testing
- Test case verification
- Quality metrics
"""

import ast
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from io import StringIO
import contextlib

sys.path.insert(0, str(Path(__file__).parent))


@dataclass
class ValidationResult:
    """Result of code validation."""
    valid: bool
    syntax_ok: bool = True
    security_ok: bool = True
    execution_ok: bool = True
    tests_passed: int = 0
    tests_failed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    output: str = ""

    @property
    def test_pass_rate(self) -> float:
        total = self.tests_passed + self.tests_failed
        return self.tests_passed / total if total > 0 else 0.0


@dataclass
class TestCase:
    """A test case for code validation."""
    name: str
    args: List[Any]
    kwargs: Dict[str, Any] = field(default_factory=dict)
    expected: Any = None
    expected_type: Optional[type] = None
    should_raise: Optional[type] = None

    def __post_init__(self):
        if isinstance(self.expected, type):
            self.expected_type = self.expected
            self.expected = None


class CodeValidator:
    """Validates and tests generated code."""

    # Safe modules that can be imported for testing
    ALLOWED_IMPORTS = {
        'math', 'collections', 'itertools', 'functools',
        'operator', 'string', 're', 'json', 'datetime',
        'typing', 'dataclasses', 'enum', 'copy',
    }

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def validate_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """Check code syntax without executing."""
        errors = []
        try:
            ast.parse(code)
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors

    def validate_security(self, code: str) -> Tuple[bool, List[str]]:
        """Check for potentially dangerous code patterns."""
        warnings = []

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False, ["Cannot parse code for security check"]

        # Check for dangerous patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in self.ALLOWED_IMPORTS:
                        warnings.append(f"Import of '{alias.name}' may be unsafe")

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in self.ALLOWED_IMPORTS:
                    warnings.append(f"Import from '{node.module}' may be unsafe")

            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('exec', 'eval', 'compile', '__import__'):
                        warnings.append(f"Use of '{node.func.id}' is dangerous")
                    elif node.func.id == 'open':
                        warnings.append("File operations detected")

            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    if node.attr not in ('__init__', '__str__', '__repr__', '__len__',
                                         '__iter__', '__next__', '__enter__', '__exit__',
                                         '__name__', '__doc__'):
                        warnings.append(f"Access to dunder '{node.attr}' may be unsafe")

        return len([w for w in warnings if 'dangerous' in w.lower()]) == 0, warnings

    def extract_functions(self, code: str) -> Dict[str, Callable]:
        """Execute code and extract defined functions."""
        namespace = {
            '__builtins__': {
                'abs': abs, 'all': all, 'any': any, 'bool': bool,
                'dict': dict, 'enumerate': enumerate, 'filter': filter,
                'float': float, 'frozenset': frozenset, 'int': int,
                'isinstance': isinstance, 'len': len, 'list': list,
                'map': map, 'max': max, 'min': min, 'pow': pow,
                'print': print, 'range': range, 'reversed': reversed,
                'round': round, 'set': set, 'slice': slice, 'sorted': sorted,
                'str': str, 'sum': sum, 'tuple': tuple, 'type': type,
                'zip': zip, 'True': True, 'False': False, 'None': None,
                'Exception': Exception, 'ValueError': ValueError,
                'TypeError': TypeError, 'IndexError': IndexError,
                'KeyError': KeyError, 'ZeroDivisionError': ZeroDivisionError,
                'StopIteration': StopIteration,
            },
            # Support for __name__ == "__main__" blocks
            '__name__': '__code_validator__',
        }

        # Allow safe imports in namespace
        import math
        import collections
        from collections import Counter
        namespace['math'] = math
        namespace['collections'] = collections
        namespace['Counter'] = Counter

        # Add Counter to builtins for 'from collections import Counter' pattern
        namespace['__builtins__']['Counter'] = Counter

        exec(code, namespace)

        # Extract callable functions
        functions = {}
        for name, obj in namespace.items():
            if callable(obj) and not name.startswith('_'):
                if not isinstance(obj, type):  # Skip classes
                    functions[name] = obj
                elif name[0].isupper():  # Include classes with capitalized names
                    functions[name] = obj

        return functions

    def run_test(self, func: Callable, test: TestCase) -> Tuple[bool, str]:
        """Run a single test case."""
        try:
            start = time.time()
            result = func(*test.args, **test.kwargs)
            elapsed = time.time() - start

            if elapsed > self.timeout:
                return False, f"Timeout: took {elapsed:.2f}s"

            if test.should_raise:
                return False, f"Expected {test.should_raise.__name__} but no exception raised"

            if test.expected_type is not None:
                if isinstance(result, test.expected_type):
                    return True, f"Type check passed: {type(result).__name__}"
                else:
                    return False, f"Expected type {test.expected_type.__name__}, got {type(result).__name__}"

            if test.expected is not None:
                if result == test.expected:
                    return True, f"Result matches: {result}"
                else:
                    return False, f"Expected {test.expected}, got {result}"

            return True, f"Executed successfully: {result}"

        except Exception as e:
            if test.should_raise and isinstance(e, test.should_raise):
                return True, f"Correctly raised {type(e).__name__}"
            return False, f"Error: {type(e).__name__}: {e}"

    def validate(
        self,
        code: str,
        function_name: Optional[str] = None,
        test_cases: Optional[List[TestCase]] = None
    ) -> ValidationResult:
        """
        Comprehensive code validation.

        Args:
            code: Python code to validate
            function_name: Specific function to test (optional)
            test_cases: Test cases to run (optional)

        Returns:
            ValidationResult with all validation details
        """
        result = ValidationResult(valid=False)

        # 1. Syntax validation
        syntax_ok, syntax_errors = self.validate_syntax(code)
        result.syntax_ok = syntax_ok
        result.errors.extend(syntax_errors)

        if not syntax_ok:
            return result

        # 2. Security validation
        security_ok, security_warnings = self.validate_security(code)
        result.security_ok = security_ok
        result.warnings.extend(security_warnings)

        # 3. Execution test
        try:
            stdout_capture = StringIO()
            start_time = time.time()

            with contextlib.redirect_stdout(stdout_capture):
                functions = self.extract_functions(code)

            result.execution_time = time.time() - start_time
            result.output = stdout_capture.getvalue()
            result.execution_ok = True

        except Exception as e:
            result.execution_ok = False
            result.errors.append(f"Execution error: {type(e).__name__}: {e}")
            return result

        # 4. Function tests
        if test_cases:
            if function_name and function_name in functions:
                func = functions[function_name]
                for test in test_cases:
                    passed, message = self.run_test(func, test)
                    if passed:
                        result.tests_passed += 1
                    else:
                        result.tests_failed += 1
                        result.errors.append(f"Test '{test.name}': {message}")
            elif function_name:
                result.errors.append(f"Function '{function_name}' not found")
                result.tests_failed = len(test_cases)
            else:
                # Try to find matching function by test name patterns
                for test in test_cases:
                    # Try to find function matching test name
                    matching_func = None
                    for fname, func in functions.items():
                        if fname in test.name or test.name in fname:
                            matching_func = func
                            break

                    if matching_func:
                        passed, message = self.run_test(matching_func, test)
                        if passed:
                            result.tests_passed += 1
                        else:
                            result.tests_failed += 1
                            result.errors.append(f"Test '{test.name}': {message}")
                    else:
                        result.tests_failed += 1
                        result.errors.append(f"No function found for test '{test.name}'")

        # Determine overall validity
        result.valid = (
            result.syntax_ok and
            result.security_ok and
            result.execution_ok and
            result.tests_failed == 0
        )

        return result

    def quick_test(
        self,
        code: str,
        function_name: str,
        args: List[Any],
        expected: Any
    ) -> Tuple[bool, str]:
        """Quick test of a function with single test case."""
        test = TestCase(
            name=f"test_{function_name}",
            args=args,
            expected=expected
        )

        result = self.validate(code, function_name, [test])

        if result.valid:
            return True, "Test passed"
        else:
            return False, "; ".join(result.errors)


# Pre-built test cases for common functions
STANDARD_TEST_CASES = {
    "is_palindrome": [
        TestCase("palindrome_true", ["racecar"], expected=True),
        TestCase("palindrome_false", ["hello"], expected=False),
        TestCase("palindrome_single", ["a"], expected=True),
    ],
    "fibonacci": [
        TestCase("fib_5", [5], expected=[0, 1, 1, 2, 3]),
        TestCase("fib_1", [1], expected=[0]),
        TestCase("fib_0", [0], expected=[]),
    ],
    "is_prime": [
        TestCase("prime_7", [7], expected=True),
        TestCase("prime_4", [4], expected=False),
        TestCase("prime_2", [2], expected=True),
        TestCase("prime_1", [1], expected=False),
    ],
    "find_max": [
        TestCase("max_list", [[1, 5, 3, 9, 2]], expected=9),
        TestCase("max_single", [[42]], expected=42),
    ],
    "find_min": [
        TestCase("min_list", [[1, 5, 3, 9, 2]], expected=1),
        TestCase("min_single", [[42]], expected=42),
    ],
    "reverse_string": [
        TestCase("reverse_hello", ["hello"], expected="olleh"),
        TestCase("reverse_single", ["a"], expected="a"),
    ],
    "sum_list": [
        TestCase("sum_basic", [[1, 2, 3, 4, 5]], expected=15),
        TestCase("sum_empty", [[]], expected=0),
    ],
    "count_words": [
        TestCase("count_two", ["hello world"], expected=2),
        TestCase("count_one", ["hello"], expected=1),
    ],
    "find_duplicates": [
        TestCase("dup_found", [[1, 2, 3, 2, 4, 3]], expected=[2, 3]),
        TestCase("dup_none", [[1, 2, 3]], expected=[]),
    ],
    "remove_duplicates": [
        TestCase("remove_basic", [[1, 2, 2, 3]], expected=[1, 2, 3]),
        TestCase("remove_empty", [[]], expected=[]),
    ],
    "flatten_list": [
        TestCase("flatten_nested", [[[1, 2], [3, [4, 5]]]], expected=[1, 2, 3, 4, 5]),
        TestCase("flatten_simple", [[[1], [2], [3]]], expected=[1, 2, 3]),
    ],
    "get_average": [
        TestCase("avg_basic", [[2, 4, 6]], expected=4.0),
        TestCase("avg_single", [[5]], expected=5.0),
    ],
    "filter_even": [
        TestCase("even_mixed", [[1, 2, 3, 4, 5, 6]], expected=[2, 4, 6]),
        TestCase("even_none", [[1, 3, 5]], expected=[]),
    ],
    "filter_odd": [
        TestCase("odd_mixed", [[1, 2, 3, 4, 5, 6]], expected=[1, 3, 5]),
        TestCase("odd_none", [[2, 4, 6]], expected=[]),
    ],
    "calculate_power": [
        TestCase("power_2_3", [2, 3], expected=8),
        TestCase("power_5_2", [5, 2], expected=25),
    ],
    "calculate_gcd": [
        TestCase("gcd_12_8", [12, 8], expected=4),
        TestCase("gcd_17_13", [17, 13], expected=1),
    ],
    "is_anagram": [
        TestCase("anagram_true", ["listen", "silent"], expected=True),
        TestCase("anagram_false", ["hello", "world"], expected=False),
    ],
    "binary_search": [
        TestCase("search_found", [[1, 2, 3, 4, 5], 3], expected=2),
        TestCase("search_not_found", [[1, 2, 3, 4, 5], 6], expected=-1),
    ],
    "capitalize_words": [
        TestCase("cap_basic", ["hello world"], expected="Hello World"),
        TestCase("cap_single", ["hello"], expected="Hello"),
    ],
    "get_unique": [
        TestCase("unique_basic", [[1, 2, 2, 3, 3, 3]], expected=[1, 2, 3]),
        TestCase("unique_same", [[1, 1, 1]], expected=[1]),
    ],
}


def validate_generated_code(
    code: str,
    function_name: Optional[str] = None
) -> ValidationResult:
    """
    Convenience function to validate generated code.

    Automatically selects test cases based on function name.
    """
    validator = CodeValidator()

    test_cases = None
    if function_name and function_name in STANDARD_TEST_CASES:
        test_cases = STANDARD_TEST_CASES[function_name]

    return validator.validate(code, function_name, test_cases)


if __name__ == "__main__":
    # Demo
    code = '''
def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
'''

    print("Validating is_prime function...")
    result = validate_generated_code(code, "is_prime")

    print(f"\nValidation Result:")
    print(f"  Valid: {result.valid}")
    print(f"  Syntax OK: {result.syntax_ok}")
    print(f"  Security OK: {result.security_ok}")
    print(f"  Execution OK: {result.execution_ok}")
    print(f"  Tests: {result.tests_passed}/{result.tests_passed + result.tests_failed}")

    if result.errors:
        print(f"\nErrors:")
        for e in result.errors:
            print(f"  - {e}")

    if result.warnings:
        print(f"\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")
