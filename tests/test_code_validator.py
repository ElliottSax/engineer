#!/usr/bin/env python3
"""
Tests for code_validator.py module.

Run with: python -m unittest tests/test_code_validator.py -v
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from code_validator import (
    CodeValidator, ValidationResult, TestCase,
    STANDARD_TEST_CASES, validate_generated_code
)


class TestValidationResult(unittest.TestCase):
    """Tests for ValidationResult dataclass."""

    def test_default_values(self):
        """Test default ValidationResult values."""
        result = ValidationResult(valid=False)
        self.assertFalse(result.valid)
        self.assertTrue(result.syntax_ok)
        self.assertTrue(result.security_ok)
        self.assertTrue(result.execution_ok)
        self.assertEqual(result.tests_passed, 0)
        self.assertEqual(result.tests_failed, 0)
        self.assertEqual(result.errors, [])

    def test_test_pass_rate_with_tests(self):
        """Test pass rate calculation with tests."""
        result = ValidationResult(
            valid=True,
            tests_passed=7,
            tests_failed=3
        )
        self.assertEqual(result.test_pass_rate, 0.7)

    def test_test_pass_rate_no_tests(self):
        """Test pass rate calculation with no tests."""
        result = ValidationResult(valid=True)
        self.assertEqual(result.test_pass_rate, 0.0)

    def test_test_pass_rate_all_passed(self):
        """Test pass rate when all tests pass."""
        result = ValidationResult(
            valid=True,
            tests_passed=10,
            tests_failed=0
        )
        self.assertEqual(result.test_pass_rate, 1.0)


class TestTestCase(unittest.TestCase):
    """Tests for TestCase dataclass."""

    def test_basic_test_case(self):
        """Test basic TestCase creation."""
        tc = TestCase(name="test_func", args=[1, 2, 3], expected=6)
        self.assertEqual(tc.name, "test_func")
        self.assertEqual(tc.args, [1, 2, 3])
        self.assertEqual(tc.expected, 6)
        self.assertIsNone(tc.expected_type)

    def test_test_case_with_kwargs(self):
        """Test TestCase with kwargs."""
        tc = TestCase(
            name="test_with_kwargs",
            args=[1],
            kwargs={"multiplier": 2},
            expected=2
        )
        self.assertEqual(tc.kwargs["multiplier"], 2)

    def test_test_case_expected_type(self):
        """Test TestCase with expected type."""
        tc = TestCase(name="test_type", args=[], expected=list)
        self.assertEqual(tc.expected_type, list)
        self.assertIsNone(tc.expected)

    def test_test_case_should_raise(self):
        """Test TestCase expecting exception."""
        tc = TestCase(
            name="test_error",
            args=[-1],
            should_raise=ValueError
        )
        self.assertEqual(tc.should_raise, ValueError)


class TestCodeValidatorSyntax(unittest.TestCase):
    """Tests for CodeValidator.validate_syntax()."""

    def setUp(self):
        self.validator = CodeValidator()

    def test_valid_syntax(self):
        """Test valid Python syntax."""
        code = "def hello(): return 'world'"
        ok, errors = self.validator.validate_syntax(code)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_valid_complex_code(self):
        """Test valid complex code."""
        code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

class Calculator:
    def add(self, a, b):
        return a + b
"""
        ok, errors = self.validator.validate_syntax(code)
        self.assertTrue(ok)

    def test_syntax_error_missing_colon(self):
        """Test missing colon syntax error."""
        code = "def hello()"  # Missing colon
        ok, errors = self.validator.validate_syntax(code)
        self.assertFalse(ok)
        self.assertGreater(len(errors), 0)

    def test_syntax_error_invalid_indent(self):
        """Test invalid indentation."""
        code = "def hello():\nreturn 'world'"  # Missing indent
        ok, errors = self.validator.validate_syntax(code)
        self.assertFalse(ok)

    def test_syntax_error_unmatched_parenthesis(self):
        """Test unmatched parenthesis."""
        code = "print('hello'"  # Missing closing paren
        ok, errors = self.validator.validate_syntax(code)
        self.assertFalse(ok)


class TestCodeValidatorSecurity(unittest.TestCase):
    """Tests for CodeValidator.validate_security()."""

    def setUp(self):
        self.validator = CodeValidator()

    def test_safe_code(self):
        """Test safe code passes security check."""
        code = """
def calculate(a, b):
    return a + b
"""
        ok, warnings = self.validator.validate_security(code)
        self.assertTrue(ok)

    def test_allowed_imports(self):
        """Test allowed imports pass security check."""
        code = """
import math
from collections import Counter
"""
        ok, warnings = self.validator.validate_security(code)
        self.assertTrue(ok)

    def test_unsafe_import_os(self):
        """Test unsafe os import is flagged."""
        code = "import os"
        ok, warnings = self.validator.validate_security(code)
        self.assertGreater(len(warnings), 0)
        self.assertTrue(any('os' in w for w in warnings))

    def test_unsafe_import_subprocess(self):
        """Test unsafe subprocess import is flagged."""
        code = "import subprocess"
        ok, warnings = self.validator.validate_security(code)
        self.assertGreater(len(warnings), 0)

    def test_dangerous_exec(self):
        """Test exec() is flagged as dangerous."""
        code = "exec('print(1)')"
        ok, warnings = self.validator.validate_security(code)
        self.assertFalse(ok)
        self.assertTrue(any('dangerous' in w.lower() for w in warnings))

    def test_dangerous_eval(self):
        """Test eval() is flagged as dangerous."""
        code = "result = eval('1 + 1')"
        ok, warnings = self.validator.validate_security(code)
        self.assertFalse(ok)

    def test_file_operations_warning(self):
        """Test file operations are warned about."""
        code = "f = open('file.txt')"
        ok, warnings = self.validator.validate_security(code)
        self.assertTrue(any('file' in w.lower() for w in warnings))

    def test_syntax_error_returns_false(self):
        """Test that syntax errors return False for security."""
        code = "def broken(:"
        ok, warnings = self.validator.validate_security(code)
        self.assertFalse(ok)


class TestCodeValidatorExtractFunctions(unittest.TestCase):
    """Tests for CodeValidator.extract_functions()."""

    def setUp(self):
        self.validator = CodeValidator()

    def test_extract_simple_function(self):
        """Test extracting a simple function."""
        code = """
def add(a, b):
    return a + b
"""
        funcs = self.validator.extract_functions(code)
        self.assertIn('add', funcs)
        self.assertEqual(funcs['add'](2, 3), 5)

    def test_extract_multiple_functions(self):
        """Test extracting multiple functions."""
        code = """
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
"""
        funcs = self.validator.extract_functions(code)
        self.assertIn('add', funcs)
        self.assertIn('multiply', funcs)
        self.assertEqual(funcs['add'](2, 3), 5)
        self.assertEqual(funcs['multiply'](2, 3), 6)

    def test_extract_with_builtin_types(self):
        """Test extracting function using builtin types."""
        code = """
def make_list(a, b, c):
    return [a, b, c]
"""
        funcs = self.validator.extract_functions(code)
        self.assertIn('make_list', funcs)
        self.assertEqual(funcs['make_list'](1, 2, 3), [1, 2, 3])

    def test_private_functions_excluded(self):
        """Test private functions are excluded."""
        code = """
def _private():
    pass

def public():
    pass
"""
        funcs = self.validator.extract_functions(code)
        self.assertNotIn('_private', funcs)
        self.assertIn('public', funcs)

    def test_function_with_math_builtin(self):
        """Test function using pre-loaded math from namespace."""
        code = """
def sqrt(n):
    return math.sqrt(n)
"""
        funcs = self.validator.extract_functions(code)
        self.assertIn('sqrt', funcs)
        self.assertEqual(funcs['sqrt'](16), 4.0)


class TestCodeValidatorRunTest(unittest.TestCase):
    """Tests for CodeValidator.run_test()."""

    def setUp(self):
        self.validator = CodeValidator()

    def test_passing_test(self):
        """Test a passing test case."""
        def add(a, b):
            return a + b

        tc = TestCase(name="test_add", args=[2, 3], expected=5)
        passed, msg = self.validator.run_test(add, tc)
        self.assertTrue(passed)
        self.assertIn("matches", msg)

    def test_failing_test(self):
        """Test a failing test case."""
        def add(a, b):
            return a + b + 1  # Wrong implementation

        tc = TestCase(name="test_add", args=[2, 3], expected=5)
        passed, msg = self.validator.run_test(add, tc)
        self.assertFalse(passed)
        self.assertIn("Expected", msg)

    def test_type_check_passing(self):
        """Test type check passing."""
        def get_list():
            return [1, 2, 3]

        tc = TestCase(name="test_type", args=[], expected=list)
        passed, msg = self.validator.run_test(get_list, tc)
        self.assertTrue(passed)

    def test_type_check_failing(self):
        """Test type check failing."""
        def get_tuple():
            return (1, 2, 3)

        tc = TestCase(name="test_type", args=[], expected=list)
        passed, msg = self.validator.run_test(get_tuple, tc)
        self.assertFalse(passed)
        self.assertIn("Expected type", msg)

    def test_expected_exception(self):
        """Test expected exception is raised."""
        def divide(a, b):
            return a / b

        tc = TestCase(name="test_divide_zero", args=[1, 0], should_raise=ZeroDivisionError)
        passed, msg = self.validator.run_test(divide, tc)
        self.assertTrue(passed)
        self.assertIn("Correctly raised", msg)

    def test_unexpected_exception(self):
        """Test unexpected exception fails."""
        def broken(x):
            raise ValueError("broken")

        tc = TestCase(name="test_broken", args=[1], expected=1)
        passed, msg = self.validator.run_test(broken, tc)
        self.assertFalse(passed)
        self.assertIn("Error", msg)


class TestCodeValidatorValidate(unittest.TestCase):
    """Tests for CodeValidator.validate() - full validation."""

    def setUp(self):
        self.validator = CodeValidator()

    def test_valid_code_no_tests(self):
        """Test valid code without test cases."""
        code = """
def hello():
    return "world"
"""
        result = self.validator.validate(code)
        self.assertTrue(result.syntax_ok)
        self.assertTrue(result.security_ok)
        self.assertTrue(result.execution_ok)

    def test_valid_code_with_tests(self):
        """Test valid code with passing test cases."""
        code = """
def add(a, b):
    return a + b
"""
        tests = [
            TestCase("test_add_1", [1, 2], expected=3),
            TestCase("test_add_2", [5, 5], expected=10),
        ]
        result = self.validator.validate(code, "add", tests)
        self.assertTrue(result.valid)
        self.assertEqual(result.tests_passed, 2)
        self.assertEqual(result.tests_failed, 0)

    def test_invalid_syntax_fails(self):
        """Test invalid syntax fails validation."""
        code = "def broken(:"
        result = self.validator.validate(code)
        self.assertFalse(result.valid)
        self.assertFalse(result.syntax_ok)

    def test_function_not_found(self):
        """Test missing function is reported."""
        code = """
def other_func():
    pass
"""
        tests = [TestCase("test", [1], expected=1)]
        result = self.validator.validate(code, "missing_func", tests)
        self.assertGreater(result.tests_failed, 0)
        self.assertTrue(any("not found" in e for e in result.errors))

    def test_execution_error(self):
        """Test execution error is caught."""
        code = """
raise RuntimeError("fail at import time")
"""
        result = self.validator.validate(code)
        self.assertFalse(result.execution_ok)


class TestCodeValidatorQuickTest(unittest.TestCase):
    """Tests for CodeValidator.quick_test()."""

    def setUp(self):
        self.validator = CodeValidator()

    def test_quick_test_passing(self):
        """Test quick_test with passing code."""
        code = """
def double(n):
    return n * 2
"""
        passed, msg = self.validator.quick_test(code, "double", [5], 10)
        self.assertTrue(passed)

    def test_quick_test_failing(self):
        """Test quick_test with failing code."""
        code = """
def double(n):
    return n  # Wrong
"""
        passed, msg = self.validator.quick_test(code, "double", [5], 10)
        self.assertFalse(passed)


class TestStandardTestCases(unittest.TestCase):
    """Tests for STANDARD_TEST_CASES."""

    def test_standard_test_cases_exist(self):
        """Test that standard test cases are defined."""
        self.assertGreater(len(STANDARD_TEST_CASES), 0)

    def test_common_functions_have_tests(self):
        """Test common functions have test cases."""
        expected = ['is_palindrome', 'fibonacci', 'is_prime', 'find_max']
        for func in expected:
            self.assertIn(func, STANDARD_TEST_CASES)

    def test_test_cases_are_valid(self):
        """Test that all test cases are valid TestCase objects."""
        for func_name, tests in STANDARD_TEST_CASES.items():
            for test in tests:
                self.assertIsInstance(test, TestCase)
                self.assertIsInstance(test.name, str)
                self.assertIsInstance(test.args, list)


class TestValidateGeneratedCode(unittest.TestCase):
    """Tests for validate_generated_code convenience function."""

    def test_with_known_function(self):
        """Test validation with known function name."""
        code = """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
        result = validate_generated_code(code, "is_prime")
        self.assertTrue(result.valid)
        self.assertGreater(result.tests_passed, 0)

    def test_with_unknown_function(self):
        """Test validation with unknown function name."""
        code = """
def custom_func(x):
    return x * 2
"""
        result = validate_generated_code(code, "custom_func")
        # Should still validate syntax and security
        self.assertTrue(result.syntax_ok)
        self.assertTrue(result.security_ok)

    def test_without_function_name(self):
        """Test validation without function name."""
        code = """
def any_func():
    return 42
"""
        result = validate_generated_code(code)
        self.assertTrue(result.syntax_ok)


class TestIntegration(unittest.TestCase):
    """Integration tests for the validator."""

    def test_fibonacci_implementation(self):
        """Test validating a fibonacci implementation."""
        code = """
def fibonacci(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib[:n]
"""
        result = validate_generated_code(code, "fibonacci")
        self.assertTrue(result.valid)
        self.assertEqual(result.tests_passed, 3)

    def test_find_max_implementation(self):
        """Test validating a find_max implementation."""
        code = """
def find_max(items):
    if not items:
        return None
    return max(items)
"""
        result = validate_generated_code(code, "find_max")
        self.assertTrue(result.valid)

    def test_binary_search_implementation(self):
        """Test validating a binary_search implementation."""
        code = """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
"""
        result = validate_generated_code(code, "binary_search")
        self.assertTrue(result.valid)
        self.assertEqual(result.tests_passed, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
