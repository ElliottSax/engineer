"""
Safe Code Executor

Provides sandboxed code execution to replace dangerous exec/eval calls.
Uses AST parsing and restricted execution environments.

SECURITY NOTE: This module should be used instead of raw exec/eval.
Even with sandboxing, executing untrusted code carries risk.
Prefer alternative approaches when possible.
"""

import ast
import sys
import signal
import logging
import multiprocessing
from typing import Any, Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from contextlib import contextmanager
import resource

logger = logging.getLogger(__name__)


class CodeExecutionError(Exception):
    """Raised when code execution fails or is blocked."""
    pass


class SecurityViolation(CodeExecutionError):
    """Raised when code attempts to use blocked functionality."""
    pass


class TimeoutError(CodeExecutionError):
    """Raised when code execution times out."""
    pass


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    stdout: str = ""
    stderr: str = ""
    execution_time: float = 0.0


# Safe built-in functions that can be used in sandboxed execution
SAFE_BUILTINS: Dict[str, Any] = {
    # Type constructors
    'bool': bool,
    'int': int,
    'float': float,
    'str': str,
    'list': list,
    'dict': dict,
    'set': set,
    'tuple': tuple,
    'frozenset': frozenset,
    'bytes': bytes,
    'bytearray': bytearray,

    # Iteration and sequences
    'len': len,
    'range': range,
    'enumerate': enumerate,
    'zip': zip,
    'map': map,
    'filter': filter,
    'reversed': reversed,
    'sorted': sorted,

    # Math functions
    'abs': abs,
    'round': round,
    'min': min,
    'max': max,
    'sum': sum,
    'pow': pow,
    'divmod': divmod,

    # String operations
    'chr': chr,
    'ord': ord,
    'repr': repr,
    'format': format,

    # Comparisons and checks
    'all': all,
    'any': any,
    'isinstance': isinstance,
    'issubclass': issubclass,
    'callable': callable,
    'hasattr': hasattr,
    'getattr': getattr,
    'setattr': setattr,

    # Other safe builtins
    'iter': iter,
    'next': next,
    'slice': slice,
    'id': id,
    'hash': hash,
    'type': type,
    'object': object,

    # Constants
    'True': True,
    'False': False,
    'None': None,

    # Exceptions (for catching, not raising arbitrary ones)
    'Exception': Exception,
    'ValueError': ValueError,
    'TypeError': TypeError,
    'IndexError': IndexError,
    'KeyError': KeyError,
    'AttributeError': AttributeError,
    'StopIteration': StopIteration,
    'ZeroDivisionError': ZeroDivisionError,
}

# AST node types that are blocked for security
BLOCKED_AST_NODES: Set[type] = {
    ast.Import,
    ast.ImportFrom,
}

# Blocked attribute names
BLOCKED_ATTRIBUTES: Set[str] = {
    '__import__',
    '__builtins__',
    '__code__',
    '__globals__',
    '__locals__',
    '__dict__',
    '__class__',
    '__bases__',
    '__mro__',
    '__subclasses__',
    '__reduce__',
    '__reduce_ex__',
    '__getstate__',
    '__setstate__',
    'func_globals',
    'func_code',
    'gi_frame',
    'gi_code',
    'co_code',
    'f_globals',
    'f_locals',
    'f_builtins',
}

# Blocked function names
BLOCKED_FUNCTIONS: Set[str] = {
    'exec',
    'eval',
    'compile',
    'open',
    'input',
    'breakpoint',
    '__import__',
    'globals',
    'locals',
    'vars',
    'dir',
    'getattr',  # Can be used to bypass restrictions
    'setattr',
    'delattr',
    'memoryview',
}


class SecurityVisitor(ast.NodeVisitor):
    """
    AST visitor that checks for security violations.
    """

    def __init__(self):
        self.violations: List[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        modules = [alias.name for alias in node.names]
        self.violations.append(f"Import blocked: {', '.join(modules)}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.violations.append(f"Import from '{node.module}' blocked")

    def visit_Call(self, node: ast.Call) -> None:
        # Check for blocked function calls
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_FUNCTIONS:
                self.violations.append(f"Function '{node.func.id}' is blocked")

        # Check for blocked method calls on attributes
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in BLOCKED_ATTRIBUTES:
                self.violations.append(f"Attribute '{node.func.attr}' is blocked")

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in BLOCKED_ATTRIBUTES:
            self.violations.append(f"Access to '{node.attr}' is blocked")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # Check for __dict__ access via subscript
        if isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str) and node.slice.value.startswith('__'):
                self.violations.append(f"Access to dunder via subscript is blocked")
        self.generic_visit(node)


def validate_code(code: str) -> List[str]:
    """
    Validate code for security issues without executing it.

    Args:
        code: Python code to validate

    Returns:
        List of security violations found (empty if code is safe)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax error: {e}"]

    visitor = SecurityVisitor()
    visitor.visit(tree)

    return visitor.violations


class SafeCodeExecutor:
    """
    Executes code in a sandboxed environment.

    Security measures:
    - AST validation to block dangerous constructs
    - Restricted builtins
    - Execution timeout
    - Memory limits (on Unix systems)
    - Separate process execution option

    Example:
        executor = SafeCodeExecutor(timeout=5.0)
        result = executor.execute("x = 1 + 2")
        print(result.result)  # Access namespace after execution
    """

    def __init__(
        self,
        timeout: float = 30.0,
        max_memory_mb: int = 512,
        allowed_builtins: Optional[Dict[str, Any]] = None,
        allow_imports: bool = False
    ):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        self.allowed_builtins = allowed_builtins or SAFE_BUILTINS.copy()
        self.allow_imports = allow_imports

    def validate(self, code: str) -> List[str]:
        """Validate code without executing it."""
        violations = validate_code(code)

        if not self.allow_imports:
            # Check for import-like strings
            if 'import ' in code or '__import__' in code:
                if not any('import' in v.lower() for v in violations):
                    violations.append("Import statements are not allowed")

        return violations

    def execute(
        self,
        code: str,
        namespace: Optional[Dict[str, Any]] = None,
        isolated_process: bool = False
    ) -> ExecutionResult:
        """
        Execute code in a sandboxed environment.

        Args:
            code: Python code to execute
            namespace: Initial namespace variables
            isolated_process: If True, execute in a separate process

        Returns:
            ExecutionResult with execution details
        """
        # Validate code first
        violations = self.validate(code)
        if violations:
            return ExecutionResult(
                success=False,
                error=f"Security violations: {'; '.join(violations)}"
            )

        if isolated_process:
            return self._execute_in_process(code, namespace)
        else:
            return self._execute_direct(code, namespace)

    def _execute_direct(
        self,
        code: str,
        namespace: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code directly in current process with restrictions."""
        import time

        # Create restricted namespace
        safe_namespace = {
            '__builtins__': self.allowed_builtins,
        }
        if namespace:
            safe_namespace.update(namespace)

        start_time = time.time()

        try:
            # Parse and compile
            tree = ast.parse(code)
            code_obj = compile(tree, '<sandboxed>', 'exec')

            # Set up timeout handler (Unix only)
            if hasattr(signal, 'SIGALRM'):
                def timeout_handler(signum, frame):
                    raise TimeoutError("Execution timed out")

                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))

            try:
                # Execute with restricted namespace
                # NOTE: We're using exec here but with a heavily restricted namespace
                exec(code_obj, safe_namespace)
            finally:
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            execution_time = time.time() - start_time

            # Remove builtins from result namespace
            result_namespace = {
                k: v for k, v in safe_namespace.items()
                if not k.startswith('_')
            }

            return ExecutionResult(
                success=True,
                result=result_namespace,
                execution_time=execution_time
            )

        except TimeoutError as e:
            return ExecutionResult(
                success=False,
                error=str(e),
                execution_time=self.timeout
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ExecutionResult(
                success=False,
                error=f"{type(e).__name__}: {str(e)}",
                execution_time=execution_time
            )

    def _execute_in_process(
        self,
        code: str,
        namespace: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute code in a separate process for complete isolation."""
        import time

        def worker(code: str, namespace: dict, result_queue):
            """Worker function to run in separate process."""
            try:
                # Set memory limit (Unix only)
                if hasattr(resource, 'RLIMIT_AS'):
                    max_mem = self.max_memory_mb * 1024 * 1024
                    resource.setrlimit(resource.RLIMIT_AS, (max_mem, max_mem))

                # Execute code
                safe_namespace = {
                    '__builtins__': SAFE_BUILTINS,
                }
                if namespace:
                    safe_namespace.update(namespace)

                tree = ast.parse(code)
                code_obj = compile(tree, '<sandboxed>', 'exec')
                exec(code_obj, safe_namespace)

                # Filter result
                result = {
                    k: v for k, v in safe_namespace.items()
                    if not k.startswith('_') and _is_serializable(v)
                }

                result_queue.put({'success': True, 'result': result})

            except Exception as e:
                result_queue.put({
                    'success': False,
                    'error': f"{type(e).__name__}: {str(e)}"
                })

        start_time = time.time()

        # Create result queue
        result_queue = multiprocessing.Queue()

        # Start worker process
        process = multiprocessing.Process(
            target=worker,
            args=(code, namespace or {}, result_queue)
        )
        process.start()

        # Wait with timeout
        process.join(timeout=self.timeout)

        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()

            return ExecutionResult(
                success=False,
                error="Execution timed out",
                execution_time=self.timeout
            )

        execution_time = time.time() - start_time

        # Get result
        try:
            result_data = result_queue.get_nowait()
            return ExecutionResult(
                success=result_data['success'],
                result=result_data.get('result'),
                error=result_data.get('error'),
                execution_time=execution_time
            )
        except Exception:
            return ExecutionResult(
                success=False,
                error="Failed to get execution result",
                execution_time=execution_time
            )


def _is_serializable(obj: Any) -> bool:
    """Check if an object can be serialized (for process communication)."""
    try:
        import pickle
        pickle.dumps(obj)
        return True
    except Exception:
        return False


def safe_eval(
    expression: str,
    namespace: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0
) -> Any:
    """
    Safely evaluate a simple expression.

    This is a convenience function for evaluating simple expressions
    like mathematical formulas without the full executor overhead.

    Args:
        expression: Expression to evaluate
        namespace: Variables available during evaluation
        timeout: Maximum execution time

    Returns:
        Result of the expression

    Raises:
        CodeExecutionError: If evaluation fails or is blocked
    """
    # Validate - expressions should not contain statements
    try:
        tree = ast.parse(expression, mode='eval')
    except SyntaxError as e:
        raise CodeExecutionError(f"Invalid expression: {e}")

    # Check for security issues
    violations = validate_code(expression)
    if violations:
        raise SecurityViolation(f"Blocked: {'; '.join(violations)}")

    # Create safe namespace
    safe_namespace = {
        '__builtins__': {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool,
        }
    }
    if namespace:
        safe_namespace.update(namespace)

    try:
        code_obj = compile(tree, '<eval>', 'eval')
        return eval(code_obj, safe_namespace)
    except Exception as e:
        raise CodeExecutionError(f"Evaluation failed: {e}")


# Alternative approach: Code transformation instead of execution
def analyze_code_structure(code: str) -> Dict[str, Any]:
    """
    Analyze code structure without executing it.

    This is a safer alternative when you need to understand
    code structure without running it.

    Returns:
        Dictionary with code structure information
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {'error': f"Syntax error: {e}"}

    structure = {
        'functions': [],
        'classes': [],
        'imports': [],
        'variables': [],
        'complexity': 0
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            structure['functions'].append({
                'name': node.name,
                'args': [arg.arg for arg in node.args.args],
                'line': node.lineno
            })
        elif isinstance(node, ast.ClassDef):
            structure['classes'].append({
                'name': node.name,
                'line': node.lineno
            })
        elif isinstance(node, ast.Import):
            for alias in node.names:
                structure['imports'].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            structure['imports'].append(f"{node.module}.{', '.join(a.name for a in node.names)}")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    structure['variables'].append(target.id)

        # Simple complexity metric
        structure['complexity'] += 1

    return structure
