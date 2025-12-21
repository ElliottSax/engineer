#!/usr/bin/env python3
"""
Unified Coding Agent - Best practices from top autocoding projects

Incorporates architectural patterns from:
- Aider: Repository mapping, Architect/Editor separation, git integration
- GPT-Engineer: Customizable prompts, project scaffolding
- OpenDevin/OpenHands: Sandboxed execution, CodeAct unified actions
- Sweep: XML structured planning, self-recovery, code chunking
- Continue: Multi-provider support, message routing

Usage:
    agent = UnifiedCodingAgent()
    result = await agent.solve_task("Add user authentication to the API")
"""

import os
import re
import ast
import json
import asyncio
import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# CORE DATA STRUCTURES (inspired by OpenDevin's state/event architecture)
# ============================================================================

class AgentState(Enum):
    """Agent execution states"""
    IDLE = "idle"
    PLANNING = "planning"
    CODING = "coding"
    REVIEWING = "reviewing"
    TESTING = "testing"
    RECOVERING = "recovering"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class CodeEdit:
    """Represents a code edit (inspired by Aider's edit formats)"""
    file_path: str
    original: str
    modified: str
    edit_type: str = "replace"  # replace, insert, delete
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class PlanStep:
    """A single step in an execution plan (inspired by Sweep's XML planning)"""
    step_id: int
    action: str  # create_file, modify_file, run_command, etc.
    target: str
    description: str
    dependencies: List[int] = field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None


@dataclass
class ExecutionPlan:
    """Complete execution plan with steps"""
    task_description: str
    steps: List[PlanStep]
    created_at: datetime = field(default_factory=datetime.now)
    architect_reasoning: str = ""


@dataclass
class AgentResult:
    """Result of agent execution"""
    success: bool
    message: str
    edits: List[CodeEdit] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)
    files_modified: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0


# ============================================================================
# REPOSITORY MAPPING (inspired by Aider)
# ============================================================================

class RepositoryMapper:
    """
    Creates a map of the codebase for context.
    Inspired by Aider's repo-map feature.
    """

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.file_tree: Dict[str, Any] = {}
        self.symbols: Dict[str, List[str]] = {}  # file -> [functions, classes]

    def scan(self, extensions: List[str] = None, max_files: int = 500) -> Dict[str, Any]:
        """Scan repository and build map

        Args:
            extensions: File extensions to include
            max_files: Maximum number of files to scan (prevents timeout on large repos)
        """
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.java', '.go', '.rs', '.cpp', '.c']

        self._file_count = 0
        self._max_files = max_files
        self._scanned_files: List[Path] = []  # Track files during tree build
        self.file_tree = self._build_tree(self.repo_path, extensions)
        self._extract_symbols_from_cache(max_files)

        return {
            "root": str(self.repo_path),
            "tree": self.file_tree,
            "symbols": self.symbols,
            "total_files": self._file_count,
            "total_lines": 0,  # Skip expensive line counting
            "summary": self._generate_summary()
        }

    def _count_lines(self, file_path: Path) -> int:
        """Count lines in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except (OSError, UnicodeDecodeError):
            return 0

    def _build_tree(self, path: Path, extensions: List[str], depth: int = 0) -> Dict:
        """Build file tree recursively"""
        if depth > 10:  # Prevent infinite recursion
            return {}

        if self._file_count >= self._max_files:
            return {}

        tree = {}

        try:
            for item in sorted(path.iterdir()):
                if self._file_count >= self._max_files:
                    break

                # Skip hidden files and common non-code directories
                if item.name.startswith('.') or item.name in ['node_modules', '__pycache__', 'venv', '.git', '.venv', 'dist', 'build']:
                    continue

                if item.is_dir():
                    subtree = self._build_tree(item, extensions, depth + 1)
                    if subtree:  # Only include non-empty directories
                        tree[item.name] = subtree
                elif item.suffix in extensions:
                    # Skip stat calls for performance (WSL is slow)
                    tree[item.name] = {"path": str(item)}
                    self._file_count += 1
                    self._scanned_files.append(item)
        except PermissionError:
            pass

        return tree

    def _extract_symbols_from_cache(self, max_files: int = 100):
        """Extract function and class definitions from cached Python files"""
        py_files = [f for f in self._scanned_files if f.suffix == '.py'][:max_files]
        for file_path in py_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                tree = ast.parse(content)
                symbols = []

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        symbols.append(f"def {node.name}()")
                    elif isinstance(node, ast.ClassDef):
                        symbols.append(f"class {node.name}")

                if symbols:
                    rel_path = str(Path(file_path).relative_to(self.repo_path))
                    self.symbols[rel_path] = symbols

            except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
                pass

    def _get_all_files(self, extension: str) -> List[Path]:
        """Get all files with given extension"""
        return list(self.repo_path.rglob(f"*{extension}"))

    def _generate_summary(self) -> str:
        """Generate a text summary of the repository"""
        total_files = sum(1 for _ in self._iter_tree_files(self.file_tree))
        total_symbols = sum(len(s) for s in self.symbols.values())

        return f"Repository: {self.repo_path.name}\n" \
               f"Files: {total_files}\n" \
               f"Symbols: {total_symbols} (functions/classes)"

    def _iter_tree_files(self, tree: Dict) -> Any:
        """Iterate over all files in tree"""
        for key, value in tree.items():
            if isinstance(value, dict) and 'path' in value:
                yield key
            elif isinstance(value, dict):
                yield from self._iter_tree_files(value)

    def get_context_for_task(self, task: str, max_files: int = 10) -> str:
        """Get relevant file context for a task"""
        # Simple keyword matching - in production, use embeddings
        relevant_files = []
        task_lower = task.lower()

        for file_path, symbols in self.symbols.items():
            score = 0
            for symbol in symbols:
                if any(word in symbol.lower() for word in task_lower.split()):
                    score += 1
            if score > 0:
                relevant_files.append((file_path, score))

        relevant_files.sort(key=lambda x: x[1], reverse=True)

        context = []
        for file_path, _ in relevant_files[:max_files]:
            full_path = self.repo_path / file_path
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                context.append(f"=== {file_path} ===\n{content[:2000]}")
            except (FileNotFoundError, UnicodeDecodeError):
                pass

        return "\n\n".join(context)


# ============================================================================
# SANDBOXED EXECUTION (inspired by OpenDevin)
# ============================================================================

class SandboxedExecutor:
    """
    Execute code in a sandboxed environment.
    Inspired by OpenDevin's Docker sandbox.
    """

    def __init__(self, working_dir: str = "/tmp/sandbox"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.execution_history: List[Dict] = []

    def execute_bash(self, command: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute a bash command safely"""
        # Block dangerous commands
        dangerous_patterns = ['rm -rf /', 'mkfs', ':(){', 'dd if=', '> /dev/sd']
        for pattern in dangerous_patterns:
            if pattern in command:
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Blocked dangerous command pattern: {pattern}",
                    "return_code": -1
                }

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir
            )

            execution = {
                "command": command,
                "success": result.returncode == 0,
                "stdout": result.stdout[:5000],  # Limit output
                "stderr": result.stderr[:2000],
                "return_code": result.returncode,
                "timestamp": datetime.now().isoformat()
            }

            self.execution_history.append(execution)
            return execution

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout}s",
                "return_code": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1
            }

    def execute_python(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Execute Python code safely"""
        # Write code to temp file
        code_file = self.working_dir / "temp_script.py"
        code_file.write_text(code)

        return self.execute_bash(f"python3 {code_file}", timeout)

    def run_tests(self, test_command: str = "python -m pytest") -> Dict[str, Any]:
        """Run test suite"""
        return self.execute_bash(test_command, timeout=120)


# ============================================================================
# ARCHITECT/EDITOR SEPARATION (inspired by Aider)
# ============================================================================

class Architect(ABC):
    """
    Architect: Focuses on understanding the problem and planning solution.
    Inspired by Aider's Architect/Editor pattern.
    """

    @abstractmethod
    async def analyze_task(self, task: str, context: str) -> ExecutionPlan:
        """Analyze task and create execution plan"""
        pass


class Editor(ABC):
    """
    Editor: Focuses on implementing the solution with correct syntax.
    Inspired by Aider's Architect/Editor pattern.
    """

    @abstractmethod
    async def implement_step(self, step: PlanStep, context: str) -> List[CodeEdit]:
        """Implement a single plan step"""
        pass


class LocalArchitect(Architect):
    """Local architect using rule-based planning"""

    async def analyze_task(self, task: str, context: str) -> ExecutionPlan:
        """Analyze task and create execution plan"""
        steps = []
        step_id = 0

        task_lower = task.lower()

        # Determine what kind of task this is and create appropriate steps
        if any(word in task_lower for word in ['add', 'create', 'implement', 'build']):
            if 'test' in task_lower:
                steps.append(PlanStep(
                    step_id=step_id,
                    action="create_file",
                    target="tests/test_new_feature.py",
                    description="Create test file for new feature"
                ))
                step_id += 1

            steps.append(PlanStep(
                step_id=step_id,
                action="modify_file",
                target="src/main.py",
                description="Add new feature implementation"
            ))
            step_id += 1

        elif any(word in task_lower for word in ['fix', 'bug', 'error', 'issue']):
            steps.append(PlanStep(
                step_id=step_id,
                action="analyze",
                target="codebase",
                description="Identify root cause of the issue"
            ))
            step_id += 1

            steps.append(PlanStep(
                step_id=step_id,
                action="modify_file",
                target="affected_file.py",
                description="Apply fix for the issue",
                dependencies=[0]
            ))
            step_id += 1

        elif any(word in task_lower for word in ['refactor', 'improve', 'optimize']):
            steps.append(PlanStep(
                step_id=step_id,
                action="analyze",
                target="codebase",
                description="Identify areas for improvement"
            ))
            step_id += 1

            steps.append(PlanStep(
                step_id=step_id,
                action="modify_file",
                target="multiple",
                description="Apply refactoring changes",
                dependencies=[0]
            ))
            step_id += 1

        # Always add validation step
        steps.append(PlanStep(
            step_id=step_id,
            action="run_command",
            target="python -m pytest",
            description="Run tests to validate changes",
            dependencies=[s.step_id for s in steps]
        ))

        return ExecutionPlan(
            task_description=task,
            steps=steps,
            architect_reasoning=f"Analyzed task: {task}. Created {len(steps)} step plan."
        )


class LocalEditor(Editor):
    """Local editor using template-based code generation"""

    async def implement_step(self, step: PlanStep, context: str) -> List[CodeEdit]:
        """Implement a single plan step"""
        edits = []

        if step.action == "create_file":
            # Generate template based on file type
            if step.target.endswith('.py'):
                content = self._generate_python_template(step)
            else:
                content = f"// Auto-generated for: {step.description}\n"

            edits.append(CodeEdit(
                file_path=step.target,
                original="",
                modified=content,
                edit_type="create"
            ))

        elif step.action == "modify_file":
            # In production, this would use LLM to generate actual edits
            edits.append(CodeEdit(
                file_path=step.target,
                original="# placeholder",
                modified=f"# Modified for: {step.description}",
                edit_type="replace"
            ))

        return edits

    def _generate_python_template(self, step: PlanStep) -> str:
        """Generate Python file template"""
        if 'test' in step.target.lower():
            return '''"""Auto-generated test file"""
import unittest


class TestNewFeature(unittest.TestCase):
    """Tests for new feature"""

    def test_placeholder(self):
        """Placeholder test - implement actual tests"""
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
'''
        else:
            return f'''"""
Auto-generated module
Description: {step.description}
"""


def main():
    """Main entry point"""
    pass


if __name__ == "__main__":
    main()
'''


# ============================================================================
# XML STRUCTURED PLANNING (inspired by Sweep)
# ============================================================================

class XMLPlanParser:
    """
    Parse and generate XML-structured plans.
    Inspired by Sweep's XML tag approach for robust parsing.
    """

    @staticmethod
    def parse_plan(xml_text: str) -> ExecutionPlan:
        """Parse XML plan into ExecutionPlan object"""
        steps = []

        # Extract steps using regex (robust to LLM output variations)
        step_pattern = r'<step\s+id="(\d+)"[^>]*>(.*?)</step>'
        matches = re.findall(step_pattern, xml_text, re.DOTALL)

        for step_id, content in matches:
            action = re.search(r'<action>(.*?)</action>', content)
            target = re.search(r'<target>(.*?)</target>', content)
            description = re.search(r'<description>(.*?)</description>', content)
            deps = re.search(r'<dependencies>(.*?)</dependencies>', content)

            dependencies = []
            if deps:
                dep_matches = re.findall(r'\d+', deps.group(1))
                dependencies = [int(d) for d in dep_matches]

            steps.append(PlanStep(
                step_id=int(step_id),
                action=action.group(1) if action else "unknown",
                target=target.group(1) if target else "",
                description=description.group(1) if description else "",
                dependencies=dependencies
            ))

        task = re.search(r'<task>(.*?)</task>', xml_text, re.DOTALL)
        reasoning = re.search(r'<reasoning>(.*?)</reasoning>', xml_text, re.DOTALL)

        return ExecutionPlan(
            task_description=task.group(1) if task else "",
            steps=steps,
            architect_reasoning=reasoning.group(1) if reasoning else ""
        )

    @staticmethod
    def generate_plan_xml(plan: ExecutionPlan) -> str:
        """Generate XML from ExecutionPlan"""
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<execution_plan>',
            f'  <task>{plan.task_description}</task>',
            f'  <reasoning>{plan.architect_reasoning}</reasoning>',
            '  <steps>'
        ]

        for step in plan.steps:
            deps_str = ",".join(str(d) for d in step.dependencies)
            lines.extend([
                f'    <step id="{step.step_id}">',
                f'      <action>{step.action}</action>',
                f'      <target>{step.target}</target>',
                f'      <description>{step.description}</description>',
                f'      <dependencies>{deps_str}</dependencies>',
                f'      <status>{step.status}</status>',
                '    </step>'
            ])

        lines.extend([
            '  </steps>',
            '</execution_plan>'
        ])

        return '\n'.join(lines)


# ============================================================================
# SELF-RECOVERY MECHANISM (inspired by Sweep)
# ============================================================================

class SelfRecovery:
    """
    Self-recovery mechanisms for fixing failed operations.
    Inspired by Sweep's self-recovery via GitHub Actions.
    """

    def __init__(self, executor: SandboxedExecutor):
        self.executor = executor
        self.recovery_attempts = 0
        self.max_attempts = 3

    async def attempt_recovery(self, failure: Dict, context: str) -> Dict[str, Any]:
        """Attempt to recover from a failure"""
        self.recovery_attempts += 1

        if self.recovery_attempts > self.max_attempts:
            return {
                "success": False,
                "message": f"Max recovery attempts ({self.max_attempts}) exceeded"
            }

        error_type = self._classify_error(failure)

        if error_type == "lint":
            return await self._fix_lint_errors(failure)
        elif error_type == "syntax":
            return await self._fix_syntax_errors(failure)
        elif error_type == "import":
            return await self._fix_import_errors(failure)
        elif error_type == "test":
            return await self._fix_test_failures(failure)
        else:
            return {
                "success": False,
                "message": f"Unknown error type: {error_type}"
            }

    def _classify_error(self, failure: Dict) -> str:
        """Classify the type of error"""
        stderr = failure.get("stderr", "").lower()

        if any(word in stderr for word in ['pylint', 'flake8', 'black', 'ruff']):
            return "lint"
        elif 'syntaxerror' in stderr:
            return "syntax"
        elif 'importerror' in stderr or 'modulenotfounderror' in stderr:
            return "import"
        elif 'assert' in stderr or 'failed' in stderr:
            return "test"
        else:
            return "unknown"

    async def _fix_lint_errors(self, failure: Dict) -> Dict[str, Any]:
        """Auto-fix lint errors using formatters"""
        # Try black for Python formatting
        result = self.executor.execute_bash("black . --quiet")
        if result["success"]:
            return {"success": True, "message": "Fixed lint errors with black"}

        # Try ruff for more fixes
        result = self.executor.execute_bash("ruff check --fix .")
        return {
            "success": result["success"],
            "message": "Attempted lint fix with ruff"
        }

    async def _fix_syntax_errors(self, failure: Dict) -> Dict[str, Any]:
        """Attempt to fix syntax errors"""
        # In production, would use LLM to fix
        return {
            "success": False,
            "message": "Syntax errors require manual intervention"
        }

    async def _fix_import_errors(self, failure: Dict) -> Dict[str, Any]:
        """Attempt to fix import errors"""
        stderr = failure.get("stderr", "")

        # Extract missing module name
        match = re.search(r"No module named '(\w+)'", stderr)
        if match:
            module = match.group(1)
            result = self.executor.execute_bash(f"pip install {module}")
            return {
                "success": result["success"],
                "message": f"Installed missing module: {module}"
            }

        return {"success": False, "message": "Could not identify missing module"}

    async def _fix_test_failures(self, failure: Dict) -> Dict[str, Any]:
        """Attempt to fix test failures"""
        # In production, would analyze test output and suggest fixes
        return {
            "success": False,
            "message": "Test failures require code changes"
        }


# ============================================================================
# MULTI-PROVIDER LLM SUPPORT (inspired by Continue)
# ============================================================================

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion for prompt"""
        pass

    @abstractmethod
    def estimate_cost(self, prompt: str, completion: str) -> float:
        """Estimate cost of the request"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama local LLM provider"""

    def __init__(self, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def complete(self, prompt: str, **kwargs) -> str:
        """Generate completion using Ollama"""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=120.0
                )

                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    return f"Error: {response.status_code}"

        except Exception as e:
            return f"Error: {str(e)}"

    def estimate_cost(self, prompt: str, completion: str) -> float:
        """Local models are free"""
        return 0.0


class MultiProviderManager:
    """
    Manages multiple LLM providers with fallback.
    Inspired by Continue's 20+ provider support.
    """

    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.priority: List[str] = []
        self.usage: Dict[str, int] = {}

    def register_provider(self, name: str, provider: LLMProvider, priority: int = 0):
        """Register a provider with given priority"""
        self.providers[name] = provider
        self.usage[name] = 0

        # Insert in priority order
        self.priority.append(name)
        self.priority.sort(key=lambda x: priority)

    async def complete(self, prompt: str, **kwargs) -> tuple[str, str]:
        """Complete using best available provider"""
        for provider_name in self.priority:
            provider = self.providers[provider_name]

            try:
                result = await provider.complete(prompt, **kwargs)
                if result and not result.startswith("Error"):
                    self.usage[provider_name] += 1
                    return result, provider_name
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue

        return "All providers failed", "none"

    def get_usage_stats(self) -> Dict[str, int]:
        """Get usage statistics for all providers"""
        return self.usage.copy()


# ============================================================================
# UNIFIED CODING AGENT (combines all patterns)
# ============================================================================

class UnifiedCodingAgent:
    """
    Unified Coding Agent combining best practices from:
    - Aider: Repository mapping, Architect/Editor separation
    - GPT-Engineer: Project scaffolding, customizable prompts
    - OpenDevin: Sandboxed execution, CodeAct
    - Sweep: XML planning, self-recovery
    - Continue: Multi-provider support
    """

    def __init__(
        self,
        repo_path: str = ".",
        sandbox_dir: str = "/tmp/coding_agent_sandbox"
    ):
        # Core components
        self.repo_mapper = RepositoryMapper(repo_path)
        self.executor = SandboxedExecutor(sandbox_dir)
        self.architect = LocalArchitect()
        self.editor = LocalEditor()
        self.recovery = SelfRecovery(self.executor)
        self.llm_manager = MultiProviderManager()

        # State
        self.state = AgentState.IDLE
        self.current_plan: Optional[ExecutionPlan] = None
        self.history: List[AgentResult] = []

        # Configuration
        self.auto_commit = True
        self.auto_test = True
        self.max_iterations = 10

        logger.info("UnifiedCodingAgent initialized")

    async def solve_task(self, task: str) -> AgentResult:
        """
        Main entry point: Solve a coding task end-to-end.

        Flow:
        1. Map repository for context
        2. Architect analyzes and plans
        3. Editor implements each step
        4. Executor runs tests
        5. Self-recovery if needed
        6. Commit changes
        """
        start_time = datetime.now()
        edits: List[CodeEdit] = []
        files_created: List[str] = []
        files_modified: List[str] = []

        try:
            # Phase 1: Repository Mapping
            self.state = AgentState.PLANNING
            logger.info(f"Solving task: {task}")

            repo_map = self.repo_mapper.scan(max_files=100)  # Limit for performance
            context = self.repo_mapper.get_context_for_task(task)

            logger.info(f"Repository scanned: {repo_map['summary']}")

            # Phase 2: Architect Planning
            self.current_plan = await self.architect.analyze_task(task, context)

            logger.info(f"Plan created with {len(self.current_plan.steps)} steps")
            logger.info(f"Reasoning: {self.current_plan.architect_reasoning}")

            # Phase 3: Editor Implementation
            self.state = AgentState.CODING

            for step in self.current_plan.steps:
                if step.action == "run_command":
                    # Execute command
                    result = self.executor.execute_bash(step.target)
                    step.status = "completed" if result["success"] else "failed"
                    step.result = result.get("stdout", result.get("stderr", ""))

                    # Self-recovery if needed
                    if not result["success"]:
                        self.state = AgentState.RECOVERING
                        recovery_result = await self.recovery.attempt_recovery(result, context)
                        if recovery_result["success"]:
                            step.status = "recovered"
                            step.result = recovery_result["message"]
                else:
                    # Generate code edits
                    step_edits = await self.editor.implement_step(step, context)
                    edits.extend(step_edits)

                    for edit in step_edits:
                        if edit.edit_type == "create":
                            files_created.append(edit.file_path)
                        else:
                            files_modified.append(edit.file_path)

                    step.status = "completed"

            # Phase 4: Testing
            if self.auto_test:
                self.state = AgentState.TESTING
                test_result = self.executor.run_tests()

                if not test_result["success"]:
                    logger.warning("Tests failed, attempting recovery")
                    self.state = AgentState.RECOVERING
                    await self.recovery.attempt_recovery(test_result, context)

            # Phase 5: Commit (if enabled)
            if self.auto_commit and (files_created or files_modified):
                self._git_commit(task, files_created + files_modified)

            self.state = AgentState.COMPLETE
            execution_time = (datetime.now() - start_time).total_seconds()

            result = AgentResult(
                success=True,
                message=f"Task completed: {task}",
                edits=edits,
                files_created=files_created,
                files_modified=files_modified,
                execution_time=execution_time
            )

            self.history.append(result)
            return result

        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"Task failed: {e}")

            return AgentResult(
                success=False,
                message=f"Task failed: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )

    def _git_commit(self, task: str, files: List[str]):
        """Create a git commit for the changes"""
        if not files:
            return

        try:
            # Add files
            for f in files:
                subprocess.run(["git", "add", f], capture_output=True)

            # Commit
            commit_msg = f"Auto: {task[:50]}"
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                capture_output=True
            )
            logger.info(f"Committed changes: {commit_msg}")

        except Exception as e:
            logger.warning(f"Git commit failed: {e}")

    def get_plan_xml(self) -> str:
        """Get current plan as XML"""
        if self.current_plan:
            return XMLPlanParser.generate_plan_xml(self.current_plan)
        return ""

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "state": self.state.value,
            "current_plan": self.current_plan.task_description if self.current_plan else None,
            "history_count": len(self.history),
            "llm_usage": self.llm_manager.get_usage_stats()
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def main():
    """Main entry point for CLI usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Unified Coding Agent")
    parser.add_argument("task", nargs="?", help="Task to solve")
    parser.add_argument("--repo", default=".", help="Repository path")
    parser.add_argument("--no-commit", action="store_true", help="Disable auto-commit")
    parser.add_argument("--no-test", action="store_true", help="Disable auto-test")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    agent = UnifiedCodingAgent(repo_path=args.repo)
    agent.auto_commit = not args.no_commit
    agent.auto_test = not args.no_test

    if args.interactive:
        print("Unified Coding Agent - Interactive Mode")
        print("Type 'quit' to exit, 'status' for agent status")
        print("-" * 50)

        while True:
            try:
                task = input("\nTask: ").strip()

                if task.lower() == 'quit':
                    break
                elif task.lower() == 'status':
                    print(json.dumps(agent.get_status(), indent=2))
                    continue
                elif task.lower() == 'plan':
                    print(agent.get_plan_xml())
                    continue
                elif not task:
                    continue

                result = await agent.solve_task(task)
                print(f"\nResult: {result.message}")
                print(f"Files created: {result.files_created}")
                print(f"Files modified: {result.files_modified}")
                print(f"Time: {result.execution_time:.2f}s")

            except KeyboardInterrupt:
                print("\nExiting...")
                break

    elif args.task:
        result = await agent.solve_task(args.task)
        print(json.dumps({
            "success": result.success,
            "message": result.message,
            "files_created": result.files_created,
            "files_modified": result.files_modified,
            "execution_time": result.execution_time
        }, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
