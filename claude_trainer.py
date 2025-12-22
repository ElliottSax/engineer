#!/usr/bin/env python3
"""
Claude-Based Training Pipeline for Autocoder

Uses Claude to:
1. Generate synthetic training data (code examples, solutions, tests)
2. Evaluate autocoder outputs and provide corrections (RLHF-style)
3. Design progressively harder curriculum tasks
4. Continuously improve autocoder capabilities

Architecture:
    Claude API → Training Data → Autocoder → Output → Claude Evaluation → Feedback Loop
"""

import os
import json
import asyncio
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import random

# Optional imports for different providers
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    import requests as httpx
    HAS_HTTPX = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Difficulty(Enum):
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


class TaskCategory(Enum):
    CODE_GENERATION = "code_generation"
    BUG_FIXING = "bug_fixing"
    REFACTORING = "refactoring"
    CODE_REVIEW = "code_review"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"


@dataclass
class TrainingExample:
    """A single training example"""
    id: str
    task: str
    category: TaskCategory
    difficulty: Difficulty
    input_code: str = ""
    expected_output: str = ""
    explanation: str = ""
    test_cases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class EvaluationResult:
    """Result of Claude evaluating autocoder output"""
    example_id: str
    autocoder_output: str
    score: float  # 0.0 to 1.0
    feedback: str
    corrections: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    improved_solution: str = ""


@dataclass
class CurriculumLevel:
    """A level in the curriculum"""
    level: int
    name: str
    difficulty: Difficulty
    required_score: float  # Score needed to advance
    tasks: List[TrainingExample] = field(default_factory=list)
    skills_taught: List[str] = field(default_factory=list)


# =============================================================================
# CLAUDE API CLIENT
# =============================================================================

class ClaudeClient:
    """Client for Claude API interactions"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = "claude-sonnet-4-20250514"  # Use claude-sonnet-4-20250514 for training
        self.base_url = "https://api.anthropic.com/v1"

        if HAS_ANTHROPIC and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None

        self.total_tokens = 0
        self.total_cost = 0.0

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 4096) -> str:
        """Send a completion request to Claude"""

        if self.client:
            # Use official SDK
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system if system else "You are an expert software engineer and coding instructor.",
                    messages=[{"role": "user", "content": prompt}]
                )

                # Track usage
                self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
                self.total_cost += self._calculate_cost(response.usage)

                return response.content[0].text

            except Exception as e:
                logger.error(f"Claude API error: {e}")
                return f"Error: {e}"
        else:
            # Fallback to HTTP
            return await self._http_complete(prompt, system, max_tokens)

    async def _http_complete(self, prompt: str, system: str, max_tokens: int) -> str:
        """HTTP-based completion (fallback)"""
        if not self.api_key:
            return "Error: No API key configured"

        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system or "You are an expert software engineer.",
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        headers=headers,
                        json=data,
                        timeout=120.0
                    )
                    result = response.json()
            else:
                response = httpx.post(
                    f"{self.base_url}/messages",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                result = response.json()

            if "content" in result:
                return result["content"][0]["text"]
            else:
                return f"Error: {result.get('error', 'Unknown error')}"

        except Exception as e:
            logger.error(f"HTTP request error: {e}")
            return f"Error: {e}"

    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage"""
        # Claude Sonnet pricing (as of 2024)
        input_cost = (usage.input_tokens / 1_000_000) * 3.0
        output_cost = (usage.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

class SyntheticDataGenerator:
    """
    Uses Claude to generate high-quality training examples for autocoder.

    Generates:
    - Code generation tasks with solutions
    - Bug fixing challenges with buggy/fixed code pairs
    - Refactoring examples
    - Code review scenarios
    - Test writing tasks
    """

    def __init__(self, claude: ClaudeClient):
        self.claude = claude
        self.output_dir = Path("training_data")
        self.output_dir.mkdir(exist_ok=True)

    async def generate_examples(
        self,
        category: TaskCategory,
        difficulty: Difficulty,
        count: int = 10
    ) -> List[TrainingExample]:
        """Generate training examples for a specific category and difficulty"""

        logger.info(f"Generating {count} {category.value} examples at {difficulty.name} level")

        prompt = self._build_generation_prompt(category, difficulty, count)

        system = """You are an expert software engineering instructor creating training examples.
Generate high-quality, realistic coding challenges with complete solutions.
Each example should teach a specific concept or skill.
Output valid JSON that can be parsed."""

        response = await self.claude.complete(prompt, system)

        examples = self._parse_examples(response, category, difficulty)

        # Save to file
        self._save_examples(examples, category, difficulty)

        return examples

    def _build_generation_prompt(
        self,
        category: TaskCategory,
        difficulty: Difficulty,
        count: int
    ) -> str:
        """Build the prompt for generating examples"""

        category_details = {
            TaskCategory.CODE_GENERATION: {
                "description": "Generate code from natural language descriptions",
                "examples": ["implement a binary search", "create a REST API endpoint", "build a caching decorator"]
            },
            TaskCategory.BUG_FIXING: {
                "description": "Identify and fix bugs in code",
                "examples": ["off-by-one error", "null pointer exception", "race condition"]
            },
            TaskCategory.REFACTORING: {
                "description": "Improve code structure without changing behavior",
                "examples": ["extract method", "introduce design pattern", "reduce complexity"]
            },
            TaskCategory.CODE_REVIEW: {
                "description": "Review code and provide feedback",
                "examples": ["security review", "performance review", "style review"]
            },
            TaskCategory.TESTING: {
                "description": "Write tests for code",
                "examples": ["unit tests", "integration tests", "edge case tests"]
            },
            TaskCategory.DOCUMENTATION: {
                "description": "Write documentation for code",
                "examples": ["API docs", "README", "inline comments"]
            },
            TaskCategory.ARCHITECTURE: {
                "description": "Design system architecture",
                "examples": ["microservices design", "database schema", "API design"]
            },
            TaskCategory.OPTIMIZATION: {
                "description": "Optimize code for performance",
                "examples": ["algorithm optimization", "memory optimization", "query optimization"]
            }
        }

        difficulty_guidance = {
            Difficulty.BEGINNER: "Simple, single-concept tasks. Clear requirements. Basic Python/JS.",
            Difficulty.INTERMEDIATE: "Multi-step tasks. Some edge cases. Standard libraries.",
            Difficulty.ADVANCED: "Complex logic. Multiple components. Design decisions required.",
            Difficulty.EXPERT: "System-level thinking. Performance critical. Trade-offs.",
            Difficulty.MASTER: "Novel problems. Research-level. Cutting-edge techniques."
        }

        details = category_details[category]

        return f"""Generate {count} training examples for an AI coding assistant.

Category: {category.value}
Description: {details['description']}
Example topics: {', '.join(details['examples'])}

Difficulty: {difficulty.name}
Guidance: {difficulty_guidance[difficulty]}

For each example, provide:
1. A clear task description
2. Input code (if applicable)
3. Expected output/solution
4. Explanation of the solution
5. 2-3 test cases

Output as a JSON array with this structure:
```json
[
  {{
    "task": "Clear description of what to do",
    "input_code": "// Starting code if any",
    "expected_output": "// Complete solution",
    "explanation": "Why this solution works and what concepts it teaches",
    "test_cases": ["test case 1", "test case 2"]
  }}
]
```

Make examples realistic, practical, and educational. Focus on Python but include some JavaScript/TypeScript.
Ensure solutions are complete and correct."""

    def _parse_examples(
        self,
        response: str,
        category: TaskCategory,
        difficulty: Difficulty
    ) -> List[TrainingExample]:
        """Parse Claude's response into TrainingExample objects"""

        examples = []

        try:
            # Extract JSON from response
            json_start = response.find('[')
            json_end = response.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                data = json.loads(json_str)

                for i, item in enumerate(data):
                    example_id = hashlib.md5(
                        f"{category.value}_{difficulty.name}_{i}_{datetime.now().isoformat()}".encode()
                    ).hexdigest()[:12]

                    examples.append(TrainingExample(
                        id=example_id,
                        task=item.get("task", ""),
                        category=category,
                        difficulty=difficulty,
                        input_code=item.get("input_code", ""),
                        expected_output=item.get("expected_output", ""),
                        explanation=item.get("explanation", ""),
                        test_cases=item.get("test_cases", [])
                    ))

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse examples: {e}")

        return examples

    def _save_examples(
        self,
        examples: List[TrainingExample],
        category: TaskCategory,
        difficulty: Difficulty
    ):
        """Save examples to file"""

        filename = f"{category.value}_{difficulty.name.lower()}.json"
        filepath = self.output_dir / filename

        # Load existing or create new
        existing = []
        if filepath.exists():
            with open(filepath) as f:
                existing = json.load(f)

        # Add new examples
        for ex in examples:
            existing.append({
                **asdict(ex),
                "category": ex.category.value,
                "difficulty": ex.difficulty.name
            })

        with open(filepath, 'w') as f:
            json.dump(existing, f, indent=2)

        logger.info(f"Saved {len(examples)} examples to {filepath}")


# =============================================================================
# EVALUATION & FEEDBACK LOOP
# =============================================================================

class AutocoderEvaluator:
    """
    Uses Claude to evaluate autocoder's outputs and provide feedback.

    Implements RLHF-style evaluation:
    1. Autocoder generates a solution
    2. Claude evaluates the solution
    3. Claude provides corrections and improved solution
    4. Feedback is used to improve autocoder
    """

    def __init__(self, claude: ClaudeClient):
        self.claude = claude
        self.evaluations: List[EvaluationResult] = []
        self.evaluation_dir = Path("evaluations")
        self.evaluation_dir.mkdir(exist_ok=True)

    async def evaluate(
        self,
        example: TrainingExample,
        autocoder_output: str
    ) -> EvaluationResult:
        """Evaluate autocoder's output against expected solution"""

        prompt = f"""Evaluate this coding solution generated by an AI assistant.

## Task
{example.task}

## Input Code (if any)
```
{example.input_code}
```

## Expected Solution
```
{example.expected_output}
```

## AI Assistant's Solution
```
{autocoder_output}
```

## Evaluation Criteria
1. Correctness: Does it solve the problem correctly?
2. Code Quality: Is it clean, readable, and well-structured?
3. Efficiency: Is it reasonably efficient?
4. Best Practices: Does it follow Python/language best practices?
5. Completeness: Does it handle edge cases?

Provide your evaluation as JSON:
```json
{{
  "score": 0.0 to 1.0,
  "feedback": "Overall assessment",
  "corrections": ["specific issues to fix"],
  "strengths": ["what was done well"],
  "weaknesses": ["areas for improvement"],
  "improved_solution": "Your improved version of the solution"
}}
```"""

        system = """You are an expert code reviewer providing constructive feedback.
Be fair but thorough. Focus on teaching and improvement.
Always provide an improved solution that demonstrates best practices."""

        response = await self.claude.complete(prompt, system)

        result = self._parse_evaluation(example.id, autocoder_output, response)
        self.evaluations.append(result)

        return result

    def _parse_evaluation(
        self,
        example_id: str,
        autocoder_output: str,
        response: str
    ) -> EvaluationResult:
        """Parse Claude's evaluation response"""

        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                return EvaluationResult(
                    example_id=example_id,
                    autocoder_output=autocoder_output,
                    score=float(data.get("score", 0.5)),
                    feedback=data.get("feedback", ""),
                    corrections=data.get("corrections", []),
                    strengths=data.get("strengths", []),
                    weaknesses=data.get("weaknesses", []),
                    improved_solution=data.get("improved_solution", "")
                )

        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse evaluation: {e}")

        return EvaluationResult(
            example_id=example_id,
            autocoder_output=autocoder_output,
            score=0.5,
            feedback="Evaluation parsing failed"
        )

    async def batch_evaluate(
        self,
        examples: List[Tuple[TrainingExample, str]]
    ) -> List[EvaluationResult]:
        """Evaluate multiple autocoder outputs"""

        results = []
        for example, output in examples:
            result = await self.evaluate(example, output)
            results.append(result)
            await asyncio.sleep(0.5)  # Rate limiting

        return results

    def get_aggregate_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from evaluations"""

        if not self.evaluations:
            return {"count": 0}

        scores = [e.score for e in self.evaluations]

        return {
            "count": len(self.evaluations),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "passing_rate": sum(1 for s in scores if s >= 0.7) / len(scores),
            "common_weaknesses": self._get_common_issues("weaknesses"),
            "common_strengths": self._get_common_issues("strengths")
        }

    def _get_common_issues(self, field: str) -> List[str]:
        """Get most common issues from evaluations"""
        from collections import Counter

        all_issues = []
        for e in self.evaluations:
            all_issues.extend(getattr(e, field, []))

        counter = Counter(all_issues)
        return [issue for issue, _ in counter.most_common(5)]

    def save_evaluations(self):
        """Save all evaluations to file"""

        filepath = self.evaluation_dir / f"evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filepath, 'w') as f:
            json.dump([asdict(e) for e in self.evaluations], f, indent=2)

        logger.info(f"Saved {len(self.evaluations)} evaluations to {filepath}")


# =============================================================================
# CURRICULUM LEARNING SYSTEM
# =============================================================================

class CurriculumDesigner:
    """
    Uses Claude to design a progressive learning curriculum for autocoder.

    The curriculum:
    1. Starts with simple tasks
    2. Tracks mastery of concepts
    3. Introduces harder challenges as skills improve
    4. Adapts based on performance
    """

    def __init__(self, claude: ClaudeClient, data_generator: SyntheticDataGenerator):
        self.claude = claude
        self.data_generator = data_generator
        self.curriculum: List[CurriculumLevel] = []
        self.current_level = 0
        self.mastery: Dict[str, float] = {}  # skill -> mastery score

    async def design_curriculum(self, total_levels: int = 10) -> List[CurriculumLevel]:
        """Design a complete curriculum"""

        logger.info(f"Designing curriculum with {total_levels} levels")

        prompt = f"""Design a progressive coding curriculum with {total_levels} levels.

Each level should:
1. Build on skills from previous levels
2. Introduce 2-3 new concepts
3. Have increasing difficulty
4. Cover different aspects of software engineering

Output as JSON:
```json
[
  {{
    "level": 1,
    "name": "Level name",
    "difficulty": "BEGINNER|INTERMEDIATE|ADVANCED|EXPERT|MASTER",
    "required_score": 0.7,
    "skills_taught": ["skill1", "skill2"],
    "task_categories": ["code_generation", "bug_fixing"],
    "example_tasks": ["task description 1", "task description 2"]
  }}
]
```

Cover these areas progressively:
- Basic syntax and data structures
- Functions and classes
- Error handling
- Testing
- Design patterns
- System design
- Performance optimization
- Security
- Advanced architecture"""

        system = "You are an expert curriculum designer for software engineering education."

        response = await self.claude.complete(prompt, system)

        self.curriculum = self._parse_curriculum(response)

        # Generate initial tasks for each level
        for level in self.curriculum[:3]:  # Pre-generate first 3 levels
            await self._generate_level_tasks(level)

        return self.curriculum

    def _parse_curriculum(self, response: str) -> List[CurriculumLevel]:
        """Parse curriculum from Claude's response"""

        levels = []

        try:
            json_start = response.find('[')
            json_end = response.rfind(']') + 1

            if json_start >= 0 and json_end > json_start:
                data = json.loads(response[json_start:json_end])

                for item in data:
                    difficulty = Difficulty[item.get("difficulty", "BEGINNER")]

                    levels.append(CurriculumLevel(
                        level=item.get("level", len(levels) + 1),
                        name=item.get("name", f"Level {len(levels) + 1}"),
                        difficulty=difficulty,
                        required_score=item.get("required_score", 0.7),
                        skills_taught=item.get("skills_taught", [])
                    ))

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse curriculum: {e}")

            # Create default curriculum
            for i, diff in enumerate([Difficulty.BEGINNER, Difficulty.INTERMEDIATE,
                                       Difficulty.ADVANCED, Difficulty.EXPERT, Difficulty.MASTER]):
                levels.append(CurriculumLevel(
                    level=i + 1,
                    name=f"{diff.name.title()} Level",
                    difficulty=diff,
                    required_score=0.6 + (i * 0.05)
                ))

        return levels

    async def _generate_level_tasks(self, level: CurriculumLevel):
        """Generate tasks for a curriculum level"""

        # Generate examples for each category appropriate for this level
        categories = [TaskCategory.CODE_GENERATION, TaskCategory.BUG_FIXING]

        if level.difficulty.value >= 2:
            categories.append(TaskCategory.REFACTORING)
        if level.difficulty.value >= 3:
            categories.extend([TaskCategory.TESTING, TaskCategory.CODE_REVIEW])
        if level.difficulty.value >= 4:
            categories.append(TaskCategory.ARCHITECTURE)

        for category in categories:
            examples = await self.data_generator.generate_examples(
                category=category,
                difficulty=level.difficulty,
                count=3
            )
            level.tasks.extend(examples)

    def get_current_level(self) -> CurriculumLevel:
        """Get current curriculum level"""
        if self.current_level < len(self.curriculum):
            return self.curriculum[self.current_level]
        return self.curriculum[-1]

    def advance_level(self, score: float) -> bool:
        """Try to advance to next level based on score"""

        current = self.get_current_level()

        if score >= current.required_score:
            self.current_level = min(self.current_level + 1, len(self.curriculum) - 1)
            logger.info(f"Advanced to level {self.current_level + 1}: {self.get_current_level().name}")
            return True

        return False

    def update_mastery(self, skill: str, score: float):
        """Update mastery score for a skill"""

        current = self.mastery.get(skill, 0.0)
        # Exponential moving average
        self.mastery[skill] = 0.7 * current + 0.3 * score


# =============================================================================
# TRAINING PIPELINE
# =============================================================================

class ClaudeTrainingPipeline:
    """
    Complete training pipeline that orchestrates:
    1. Data generation
    2. Autocoder training/inference
    3. Evaluation
    4. Curriculum progression
    5. Continuous improvement
    """

    def __init__(self, api_key: Optional[str] = None):
        self.claude = ClaudeClient(api_key)
        self.data_generator = SyntheticDataGenerator(self.claude)
        self.evaluator = AutocoderEvaluator(self.claude)
        self.curriculum = CurriculumDesigner(self.claude, self.data_generator)

        self.training_history: List[Dict] = []
        self.output_dir = Path("training_output")
        self.output_dir.mkdir(exist_ok=True)

    async def initialize(self):
        """Initialize the training pipeline"""

        logger.info("Initializing Claude Training Pipeline...")

        # Design curriculum
        await self.curriculum.design_curriculum()

        logger.info(f"Curriculum created with {len(self.curriculum.curriculum)} levels")

    async def run_training_session(
        self,
        autocoder_fn,  # Function that takes task and returns solution
        num_iterations: int = 10
    ):
        """
        Run a training session.

        Args:
            autocoder_fn: Callable that takes a task string and returns a solution string
            num_iterations: Number of training iterations
        """

        logger.info(f"Starting training session with {num_iterations} iterations")

        for i in range(num_iterations):
            logger.info(f"\n{'='*60}\nIteration {i+1}/{num_iterations}\n{'='*60}")

            # Get current level
            level = self.curriculum.get_current_level()
            logger.info(f"Current level: {level.name} ({level.difficulty.name})")

            # Ensure we have tasks
            if not level.tasks:
                await self.curriculum._generate_level_tasks(level)

            # Select a random task
            task = random.choice(level.tasks)
            logger.info(f"Task: {task.task[:100]}...")

            # Get autocoder's solution
            try:
                autocoder_output = await autocoder_fn(task.task)
            except Exception as e:
                logger.error(f"Autocoder error: {e}")
                autocoder_output = f"Error: {e}"

            # Evaluate
            evaluation = await self.evaluator.evaluate(task, autocoder_output)

            logger.info(f"Score: {evaluation.score:.2f}")
            logger.info(f"Feedback: {evaluation.feedback[:200]}...")

            # Record history
            self.training_history.append({
                "iteration": i + 1,
                "level": level.level,
                "task_id": task.id,
                "score": evaluation.score,
                "feedback": evaluation.feedback
            })

            # Try to advance
            if evaluation.score >= level.required_score:
                self.curriculum.advance_level(evaluation.score)

            # Update mastery for skills
            for skill in level.skills_taught:
                self.curriculum.update_mastery(skill, evaluation.score)

            # Small delay between iterations
            await asyncio.sleep(1)

        # Save results
        self._save_session_results()

        return self.get_training_summary()

    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training session"""

        if not self.training_history:
            return {"status": "No training completed"}

        scores = [h["score"] for h in self.training_history]

        return {
            "total_iterations": len(self.training_history),
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "final_level": self.curriculum.current_level + 1,
            "mastery": self.curriculum.mastery,
            "evaluation_stats": self.evaluator.get_aggregate_stats(),
            "api_usage": {
                "total_tokens": self.claude.total_tokens,
                "total_cost": f"${self.claude.total_cost:.4f}"
            }
        }

    def _save_session_results(self):
        """Save training session results"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save training history
        history_file = self.output_dir / f"training_history_{timestamp}.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Save evaluations
        self.evaluator.save_evaluations()

        # Save summary
        summary_file = self.output_dir / f"training_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(self.get_training_summary(), f, indent=2)

        logger.info(f"Results saved to {self.output_dir}")


# =============================================================================
# INTEGRATION WITH AUTOCODER
# =============================================================================

async def create_autocoder_training_fn():
    """
    Create a training function that uses the existing autocoder.

    Returns a function that takes a task and returns autocoder's solution.
    """

    # Try to import autocoder components
    try:
        from unified_coding_agent import UnifiedCodingAgent

        agent = UnifiedCodingAgent(repo_path=".")
        agent.auto_commit = False
        agent.auto_test = False

        async def autocoder_fn(task: str) -> str:
            result = await agent.solve_task(task)
            if result.success:
                # Return the generated code/edits
                if result.edits:
                    return "\n".join(e.modified for e in result.edits)
                return result.message
            return f"Failed: {result.message}"

        return autocoder_fn

    except ImportError:
        logger.warning("UnifiedCodingAgent not available, using mock")

        async def mock_autocoder(task: str) -> str:
            return f"# Mock solution for: {task}\npass"

        return mock_autocoder


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """Main entry point for training"""

    print("=" * 70)
    print("CLAUDE-BASED AUTOCODER TRAINING PIPELINE")
    print("=" * 70)

    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nWARNING: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY=your-key")
        print("\nRunning in demo mode (no actual API calls)...")

    # Initialize pipeline
    pipeline = ClaudeTrainingPipeline(api_key)
    await pipeline.initialize()

    # Create autocoder function
    autocoder_fn = await create_autocoder_training_fn()

    # Run training
    print("\nStarting training session...")
    summary = await pipeline.run_training_session(
        autocoder_fn=autocoder_fn,
        num_iterations=5  # Start small
    )

    # Print summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
