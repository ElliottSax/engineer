#!/usr/bin/env python3
"""
Multi-Provider Continuous Training for Autocoder

Uses multiple FREE/cheap providers:
1. DeepSeek ($0.14/M tokens) - Primary for quality
2. GitHub Models (FREE) - GPT-4o-mini
3. HuggingFace (FREE) - Inference API
4. Local Ollama (FREE) - If available

Workers run continuously, falling back between providers.
"""

import os
import json
import asyncio
import random
import hashlib
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import requests
from concurrent.futures import ThreadPoolExecutor

from logging.handlers import RotatingFileHandler

# Setup logging with rotation (10MB max, 5 backups)
log_format = '%(asctime)s [%(name)s] %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)

# Add rotating file handler
_file_handler = RotatingFileHandler(
    'multi_provider_training.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
_file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(_file_handler)


# =============================================================================
# PROVIDER CLIENTS
# =============================================================================

class LLMProvider(ABC):
    """Base class for LLM providers"""

    name: str = "base"
    cost_per_million: float = 0.0

    @abstractmethod
    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> Optional[str]:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass


class DeepSeekProvider(LLMProvider):
    """DeepSeek API - $0.14/M tokens"""

    name = "deepseek"
    cost_per_million = 0.14

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.tokens_used = 0

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> Optional[str]:
        if not self.api_key:
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": "deepseek-chat",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                self.tokens_used += result.get("usage", {}).get("total_tokens", 0)
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"DeepSeek error: {e}")

        return None


class GitHubModelsProvider(LLMProvider):
    """GitHub Models API - FREE"""

    name = "github"
    cost_per_million = 0.0

    def __init__(self):
        self.token = os.getenv("GITHUB_TOKEN")
        self.url = "https://models.inference.ai.azure.com/chat/completions"
        self.requests_made = 0

    def is_available(self) -> bool:
        return bool(self.token)

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> Optional[str]:
        if not self.token:
            return None

        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        data = {
            "model": "gpt-4o-mini",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=60)

            if response.status_code == 200:
                self.requests_made += 1
                result = response.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"GitHub Models error: {e}")

        return None


class HuggingFaceProvider(LLMProvider):
    """HuggingFace Inference API - FREE tier"""

    name = "huggingface"
    cost_per_million = 0.0

    MODELS = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Llama-2-70b-chat-hf",
        "codellama/CodeLlama-34b-Instruct-hf",
    ]

    def __init__(self):
        self.token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        self.requests_made = 0
        self.current_model_idx = 0

    def is_available(self) -> bool:
        return bool(self.token)

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> Optional[str]:
        if not self.token:
            return None

        headers = {"Authorization": f"Bearer {self.token}"}

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        # Try each model until one works
        for _ in range(len(self.MODELS)):
            model = self.MODELS[self.current_model_idx]
            url = f"https://api-inference.huggingface.co/models/{model}"

            data = {
                "inputs": full_prompt,
                "parameters": {"max_new_tokens": max_tokens, "temperature": 0.7}
            }

            try:
                response = requests.post(url, headers=headers, json=data, timeout=60)

                if response.status_code == 200:
                    self.requests_made += 1
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "")
                    return str(result)

                elif response.status_code == 503:
                    # Model loading, try next
                    self.current_model_idx = (self.current_model_idx + 1) % len(self.MODELS)
                    continue

            except Exception as e:
                logger.error(f"HuggingFace error: {e}")

            self.current_model_idx = (self.current_model_idx + 1) % len(self.MODELS)

        return None


class OllamaProvider(LLMProvider):
    """Local Ollama - FREE"""

    name = "ollama"
    cost_per_million = 0.0

    MODELS = ["qwen2.5-coder:7b", "codellama:7b", "llama3.2:3b"]

    def __init__(self):
        self._available = None
        self.requests_made = 0

    def is_available(self) -> bool:
        if self._available is None:
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                self._available = response.status_code == 200
            except (requests.RequestException, ConnectionError, TimeoutError):
                self._available = False
        return self._available

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> Optional[str]:
        if not self.is_available():
            return None

        for model in self.MODELS:
            try:
                data = {
                    "model": model,
                    "prompt": f"{system}\n\n{prompt}" if system else prompt,
                    "stream": False,
                    "options": {"num_predict": max_tokens}
                }

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=data,
                    timeout=120
                )

                if response.status_code == 200:
                    self.requests_made += 1
                    return response.json().get("response", "")

            except Exception as e:
                logger.debug(f"Ollama {model} error: {e}")
                continue

        return None


# =============================================================================
# MULTI-PROVIDER MANAGER
# =============================================================================

class ProviderManager:
    """Manages multiple LLM providers with fallback"""

    def __init__(self):
        self.providers: List[LLMProvider] = [
            GitHubModelsProvider(),   # FREE - try first
            HuggingFaceProvider(),    # FREE
            DeepSeekProvider(),       # Cheap
            OllamaProvider(),         # Local FREE
        ]

        self.available = [p for p in self.providers if p.is_available()]
        self.usage: Dict[str, int] = {p.name: 0 for p in self.providers}
        self.total_cost = 0.0

        logger.info(f"Available providers: {[p.name for p in self.available]}")

    async def complete(self, prompt: str, system: str = "", max_tokens: int = 1000) -> tuple[Optional[str], str]:
        """Complete using best available provider"""

        for provider in self.available:
            result = await provider.complete(prompt, system, max_tokens)

            if result:
                self.usage[provider.name] += 1
                self.total_cost += (max_tokens / 1_000_000) * provider.cost_per_million
                return result, provider.name

            # Small delay before trying next provider
            await asyncio.sleep(0.5)

        return None, "none"

    def get_stats(self) -> Dict:
        return {
            "usage": self.usage,
            "total_cost": f"${self.total_cost:.4f}",
            "available_providers": [p.name for p in self.available]
        }


# =============================================================================
# TRAINING WORKER
# =============================================================================

class MultiProviderWorker:
    """Training worker using multiple providers"""

    TASKS = [
        ("find_duplicates", "finds duplicate elements in a list and returns them sorted"),
        ("find_max", "finds the maximum value in a list"),
        ("find_min", "finds the minimum value in a list"),
        ("is_palindrome", "checks if a string is a palindrome"),
        ("fibonacci", "returns the first n fibonacci numbers"),
        ("is_prime", "checks if a number is prime"),
        ("reverse_string", "reverses a string"),
        ("count_words", "counts the number of words in a string"),
        ("remove_duplicates", "removes duplicate elements from a list"),
        ("sum_list", "calculates the sum of all elements in a list"),
        # New tasks
        ("flatten_list", "flattens a nested list into a single list"),
        ("get_average", "calculates the average of numbers in a list"),
        ("filter_even", "filters and returns only even numbers from a list"),
        ("filter_odd", "filters and returns only odd numbers from a list"),
        ("get_unique", "returns unique elements preserving order"),
        ("calculate_power", "calculates base raised to the power of exponent"),
        ("calculate_gcd", "calculates the greatest common divisor of two numbers"),
        ("is_anagram", "checks if two strings are anagrams of each other"),
        ("binary_search", "performs binary search on a sorted array"),
        ("capitalize_words", "capitalizes the first letter of each word in a string"),
    ]

    TEST_CASES = {
        "find_duplicates": [
            {"args": [[1, 2, 3, 2, 4, 3]], "expected": [2, 3]},
            {"args": [[1, 2, 3]], "expected": []},
        ],
        "find_max": [
            {"args": [[1, 5, 3, 9, 2]], "expected": 9},
            {"args": [[42]], "expected": 42},
        ],
        "find_min": [
            {"args": [[1, 5, 3, 9, 2]], "expected": 1},
            {"args": [[42]], "expected": 42},
        ],
        "is_palindrome": [
            {"args": ["racecar"], "expected": True},
            {"args": ["hello"], "expected": False},
        ],
        "fibonacci": [
            {"args": [5], "expected": [0, 1, 1, 2, 3]},
            {"args": [1], "expected": [0]},
        ],
        "is_prime": [
            {"args": [7], "expected": True},
            {"args": [4], "expected": False},
        ],
        "reverse_string": [
            {"args": ["hello"], "expected": "olleh"},
            {"args": ["a"], "expected": "a"},
        ],
        "count_words": [
            {"args": ["hello world"], "expected": 2},
            {"args": ["one"], "expected": 1},
        ],
        "remove_duplicates": [
            {"args": [[1, 2, 2, 3]], "expected": [1, 2, 3]},
            {"args": [[]], "expected": []},
        ],
        "sum_list": [
            {"args": [[1, 2, 3, 4, 5]], "expected": 15},
            {"args": [[]], "expected": 0},
        ],
        # New test cases
        "flatten_list": [
            {"args": [[[1, 2], [3, [4, 5]]]], "expected": [1, 2, 3, 4, 5]},
            {"args": [[[1], [2], [3]]], "expected": [1, 2, 3]},
        ],
        "get_average": [
            {"args": [[1, 2, 3, 4, 5]], "expected": 3.0},
            {"args": [[10, 20]], "expected": 15.0},
        ],
        "filter_even": [
            {"args": [[1, 2, 3, 4, 5, 6]], "expected": [2, 4, 6]},
            {"args": [[1, 3, 5]], "expected": []},
        ],
        "filter_odd": [
            {"args": [[1, 2, 3, 4, 5, 6]], "expected": [1, 3, 5]},
            {"args": [[2, 4, 6]], "expected": []},
        ],
        "get_unique": [
            {"args": [[1, 2, 2, 3, 3, 3]], "expected": [1, 2, 3]},
            {"args": [[1, 1, 1]], "expected": [1]},
        ],
        "calculate_power": [
            {"args": [2, 3], "expected": 8},
            {"args": [5, 2], "expected": 25},
        ],
        "calculate_gcd": [
            {"args": [12, 8], "expected": 4},
            {"args": [17, 13], "expected": 1},
        ],
        "is_anagram": [
            {"args": ["listen", "silent"], "expected": True},
            {"args": ["hello", "world"], "expected": False},
        ],
        "binary_search": [
            {"args": [[1, 2, 3, 4, 5], 3], "expected": 2},
            {"args": [[1, 2, 3, 4, 5], 6], "expected": -1},
        ],
        "capitalize_words": [
            {"args": ["hello world"], "expected": "Hello World"},
            {"args": ["python programming"], "expected": "Python Programming"},
        ],
    }

    def __init__(self, worker_id: int = 0):
        self.worker_id = worker_id
        self.provider_manager = ProviderManager()

        self.iterations = 0
        self.successes = 0
        self.failures = 0

        self.output_dir = Path("training_output")
        self.output_dir.mkdir(exist_ok=True)

        self.running = False
        self.results: List[Dict] = []

    async def get_autocoder(self):
        """Get fresh autocoder instance"""
        import sys
        sys.path.insert(0, str(Path(__file__).parent))

        if 'unified_coding_agent' in sys.modules:
            del sys.modules['unified_coding_agent']

        from unified_coding_agent import UnifiedCodingAgent

        agent = UnifiedCodingAgent(repo_path='/tmp/training_workspace')
        agent.auto_commit = False
        agent.auto_test = False
        return agent

    async def run_iteration(self) -> Dict:
        """Run single training iteration"""

        self.iterations += 1

        # Pick random task
        func_name, description = random.choice(self.TASKS)
        task = f"Create a Python function called '{func_name}' that {description}"

        logger.info(f"[W{self.worker_id}] Iter {self.iterations}: {func_name}")

        # Run autocoder
        try:
            agent = await self.get_autocoder()
            result = await agent.solve_task(task)

            if not result.edits:
                self.failures += 1
                return {"success": False, "task": func_name, "error": "No code generated"}

            code = result.edits[0].modified

        except Exception as e:
            self.failures += 1
            return {"success": False, "task": func_name, "error": str(e)}

        # Test the code
        test_cases = self.TEST_CASES.get(func_name, [])
        passed, total = self._run_tests(code, func_name, test_cases)

        score = passed / max(1, total)

        if score >= 0.8:
            self.successes += 1
            logger.info(f"  ✅ {passed}/{total} tests passed")
        else:
            self.failures += 1
            logger.info(f"  ❌ {passed}/{total} tests passed")

        result_data = {
            "iteration": self.iterations,
            "task": func_name,
            "passed": passed,
            "total": total,
            "score": score,
            "success": score >= 0.8
        }

        self.results.append(result_data)
        return result_data

    def _run_tests(self, code: str, func_name: str, test_cases: List[Dict]) -> tuple[int, int]:
        """Run tests on generated code"""

        if not test_cases:
            return 0, 0

        passed = 0
        exec_globals = {}

        try:
            exec(code, exec_globals)
            func = exec_globals.get(func_name)

            if not func:
                return 0, len(test_cases)

            for test in test_cases:
                try:
                    actual = func(*test['args'])
                    if actual == test['expected']:
                        passed += 1
                except (TypeError, ValueError, IndexError, KeyError, RuntimeError) as e:
                    logger.debug(f"Test case failed: {e}")

        except Exception as e:
            logger.debug(f"Test execution error: {e}")

        return passed, len(test_cases)

    async def run_continuous(self, max_iterations: int = 50, delay: float = 1.0):
        """Run continuous training"""

        self.running = True
        logger.info(f"[W{self.worker_id}] Starting - {max_iterations} iterations")

        while self.running and self.iterations < max_iterations:
            try:
                await self.run_iteration()

                if self.iterations % 10 == 0:
                    self._save_checkpoint()

                await asyncio.sleep(delay)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"[W{self.worker_id}] Error: {e}")
                await asyncio.sleep(2)

        self._save_checkpoint()
        self._print_summary()

    def _save_checkpoint(self):
        """Save training state"""

        checkpoint = {
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "iterations": self.iterations,
            "successes": self.successes,
            "failures": self.failures,
            "success_rate": self.successes / max(1, self.iterations),
            "provider_stats": self.provider_manager.get_stats(),
            "results": self.results[-20:]  # Last 20 results
        }

        path = self.output_dir / f"worker_{self.worker_id}_state.json"
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _print_summary(self):
        """Print worker summary"""

        stats = self.provider_manager.get_stats()
        rate = self.successes / max(1, self.iterations)

        print(f"\n[Worker {self.worker_id}] Complete")
        print(f"  Iterations: {self.iterations}")
        print(f"  Success rate: {rate:.0%}")
        print(f"  Providers used: {stats['usage']}")
        print(f"  Cost: {stats['total_cost']}")


# =============================================================================
# MAIN
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Provider Training")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    print("=" * 60)
    print("MULTI-PROVIDER CONTINUOUS TRAINING")
    print("=" * 60)

    # Check available providers
    manager = ProviderManager()
    print(f"Available: {[p.name for p in manager.available]}")

    if not manager.available:
        print("\nERROR: No providers available!")
        print("Set one of:")
        print("  - GITHUB_TOKEN (FREE)")
        print("  - HF_TOKEN (FREE)")
        print("  - DEEPSEEK_API_KEY ($0.14/M)")
        return

    print(f"\nWorkers: {args.workers}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)

    # Create and run workers
    workers = [MultiProviderWorker(i) for i in range(args.workers)]

    tasks = [w.run_continuous(args.iterations, args.delay) for w in workers]

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        print("\nStopping...")

    # Final summary
    total_success = sum(w.successes for w in workers)
    total_iters = sum(w.iterations for w in workers)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {total_iters}")
    print(f"Total successes: {total_success} ({total_success/max(1,total_iters):.0%})")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
