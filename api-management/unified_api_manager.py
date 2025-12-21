#!/usr/bin/env python3
"""
Unified API Manager
Central interface for managing all API services with intelligent routing,
cost optimization, and fallback mechanisms
"""

import os
import sys
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import requests
import json
import time

# Add services directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config_loader import ConfigLoader
from services.huggingface_manager import HuggingFaceManager
from services.deepseek_manager import DeepSeekManager
from services.github_models_manager import GitHubModelsManager

# Optional import for OCI (requires oci package)
try:
    from services.oci_manager import OCIComputeManager
    OCI_AVAILABLE = True
except ImportError:
    OCI_AVAILABLE = False
    print("Note: OCI module not available. Install 'oci' package to enable Oracle Cloud features.")


class ModelProvider(Enum):
    """Available model providers"""
    DEEPSEEK = "deepseek"
    HUGGINGFACE = "huggingface"
    GITHUB = "github"  # GitHub Models
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"
    CLAUDE = "claude"
    SILICONFLOW = "siliconflow"
    REPLICATE = "replicate"
    LOCAL = "local"  # For self-hosted models


class TaskType(Enum):
    """Types of AI tasks"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    IMAGE_GENERATION = "image_generation"


class UnifiedAPIManager:
    """Central manager for all API services"""

    def __init__(self):
        """Initialize all API managers"""
        self.config = ConfigLoader()

        # Initialize individual managers
        self.huggingface = HuggingFaceManager(self.config.huggingface_token)
        self.deepseek = DeepSeekManager(self.config.deepseek_api_key)
        self.github_models = GitHubModelsManager(self.config.github_token)

        # Initialize OCI manager only if available
        if OCI_AVAILABLE:
            self.oci_manager = OCIComputeManager()
        else:
            self.oci_manager = None

        # Cost-optimized model routing (updated with working models)
        self.cost_routing = {
            TaskType.TEXT_GENERATION: [
                (ModelProvider.SILICONFLOW, "Qwen/Qwen2.5-7B-Instruct", 0.00),  # Free tier
                (ModelProvider.GROQ, "llama-3.1-8b-instant", 0.05),  # $0.05/1M tokens - FASTEST!
                (ModelProvider.GITHUB, "meta-llama-3.1-8b-instruct", 0.10),  # GitHub Models
                (ModelProvider.DEEPSEEK, "deepseek-chat", 0.14),  # $0.14/1M input
                (ModelProvider.GITHUB, "gpt-4o-mini", 0.15),  # Best quality/price
                (ModelProvider.GITHUB, "mistral-nemo", 0.15),  # Good alternative
                (ModelProvider.TOGETHER, "meta-llama/Llama-3-8b-chat-hf", 0.20),
            ],
            TaskType.CODE_GENERATION: [
                (ModelProvider.GROQ, "llama-3.1-8b-instant", 0.05),  # Fast & cheap
                (ModelProvider.DEEPSEEK, "deepseek-coder", 0.14),  # Specialized for code
                (ModelProvider.GITHUB, "gpt-4o-mini", 0.15),  # Excellent for code
                (ModelProvider.TOGETHER, "codellama/CodeLlama-34b-Instruct-hf", 0.40),
                (ModelProvider.GROQ, "llama-3.1-70b-versatile", 0.59),
                (ModelProvider.GITHUB, "meta-llama-3.1-70b-instruct", 0.90),
            ],
            TaskType.EMBEDDINGS: [
                (ModelProvider.HUGGINGFACE, "sentence-transformers/all-MiniLM-L6-v2", 0.00),
                (ModelProvider.TOGETHER, "togethercomputer/m2-bert-80M-8k-retrieval", 0.008),
                (ModelProvider.GITHUB, "text-embedding-3-small", 0.02),
            ],
            TaskType.CHAT: [
                (ModelProvider.SILICONFLOW, "deepseek-ai/DeepSeek-V2.5", 0.00),  # Free
                (ModelProvider.GROQ, "llama-3.1-8b-instant", 0.05),  # Fastest!
                (ModelProvider.GITHUB, "meta-llama-3.1-8b-instruct", 0.10),
                (ModelProvider.DEEPSEEK, "deepseek-chat", 0.14),
                (ModelProvider.GITHUB, "gpt-4o-mini", 0.15),  # Best for quality
                (ModelProvider.GITHUB, "mistral-nemo", 0.15),  # Good alternative
            ],
        }

        # Initialize API clients for other providers
        self._init_groq_client()
        self._init_together_client()
        self._init_openrouter_client()
        self._init_siliconflow_client()

    def _init_groq_client(self):
        """Initialize GROQ client"""
        self.groq_config = {
            "api_key": self.config.groq_api_key,
            "base_url": "https://api.groq.com/openai/v1",
            "models": {
                "llama-3.1-8b-instant": {"context": 128000, "cost": 0.05},
                "llama-3.1-70b-versatile": {"context": 128000, "cost": 0.59},
                "mixtral-8x7b-32768": {"context": 32768, "cost": 0.24},
                "gemma2-9b-it": {"context": 8192, "cost": 0.20},
            }
        }

    def _init_together_client(self):
        """Initialize Together AI client"""
        self.together_config = {
            "api_key": self.config.together_api_key,
            "base_url": "https://api.together.xyz/v1",
            "models": {
                "meta-llama/Llama-3-8b-chat-hf": {"cost": 0.20},
                "meta-llama/Llama-3-70b-chat-hf": {"cost": 0.90},
                "codellama/CodeLlama-34b-Instruct-hf": {"cost": 0.40},
                "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": {"cost": 0.60},
            }
        }

    def _init_openrouter_client(self):
        """Initialize OpenRouter client"""
        self.openrouter_config = {
            "api_key": self.config.openrouter_api_key,
            "base_url": "https://openrouter.ai/api/v1",
            "headers": {
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/unified-api-manager",
                "X-Title": "Unified API Manager"
            }
        }

    def _init_siliconflow_client(self):
        """Initialize SiliconFlow client (Chinese provider with free tier)"""
        self.siliconflow_config = {
            "api_key": self.config.siliconflow_api_key,
            "base_url": "https://api.siliconflow.cn/v1",
            "free_models": [
                "Qwen/Qwen2.5-7B-Instruct",
                "deepseek-ai/DeepSeek-V2.5",
                "THUDM/glm-4-9b-chat",
            ]
        }

    def generate_text(
        self,
        prompt: str,
        provider: Optional[ModelProvider] = None,
        model: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        use_cheapest: bool = True
    ) -> Dict[str, Any]:
        """Generate text using the most appropriate provider

        Args:
            prompt: Input prompt
            provider: Specific provider to use
            model: Specific model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cheapest: Automatically use cheapest available provider

        Returns:
            Response with generated text and metadata
        """
        if use_cheapest and not provider:
            provider, model, cost = self._get_cheapest_provider(TaskType.TEXT_GENERATION)

        if provider == ModelProvider.DEEPSEEK:
            response = self.deepseek.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model or "deepseek-chat",
                max_tokens=max_tokens,
                temperature=temperature
            )
            return {
                "text": response.get("choices", [{}])[0].get("message", {}).get("content", ""),
                "provider": provider.value,
                "model": model,
                "cost_estimate": self._estimate_cost(prompt, max_tokens, provider, model)
            }

        elif provider == ModelProvider.GROQ:
            return self._groq_generate(prompt, model, max_tokens, temperature)

        elif provider == ModelProvider.HUGGINGFACE:
            text = self.huggingface.text_generation(
                prompt=prompt,
                model=model or "microsoft/Phi-3-mini-4k-instruct",
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            return {
                "text": text,
                "provider": provider.value,
                "model": model,
                "cost_estimate": 0.0  # Free tier
            }

        elif provider == ModelProvider.GITHUB:
            response = self.github_models.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=model or "phi-3.5-mini-instruct",
                max_tokens=max_tokens,
                temperature=temperature
            )
            if "choices" in response:
                return {
                    "text": response["choices"][0]["message"]["content"],
                    "provider": provider.value,
                    "model": model,
                    "cost_estimate": self._estimate_cost(prompt, max_tokens, provider, model)
                }
            else:
                return {
                    "text": f"Error: {response.get('error', {}).get('message', 'Unknown error')}",
                    "provider": provider.value,
                    "model": model,
                    "cost_estimate": 0.0
                }

        elif provider == ModelProvider.TOGETHER:
            return self._together_generate(prompt, model, max_tokens, temperature)

        elif provider == ModelProvider.SILICONFLOW:
            return self._siliconflow_generate(prompt, model, max_tokens, temperature)

        else:
            return {
                "text": "Provider not implemented yet",
                "provider": "none",
                "model": "none",
                "cost_estimate": 0.0
            }

    def _groq_generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Dict:
        """Generate text using GROQ"""
        url = f"{self.groq_config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.groq_config['api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model or "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "provider": "groq",
                    "model": model,
                    "cost_estimate": self._estimate_cost(prompt, max_tokens, ModelProvider.GROQ, model)
                }
        except Exception as e:
            return {"text": f"Error: {str(e)}", "provider": "groq", "model": model, "cost_estimate": 0.0}

    def _together_generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Dict:
        """Generate text using Together AI"""
        url = f"{self.together_config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.together_config['api_key']}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model or "meta-llama/Llama-3-8b-chat-hf",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "provider": "together",
                    "model": model,
                    "cost_estimate": self._estimate_cost(prompt, max_tokens, ModelProvider.TOGETHER, model)
                }
        except Exception as e:
            return {"text": f"Error: {str(e)}", "provider": "together", "model": model, "cost_estimate": 0.0}

    def _siliconflow_generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> Dict:
        """Generate text using SiliconFlow (Chinese provider with free tier)"""
        url = f"{self.siliconflow_config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.siliconflow_config['api_key']}",
            "Content-Type": "application/json"
        }

        # Use free model by default
        if not model or model not in self.siliconflow_config['free_models']:
            model = "Qwen/Qwen2.5-7B-Instruct"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                return {
                    "text": data["choices"][0]["message"]["content"],
                    "provider": "siliconflow",
                    "model": model,
                    "cost_estimate": 0.0  # Free tier
                }
        except Exception as e:
            return {"text": f"Error: {str(e)}", "provider": "siliconflow", "model": model, "cost_estimate": 0.0}

    def _get_cheapest_provider(self, task: TaskType) -> tuple:
        """Get the cheapest available provider for a task

        Args:
            task: Type of task

        Returns:
            Tuple of (provider, model, cost_per_million_tokens)
        """
        providers = self.cost_routing.get(task, [])
        # Return the first (cheapest) option
        if providers:
            return providers[0]
        return (ModelProvider.HUGGINGFACE, "microsoft/Phi-3-mini-4k-instruct", 0.0)

    def _estimate_cost(self, prompt: str, max_tokens: int, provider: ModelProvider, model: str) -> float:
        """Estimate cost for a request

        Args:
            prompt: Input prompt
            max_tokens: Maximum output tokens
            provider: Model provider
            model: Model name

        Returns:
            Estimated cost in USD
        """
        # Rough token estimation (1 token ≈ 4 characters)
        input_tokens = len(prompt) // 4
        output_tokens = max_tokens

        # Get cost per million tokens based on provider
        cost_per_million = 0.0

        if provider == ModelProvider.GROQ:
            cost_per_million = self.groq_config["models"].get(model, {}).get("cost", 0.05)
        elif provider == ModelProvider.DEEPSEEK:
            # DeepSeek charges separately for input/output
            cost_per_million = 0.14  # Average
        elif provider == ModelProvider.TOGETHER:
            cost_per_million = self.together_config["models"].get(model, {}).get("cost", 0.20)
        elif provider == ModelProvider.SILICONFLOW:
            cost_per_million = 0.0  # Free tier
        elif provider == ModelProvider.GITHUB:
            # GitHub Models pricing (many have free tiers)
            model_info = self.github_models.get_model_info(model)
            cost_per_million = model_info.get("cost_estimate", 0.15)

        total_tokens = input_tokens + output_tokens
        return (total_tokens / 1_000_000) * cost_per_million

    def compare_providers(self, prompt: str, max_tokens: int = 500) -> List[Dict]:
        """Compare response quality and cost across providers

        Args:
            prompt: Test prompt
            max_tokens: Maximum tokens

        Returns:
            Comparison results
        """
        results = []

        # Test each provider
        providers_to_test = [
            (ModelProvider.DEEPSEEK, "deepseek-chat"),
            (ModelProvider.GROQ, "llama-3.1-8b-instant"),
            (ModelProvider.HUGGINGFACE, "microsoft/Phi-3-mini-4k-instruct"),
            (ModelProvider.SILICONFLOW, "Qwen/Qwen2.5-7B-Instruct"),
        ]

        for provider, model in providers_to_test:
            try:
                start_time = time.time()
                response = self.generate_text(
                    prompt=prompt,
                    provider=provider,
                    model=model,
                    max_tokens=max_tokens,
                    use_cheapest=False
                )
                elapsed = time.time() - start_time

                results.append({
                    "provider": provider.value,
                    "model": model,
                    "response_time": round(elapsed, 2),
                    "cost_estimate": response.get("cost_estimate", 0.0),
                    "response_length": len(response.get("text", "")),
                    "success": bool(response.get("text"))
                })
            except Exception as e:
                results.append({
                    "provider": provider.value,
                    "model": model,
                    "error": str(e),
                    "success": False
                })

        return results

    def get_provider_status(self) -> Dict[str, bool]:
        """Check which providers are available

        Returns:
            Dictionary of provider availability
        """
        status = {}

        # Check DeepSeek
        status["deepseek"] = self.deepseek.test_connection()

        # Check HuggingFace
        status["huggingface"] = self.huggingface.check_api_status()

        # Check GROQ
        try:
            url = f"{self.groq_config['base_url']}/models"
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.groq_config['api_key']}"},
                timeout=5
            )
            status["groq"] = response.status_code == 200
        except (requests.RequestException, KeyError, ConnectionError):
            status["groq"] = False

        # Check Together
        try:
            url = f"{self.together_config['base_url']}/models"
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.together_config['api_key']}"},
                timeout=5
            )
            status["together"] = response.status_code == 200
        except (requests.RequestException, KeyError, ConnectionError):
            status["together"] = False

        # Check SiliconFlow
        try:
            url = f"{self.siliconflow_config['base_url']}/models"
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.siliconflow_config['api_key']}"},
                timeout=5
            )
            status["siliconflow"] = response.status_code == 200
        except (requests.RequestException, KeyError, ConnectionError):
            status["siliconflow"] = False

        # Check GitHub Models
        status["github"] = self.github_models.test_connection()

        # Check OCI
        if self.oci_manager:
            try:
                status["oci"] = len(self.oci_manager.list_instances()) >= 0
            except (AttributeError, TypeError, ConnectionError):
                status["oci"] = False
        else:
            status["oci"] = False

        return status


if __name__ == "__main__":
    print("Unified API Manager")
    print("=" * 80)

    # Initialize the unified manager
    manager = UnifiedAPIManager()

    # Check provider status
    print("\nProvider Status:")
    print("-" * 40)
    status = manager.get_provider_status()
    for provider, is_available in status.items():
        icon = "✓" if is_available else "✗"
        print(f"{icon} {provider:15} {'Available' if is_available else 'Not Available'}")

    # Show cost routing
    print("\n" + "=" * 80)
    print("COST-OPTIMIZED ROUTING")
    print("=" * 80)
    for task, providers in manager.cost_routing.items():
        print(f"\n{task.value.upper().replace('_', ' ')}:")
        for provider, model, cost in providers[:3]:  # Show top 3 cheapest
            print(f"  • {provider.value:12} | {model:40} | ${cost:.3f}/1M tokens")

    print("\n" + "=" * 80)
    print("Ready to use! Examples:")
    print("-" * 40)
    print("1. Generate text (cheapest): manager.generate_text('Hello world')")
    print("2. Generate code: manager.generate_text('Write Python function', provider=ModelProvider.DEEPSEEK)")
    print("3. Compare providers: manager.compare_providers('Explain quantum computing')")
    print("4. Get cost estimate: manager._estimate_cost('prompt', 100, ModelProvider.GROQ, 'llama-3.1-8b')")