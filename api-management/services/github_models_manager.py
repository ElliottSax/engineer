#!/usr/bin/env python3
"""
GitHub Models Manager
Integration with GitHub's AI model marketplace
Provides access to various models including GPT-4, Llama, Mistral, and more
"""

import requests
import json
from typing import Dict, List, Optional, Any, Generator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_loader import ConfigLoader


class GitHubModelsManager:
    """Manager for GitHub Models API"""

    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub Models manager

        Args:
            token: GitHub personal access token. If None, loads from config
        """
        if not token:
            config = ConfigLoader()
            token = config.github_token

        self.token = token
        self.base_url = "https://models.inference.ai.azure.com"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Available models on GitHub Models with their characteristics
        self.models = {
            # OpenAI Models
            "gpt-4o": {
                "provider": "OpenAI",
                "description": "Most capable GPT-4 model",
                "context_length": 128000,
                "free_tier": False,
                "capabilities": ["chat", "code", "analysis", "vision"],
                "cost_estimate": 5.00  # Per 1M tokens (estimate)
            },
            "gpt-4o-mini": {
                "provider": "OpenAI",
                "description": "Small, fast, and cost-effective",
                "context_length": 128000,
                "free_tier": True,  # Limited free tier
                "capabilities": ["chat", "code"],
                "cost_estimate": 0.15
            },

            # Meta Llama Models
            "meta-llama-3.1-405b-instruct": {
                "provider": "Meta",
                "description": "Largest Llama 3.1 model",
                "context_length": 128000,
                "free_tier": False,
                "capabilities": ["chat", "code", "analysis"],
                "cost_estimate": 2.00
            },
            "meta-llama-3.1-70b-instruct": {
                "provider": "Meta",
                "description": "Large Llama model, good balance",
                "context_length": 128000,
                "free_tier": True,  # Limited free tier
                "capabilities": ["chat", "code"],
                "cost_estimate": 0.90
            },
            "meta-llama-3.1-8b-instruct": {
                "provider": "Meta",
                "description": "Smaller, faster Llama model",
                "context_length": 128000,
                "free_tier": True,
                "capabilities": ["chat", "code"],
                "cost_estimate": 0.10
            },

            # Mistral Models
            "mistral-large": {
                "provider": "Mistral",
                "description": "Mistral's flagship model",
                "context_length": 32000,
                "free_tier": False,
                "capabilities": ["chat", "code", "analysis"],
                "cost_estimate": 1.00
            },
            "mistral-large-2407": {
                "provider": "Mistral",
                "description": "Latest Mistral large model",
                "context_length": 128000,
                "free_tier": False,
                "capabilities": ["chat", "code"],
                "cost_estimate": 1.20
            },
            "mistral-small": {
                "provider": "Mistral",
                "description": "Efficient Mistral model",
                "context_length": 32000,
                "free_tier": True,
                "capabilities": ["chat"],
                "cost_estimate": 0.20
            },
            "mistral-nemo": {
                "provider": "Mistral",
                "description": "12B parameter model",
                "context_length": 128000,
                "free_tier": True,
                "capabilities": ["chat", "code"],
                "cost_estimate": 0.15
            },

            # Cohere Models
            "cohere-command-r-plus": {
                "provider": "Cohere",
                "description": "Cohere's powerful RAG model",
                "context_length": 128000,
                "free_tier": False,
                "capabilities": ["chat", "rag", "analysis"],
                "cost_estimate": 1.50
            },
            "cohere-command-r": {
                "provider": "Cohere",
                "description": "Efficient RAG-optimized model",
                "context_length": 128000,
                "free_tier": True,
                "capabilities": ["chat", "rag"],
                "cost_estimate": 0.50
            },

            # AI21 Models
            "ai21-jamba-1.5-mini": {
                "provider": "AI21",
                "description": "Hybrid SSM-Transformer model",
                "context_length": 256000,
                "free_tier": True,
                "capabilities": ["chat"],
                "cost_estimate": 0.20
            },
            "ai21-jamba-1.5-large": {
                "provider": "AI21",
                "description": "Large hybrid model",
                "context_length": 256000,
                "free_tier": False,
                "capabilities": ["chat", "analysis"],
                "cost_estimate": 2.00
            },

            # Phi Models (Microsoft)
            # Note: Phi models may not be available through GitHub Models API
            # Using GPT-4o-mini as the free/cheap alternative
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stream: bool = False,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """Create a chat completion using GitHub Models

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model ID from GitHub Models
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty (-2 to 2)
            presence_penalty: Presence penalty (-2 to 2)
            stream: Whether to stream the response
            stop: Stop sequences

        Returns:
            API response or generator for streaming
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens
        if stop:
            payload["stop"] = stop

        if stream:
            return self._stream_chat_completion(url, payload)
        else:
            try:
                response = requests.post(url, headers=self.headers, json=payload)

                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "error": {
                            "message": f"API Error: {response.status_code} - {response.text}",
                            "type": "api_error",
                            "code": response.status_code
                        }
                    }
            except Exception as e:
                return {
                    "error": {
                        "message": str(e),
                        "type": "connection_error"
                    }
                }

    def _stream_chat_completion(self, url: str, payload: Dict) -> Generator:
        """Stream chat completion responses

        Args:
            url: API endpoint
            payload: Request payload

        Yields:
            Streamed response chunks
        """
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data != '[DONE]':
                                try:
                                    yield json.loads(data)
                                except json.JSONDecodeError:
                                    continue
            else:
                yield {
                    "error": {
                        "message": f"Stream Error: {response.status_code}",
                        "type": "stream_error",
                        "code": response.status_code
                    }
                }
        except Exception as e:
            yield {
                "error": {
                    "message": str(e),
                    "type": "connection_error"
                }
            }

    def embeddings(
        self,
        input_texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> Dict:
        """Generate embeddings for texts

        Args:
            input_texts: List of texts to embed
            model: Embedding model to use

        Returns:
            Embeddings response
        """
        url = f"{self.base_url}/embeddings"

        payload = {
            "model": model,
            "input": input_texts
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": {
                        "message": f"API Error: {response.status_code} - {response.text}",
                        "type": "api_error",
                        "code": response.status_code
                    }
                }
        except Exception as e:
            return {
                "error": {
                    "message": str(e),
                    "type": "connection_error"
                }
            }

    def list_models(self) -> List[Dict]:
        """List all available models on GitHub Models

        Returns:
            List of available models with details
        """
        url = f"{self.base_url}/models"

        try:
            response = requests.get(url, headers=self.headers)

            if response.status_code == 200:
                api_models = response.json().get("data", [])

                # Combine with our local model information
                enhanced_models = []
                for api_model in api_models:
                    model_id = api_model.get("id")
                    local_info = self.models.get(model_id, {})

                    enhanced_models.append({
                        "id": model_id,
                        "provider": local_info.get("provider", "Unknown"),
                        "description": local_info.get("description", ""),
                        "free_tier": local_info.get("free_tier", False),
                        "context_length": local_info.get("context_length", 0),
                        "capabilities": local_info.get("capabilities", []),
                        "cost_estimate": local_info.get("cost_estimate", 0)
                    })

                return enhanced_models
            else:
                return list(self.models.values())

        except (requests.RequestException, ConnectionError, TimeoutError, KeyError, ValueError):
            # Return local model list if API fails
            return [
                {
                    "id": model_id,
                    **info
                }
                for model_id, info in self.models.items()
            ]

    def get_free_tier_models(self) -> List[str]:
        """Get list of models available in free tier

        Returns:
            List of free tier model IDs
        """
        return [
            model_id
            for model_id, info in self.models.items()
            if info.get("free_tier", False)
        ]

    def get_model_info(self, model_id: str) -> Dict:
        """Get detailed information about a specific model

        Args:
            model_id: Model identifier

        Returns:
            Model information dictionary
        """
        return self.models.get(model_id, {
            "description": "Unknown model",
            "free_tier": False,
            "capabilities": [],
            "cost_estimate": 1.0
        })

    def code_generation(
        self,
        prompt: str,
        language: Optional[str] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """Generate code using GitHub Models

        Args:
            prompt: Code generation prompt
            language: Programming language
            model: Model to use
            temperature: Lower for more deterministic code
            max_tokens: Maximum tokens to generate

        Returns:
            Generated code
        """
        system_prompt = "You are an expert programmer. Generate clean, efficient, and well-commented code."

        if language:
            system_prompt += f" Use {language} programming language."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.get('error', {}).get('message', 'Unknown error')}"

    def analyze_image(
        self,
        image_url: str,
        prompt: str = "What's in this image?",
        model: str = "gpt-4o",
        max_tokens: int = 500
    ) -> str:
        """Analyze an image using vision-capable models

        Args:
            image_url: URL of the image to analyze
            prompt: Question about the image
            model: Vision-capable model (e.g., gpt-4o)
            max_tokens: Maximum tokens for response

        Returns:
            Image analysis result
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]

        response = self.chat_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens
        )

        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.get('error', {}).get('message', 'Unknown error')}"

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "gpt-4o-mini"
    ) -> Dict[str, float]:
        """Estimate API cost for a request

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model to use

        Returns:
            Cost breakdown
        """
        model_info = self.models.get(model, {"cost_estimate": 1.0})
        cost_per_million = model_info.get("cost_estimate", 1.0)

        # Most models charge same for input/output, but we can adjust if needed
        total_tokens = input_tokens + output_tokens
        total_cost = (total_tokens / 1_000_000) * cost_per_million

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_per_million": cost_per_million,
            "total_cost": round(total_cost, 6),
            "is_free_tier": model_info.get("free_tier", False)
        }

    def test_connection(self) -> bool:
        """Test GitHub Models API connection

        Returns:
            True if connection successful
        """
        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                model="gpt-4o-mini",  # More reliable for testing
                max_tokens=5
            )
            return "choices" in response
        except (requests.RequestException, ConnectionError, TimeoutError, KeyError):
            return False


# Recommended usage patterns
GITHUB_MODELS_RECOMMENDATIONS = {
    "free_tier": {
        "models": [
            "phi-3.5-mini-instruct",  # Completely free
            "gpt-4o-mini",  # Limited free
            "meta-llama-3.1-8b-instruct",  # Limited free
            "mistral-small",  # Limited free
            "mistral-nemo",  # Limited free
        ],
        "use_cases": [
            "Development and testing",
            "Light production workloads",
            "Educational projects",
            "Prototyping"
        ]
    },
    "best_value": {
        "models": [
            ("gpt-4o-mini", "Best overall value - fast and capable"),
            ("meta-llama-3.1-8b-instruct", "Great for chat and code"),
            ("mistral-nemo", "Good balance of cost and capability"),
            ("phi-3.5-mini-instruct", "Free for basic tasks")
        ]
    },
    "specialized": {
        "code": ["gpt-4o-mini", "meta-llama-3.1-70b-instruct"],
        "long_context": ["ai21-jamba-1.5-mini", "meta-llama-3.1-8b-instruct"],
        "vision": ["gpt-4o", "gpt-4o-mini"],
        "rag": ["cohere-command-r", "cohere-command-r-plus"]
    }
}


if __name__ == "__main__":
    # Test and demonstrate GitHub Models capabilities
    print("GitHub Models Manager")
    print("=" * 80)

    # Initialize manager
    gh = GitHubModelsManager()

    # Test connection
    print("\nTesting GitHub Models API...")
    if gh.test_connection():
        print("✓ GitHub Models API is accessible")

        # Show available models
        print("\n" + "=" * 80)
        print("AVAILABLE MODELS")
        print("=" * 80)

        # Group by provider
        models_by_provider = {}
        for model_id, info in gh.models.items():
            provider = info.get("provider", "Unknown")
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append((model_id, info))

        for provider, models in models_by_provider.items():
            print(f"\n{provider}:")
            for model_id, info in models:
                free = "FREE" if info.get("free_tier") else f"${info.get('cost_estimate', 0):.2f}/1M"
                print(f"  • {model_id:30} {free:10} - {info.get('description', '')}")

        # Show recommendations
        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)

        print("\n🎯 FREE TIER MODELS:")
        for model in GITHUB_MODELS_RECOMMENDATIONS["free_tier"]["models"]:
            print(f"  • {model}")

        print("\n💰 BEST VALUE MODELS:")
        for model, description in GITHUB_MODELS_RECOMMENDATIONS["best_value"]["models"]:
            print(f"  • {model}: {description}")

        print("\n🔧 SPECIALIZED USE CASES:")
        for use_case, models in GITHUB_MODELS_RECOMMENDATIONS["specialized"].items():
            print(f"  {use_case.upper()}: {', '.join(models)}")

    else:
        print("✗ GitHub Models API is not accessible")
        print("  Make sure your GitHub token has the necessary permissions")
        print("  You may need to join the GitHub Models waitlist")