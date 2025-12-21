#!/usr/bin/env python3
"""
DeepSeek API Manager
Integration with DeepSeek's cost-effective AI models
DeepSeek offers very competitive pricing for high-quality models
"""

import requests
import json
from typing import Dict, List, Optional, Generator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_loader import ConfigLoader


class DeepSeekManager:
    """Manager for DeepSeek API services"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize DeepSeek manager

        Args:
            api_key: DeepSeek API key. If None, loads from config
        """
        if not api_key:
            config = ConfigLoader()
            api_key = config.deepseek_api_key

        self.api_key = api_key
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # DeepSeek models with their characteristics
        self.models = {
            "deepseek-chat": {
                "description": "General purpose chat model",
                "context_length": 32768,
                "cost_per_million_input": 0.14,  # $0.14 per 1M tokens
                "cost_per_million_output": 0.28,  # $0.28 per 1M tokens
            },
            "deepseek-coder": {
                "description": "Specialized for coding tasks",
                "context_length": 16384,
                "cost_per_million_input": 0.14,
                "cost_per_million_output": 0.28,
            }
        }

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.95,
        frequency_penalty: float = 0,
        presence_penalty: float = 0,
        stream: bool = False,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """Create a chat completion

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use ('deepseek-chat' or 'deepseek-coder')
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

    def _stream_chat_completion(self, url: str, payload: Dict) -> Generator:
        """Stream chat completion responses

        Args:
            url: API endpoint
            payload: Request payload

        Yields:
            Streamed response chunks
        """
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

    def code_completion(
        self,
        prompt: str,
        language: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000
    ) -> str:
        """Generate code using DeepSeek Coder model

        Args:
            prompt: Code generation prompt
            language: Programming language (optional)
            temperature: Lower temperature for more deterministic code
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
            model="deepseek-coder",
            temperature=temperature,
            max_tokens=max_tokens
        )

        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.get('error', {}).get('message', 'Unknown error')}"

    def code_review(
        self,
        code: str,
        language: Optional[str] = None,
        focus_areas: Optional[List[str]] = None
    ) -> str:
        """Review code for issues and improvements

        Args:
            code: Code to review
            language: Programming language
            focus_areas: Specific areas to focus on (e.g., ['security', 'performance'])

        Returns:
            Code review feedback
        """
        prompt = f"Please review the following code:\n\n```{language or ''}\n{code}\n```\n\n"

        if focus_areas:
            prompt += f"Focus particularly on: {', '.join(focus_areas)}\n"

        prompt += "Provide feedback on:\n1. Potential bugs or issues\n2. Performance optimizations\n3. Code quality and best practices\n4. Security concerns\n5. Suggestions for improvement"

        messages = [
            {"role": "system", "content": "You are an experienced code reviewer providing constructive feedback."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(
            messages=messages,
            model="deepseek-coder",
            temperature=0.3,
            max_tokens=1500
        )

        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.get('error', {}).get('message', 'Unknown error')}"

    def explain_code(
        self,
        code: str,
        language: Optional[str] = None,
        detail_level: str = "medium"
    ) -> str:
        """Explain how code works

        Args:
            code: Code to explain
            language: Programming language
            detail_level: 'basic', 'medium', or 'detailed'

        Returns:
            Code explanation
        """
        detail_instructions = {
            "basic": "Provide a simple, high-level explanation suitable for beginners.",
            "medium": "Provide a clear explanation with moderate technical detail.",
            "detailed": "Provide a comprehensive explanation including all technical details."
        }

        prompt = f"Explain the following code:\n\n```{language or ''}\n{code}\n```\n\n"
        prompt += detail_instructions.get(detail_level, detail_instructions["medium"])

        messages = [
            {"role": "system", "content": "You are a patient programming teacher."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(
            messages=messages,
            model="deepseek-coder",
            temperature=0.5,
            max_tokens=1000
        )

        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.get('error', {}).get('message', 'Unknown error')}"

    def translate_code(
        self,
        code: str,
        from_language: str,
        to_language: str
    ) -> str:
        """Translate code from one language to another

        Args:
            code: Source code
            from_language: Source programming language
            to_language: Target programming language

        Returns:
            Translated code
        """
        prompt = f"""Translate the following {from_language} code to {to_language}:

```{from_language}
{code}
```

Provide the translated code with appropriate comments and maintain the same functionality."""

        messages = [
            {"role": "system", "content": f"You are an expert in both {from_language} and {to_language}."},
            {"role": "user", "content": prompt}
        ]

        response = self.chat_completion(
            messages=messages,
            model="deepseek-coder",
            temperature=0.3,
            max_tokens=2000
        )

        if "choices" in response:
            return response["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.get('error', {}).get('message', 'Unknown error')}"

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str = "deepseek-chat"
    ) -> Dict[str, float]:
        """Estimate API cost for a request

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model: Model to use

        Returns:
            Cost breakdown
        """
        model_info = self.models.get(model, self.models["deepseek-chat"])

        input_cost = (input_tokens / 1_000_000) * model_info["cost_per_million_input"]
        output_cost = (output_tokens / 1_000_000) * model_info["cost_per_million_output"]
        total_cost = input_cost + output_cost

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "cost_per_1k_tokens": round((total_cost * 1000) / (input_tokens + output_tokens), 6)
        }

    def get_model_info(self) -> Dict:
        """Get information about available models

        Returns:
            Model information dictionary
        """
        return self.models

    def test_connection(self) -> bool:
        """Test API connection

        Returns:
            True if connection successful
        """
        try:
            response = self.chat_completion(
                messages=[{"role": "user", "content": "Hello"}],
                model="deepseek-chat",
                max_tokens=10
            )
            return "choices" in response
        except (requests.RequestException, ConnectionError, TimeoutError, KeyError):
            return False


# Example usage patterns for cost-effective operations
USAGE_PATTERNS = {
    "batch_processing": """
    For cost-effective batch processing:
    1. Combine multiple small requests into one larger request
    2. Use lower temperature for deterministic outputs
    3. Set appropriate max_tokens to avoid unnecessary generation
    """,

    "code_tasks": """
    For coding tasks:
    1. Use deepseek-coder model for better code understanding
    2. Provide clear, specific prompts
    3. Use temperature 0.1-0.3 for consistent code generation
    """,

    "streaming": """
    For real-time applications:
    1. Use streaming to get faster initial responses
    2. Implement proper error handling for stream interruptions
    3. Consider caching frequent queries
    """
}


if __name__ == "__main__":
    # Example usage and testing
    print("DeepSeek API Manager")
    print("=" * 60)

    # Initialize manager
    ds = DeepSeekManager()

    # Test connection
    print("\nTesting API Connection...")
    if ds.test_connection():
        print("✓ DeepSeek API is accessible")

        # Show model information
        print("\nAvailable Models:")
        print("-" * 40)
        for model_name, info in ds.get_model_info().items():
            print(f"\n{model_name}:")
            print(f"  Description: {info['description']}")
            print(f"  Context Length: {info['context_length']:,} tokens")
            print(f"  Input Cost: ${info['cost_per_million_input']}/1M tokens")
            print(f"  Output Cost: ${info['cost_per_million_output']}/1M tokens")

        # Cost estimation example
        print("\n" + "=" * 60)
        print("COST ESTIMATION EXAMPLE")
        print("=" * 60)
        example_cost = ds.estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            model="deepseek-chat"
        )
        print(f"For a request with {example_cost['input_tokens']} input and {example_cost['output_tokens']} output tokens:")
        print(f"  Input Cost: ${example_cost['input_cost']:.6f}")
        print(f"  Output Cost: ${example_cost['output_cost']:.6f}")
        print(f"  Total Cost: ${example_cost['total_cost']:.6f}")
        print(f"  Cost per 1K tokens: ${example_cost['cost_per_1k_tokens']:.6f}")

        # Usage recommendations
        print("\n" + "=" * 60)
        print("USAGE RECOMMENDATIONS")
        print("=" * 60)
        for pattern, description in USAGE_PATTERNS.items():
            print(f"\n{pattern.upper().replace('_', ' ')}:")
            print(description)

    else:
        print("✗ DeepSeek API is not accessible. Check your API key.")