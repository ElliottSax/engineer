#!/usr/bin/env python3
"""
Hugging Face Ecosystem Manager
Complete integration with Hugging Face services including Inference API,
Model Hub, Datasets, and Spaces
"""

import requests
import json
from typing import Dict, List, Optional, Any, Union
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_loader import ConfigLoader


class HuggingFaceManager:
    """Comprehensive manager for Hugging Face ecosystem"""

    def __init__(self, token: Optional[str] = None):
        """Initialize Hugging Face manager

        Args:
            token: Hugging Face API token. If None, loads from config
        """
        if not token:
            config = ConfigLoader()
            token = config.huggingface_token

        self.token = token
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.inference_api_base = "https://api-inference.huggingface.co/models/"
        self.hub_api_base = "https://huggingface.co/api/"

    # ============== Inference API Methods ==============

    def text_generation(
        self,
        prompt: str,
        model: str = "microsoft/Phi-3-mini-4k-instruct",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True
    ) -> str:
        """Generate text using Hugging Face Inference API

        Args:
            prompt: Input text prompt
            model: Model ID from Hugging Face Hub
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        url = f"{self.inference_api_base}{model}"

        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample
            }
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"Error: {response.status_code} - {response.text}"

    def text_classification(
        self,
        text: str,
        model: str = "distilbert-base-uncased-finetuned-sst-2-english"
    ) -> List[Dict]:
        """Classify text sentiment/category

        Args:
            text: Text to classify
            model: Classification model ID

        Returns:
            List of labels with scores
        """
        url = f"{self.inference_api_base}{model}"
        payload = {"inputs": text}

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()[0]
        else:
            return [{"error": f"{response.status_code} - {response.text}"}]

    def question_answering(
        self,
        question: str,
        context: str,
        model: str = "deepset/roberta-base-squad2"
    ) -> Dict:
        """Answer questions based on context

        Args:
            question: Question to answer
            context: Context containing the answer
            model: QA model ID

        Returns:
            Answer with confidence score
        """
        url = f"{self.inference_api_base}{model}"
        payload = {
            "inputs": {
                "question": question,
                "context": context
            }
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"{response.status_code} - {response.text}"}

    def summarization(
        self,
        text: str,
        model: str = "facebook/bart-large-cnn",
        max_length: int = 130,
        min_length: int = 30
    ) -> str:
        """Summarize long text

        Args:
            text: Text to summarize
            model: Summarization model ID
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            Summarized text
        """
        url = f"{self.inference_api_base}{model}"
        payload = {
            "inputs": text,
            "parameters": {
                "max_length": max_length,
                "min_length": min_length
            }
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()[0]["summary_text"]
        else:
            return f"Error: {response.status_code} - {response.text}"

    def translation(
        self,
        text: str,
        model: str = "Helsinki-NLP/opus-mt-en-fr",
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> str:
        """Translate text between languages

        Args:
            text: Text to translate
            model: Translation model ID
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Translated text
        """
        url = f"{self.inference_api_base}{model}"
        payload = {"inputs": text}

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()[0]["translation_text"]
        else:
            return f"Error: {response.status_code} - {response.text}"

    def embeddings(
        self,
        texts: Union[str, List[str]],
        model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> List[List[float]]:
        """Generate text embeddings

        Args:
            texts: Text or list of texts
            model: Embedding model ID

        Returns:
            List of embedding vectors
        """
        url = f"{self.inference_api_base}{model}"

        if isinstance(texts, str):
            texts = [texts]

        payload = {"inputs": texts}

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            return [[]]

    def zero_shot_classification(
        self,
        text: str,
        candidate_labels: List[str],
        model: str = "facebook/bart-large-mnli"
    ) -> Dict:
        """Classify text without training

        Args:
            text: Text to classify
            candidate_labels: Possible labels
            model: Zero-shot model ID

        Returns:
            Classification results
        """
        url = f"{self.inference_api_base}{model}"
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": candidate_labels}
        }

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"{response.status_code} - {response.text}"}

    def image_generation(
        self,
        prompt: str,
        model: str = "stabilityai/stable-diffusion-2",
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.5
    ) -> bytes:
        """Generate images from text

        Args:
            prompt: Text description of image
            model: Image generation model ID
            negative_prompt: What to avoid in image
            guidance_scale: Guidance scale for generation

        Returns:
            Image bytes
        """
        url = f"{self.inference_api_base}{model}"
        payload = {
            "inputs": prompt,
            "parameters": {
                "guidance_scale": guidance_scale
            }
        }

        if negative_prompt:
            payload["parameters"]["negative_prompt"] = negative_prompt

        response = requests.post(url, headers=self.headers, json=payload)

        if response.status_code == 200:
            return response.content
        else:
            return b""

    # ============== Model Hub Methods ==============

    def list_models(
        self,
        search: Optional[str] = None,
        task: Optional[str] = None,
        library: Optional[str] = None,
        language: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 10
    ) -> List[Dict]:
        """List models from Hugging Face Hub

        Args:
            search: Search query
            task: Task type (e.g., 'text-generation')
            library: Framework (e.g., 'transformers', 'pytorch')
            language: Model language
            sort: Sort by (downloads, likes, created)
            limit: Number of results

        Returns:
            List of model information
        """
        url = "https://huggingface.co/api/models"
        params = {
            "sort": sort,
            "limit": limit
        }

        if search:
            params["search"] = search
        if task:
            params["filter"] = task
        if library:
            params["library"] = library
        if language:
            params["language"] = language

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return []

    def get_model_info(self, model_id: str) -> Dict:
        """Get detailed model information

        Args:
            model_id: Model identifier

        Returns:
            Model metadata
        """
        url = f"https://huggingface.co/api/models/{model_id}"

        response = requests.get(url)

        if response.status_code == 200:
            return response.json()
        else:
            return {}

    def list_datasets(
        self,
        search: Optional[str] = None,
        task: Optional[str] = None,
        language: Optional[str] = None,
        sort: str = "downloads",
        limit: int = 10
    ) -> List[Dict]:
        """List datasets from Hugging Face Hub

        Args:
            search: Search query
            task: Task type
            language: Dataset language
            sort: Sort by (downloads, likes, created)
            limit: Number of results

        Returns:
            List of dataset information
        """
        url = "https://huggingface.co/api/datasets"
        params = {
            "sort": sort,
            "limit": limit
        }

        if search:
            params["search"] = search
        if task:
            params["task_categories"] = task
        if language:
            params["languages"] = language

        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json()
        else:
            return []

    # ============== Utility Methods ==============

    def check_api_status(self) -> bool:
        """Check if Hugging Face API is accessible

        Returns:
            True if API is working
        """
        try:
            url = f"{self.inference_api_base}bert-base-uncased"
            response = requests.get(url, headers=self.headers, timeout=5)
            return response.status_code in [200, 503]  # 503 means model is loading
        except (requests.RequestException, ConnectionError, TimeoutError):
            return False

    def get_token_info(self) -> Dict:
        """Get information about the current token

        Returns:
            Token information
        """
        url = "https://huggingface.co/api/whoami"

        response = requests.get(url, headers=self.headers)

        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "Invalid token or API error"}


# Cost-effective model recommendations
COST_EFFECTIVE_MODELS = {
    "text_generation": [
        "microsoft/Phi-3-mini-4k-instruct",  # Small, efficient
        "google/flan-t5-small",  # Good for many tasks
        "EleutherAI/gpt-neo-125M",  # Small GPT variant
    ],
    "embeddings": [
        "sentence-transformers/all-MiniLM-L6-v2",  # Fast and good
        "sentence-transformers/paraphrase-MiniLM-L3-v2",  # Very small
    ],
    "classification": [
        "distilbert-base-uncased-finetuned-sst-2-english",  # Fast
        "nlptown/bert-base-multilingual-uncased-sentiment",  # Multilingual
    ],
    "translation": [
        "Helsinki-NLP/opus-mt-en-es",  # English to Spanish
        "Helsinki-NLP/opus-mt-en-fr",  # English to French
        "Helsinki-NLP/opus-mt-en-de",  # English to German
    ],
    "summarization": [
        "sshleifer/distilbart-cnn-12-6",  # Faster BART
        "google/flan-t5-small",  # Multi-purpose
    ],
    "qa": [
        "distilbert-base-cased-distilled-squad",  # Fast QA
        "deepset/minilm-uncased-squad2",  # Small and effective
    ]
}


if __name__ == "__main__":
    # Example usage and testing
    print("Hugging Face Ecosystem Manager")
    print("=" * 60)

    # Initialize manager
    hf = HuggingFaceManager()

    # Check API status
    print("\nChecking API Status...")
    if hf.check_api_status():
        print("✓ Hugging Face API is accessible")

        # Get token info
        print("\nToken Information:")
        token_info = hf.get_token_info()
        if "name" in token_info:
            print(f"  User: {token_info.get('name', 'N/A')}")
            print(f"  Type: {token_info.get('type', 'N/A')}")
        else:
            print(f"  {token_info.get('error', 'Unknown error')}")

        # Show cost-effective models
        print("\n" + "=" * 60)
        print("COST-EFFECTIVE MODEL RECOMMENDATIONS")
        print("=" * 60)
        for task, models in COST_EFFECTIVE_MODELS.items():
            print(f"\n{task.upper().replace('_', ' ')}:")
            for model in models:
                print(f"  • {model}")

    else:
        print("✗ Hugging Face API is not accessible")