"""
Ultra-Cheap API Router for Video Generation
Routes to the cheapest available API while maintaining quality
Target: < $0.50 per complete video
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import aiohttp
from collections import defaultdict

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of tasks for routing"""
    SCRIPT_GENERATION = "script"
    IMAGE_GENERATION = "image"
    VIDEO_GENERATION = "video"
    AUDIO_GENERATION = "audio"
    EMBEDDING = "embedding"


@dataclass
class APIProvider:
    """API provider configuration"""
    name: str
    endpoint: str
    api_key_env: str
    cost_per_unit: float  # Cost per 1K tokens or per image
    quality_score: float  # 0-1 quality rating
    speed_score: float    # 0-1 speed rating
    free_tier_limit: Optional[int] = None
    current_usage: int = 0
    is_available: bool = True


class CheapAPIRouter:
    """
    Intelligent router that selects the cheapest API
    while maintaining quality standards
    """

    def __init__(self, quality_threshold: float = 0.7):
        self.quality_threshold = quality_threshold
        self.usage_tracker = defaultdict(int)
        self.cost_tracker = defaultdict(float)

        # Initialize API providers
        self.providers = self._initialize_providers()

        # Track failures for circuit breaking
        self.failure_counts = defaultdict(int)
        self.max_failures = 3

    def _initialize_providers(self) -> Dict[TaskType, List[APIProvider]]:
        """Initialize all available API providers"""

        providers = {
            TaskType.SCRIPT_GENERATION: [
                APIProvider(
                    name="deepseek_v3",
                    endpoint="https://api.deepseek.com/v1/chat/completions",
                    api_key_env="DEEPSEEK_API_KEY",
                    cost_per_unit=0.14,  # per 1M input tokens
                    quality_score=0.9,
                    speed_score=0.8
                ),
                APIProvider(
                    name="qwen_72b",
                    endpoint="https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation",
                    api_key_env="ALIBABA_API_KEY",
                    cost_per_unit=0.5,  # per 1M tokens
                    quality_score=0.85,
                    speed_score=0.7
                ),
                APIProvider(
                    name="mixtral_8x7b",
                    endpoint="https://api.together.xyz/v1/chat/completions",
                    api_key_env="TOGETHER_API_KEY",
                    cost_per_unit=0.27,  # per 1M tokens
                    quality_score=0.8,
                    speed_score=0.9
                ),
                APIProvider(
                    name="openrouter_hermes",
                    endpoint="https://openrouter.ai/api/v1/chat/completions",
                    api_key_env="OPENROUTER_API_KEY",
                    cost_per_unit=0.18,  # per 1M tokens
                    quality_score=0.75,
                    speed_score=0.85
                ),
                APIProvider(
                    name="huggingface_zephyr",
                    endpoint="https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
                    api_key_env="HUGGINGFACE_API_KEY",
                    cost_per_unit=0.0,  # Free tier
                    quality_score=0.7,
                    speed_score=0.6,
                    free_tier_limit=1000  # requests per month
                )
            ],

            TaskType.IMAGE_GENERATION: [
                APIProvider(
                    name="gemini_imagen3",
                    endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    api_key_env="GEMINI_API_KEY",
                    cost_per_unit=0.0,  # FREE tier!
                    quality_score=0.95,
                    speed_score=0.9,
                    free_tier_limit=1000  # images per day
                ),
                APIProvider(
                    name="together_sdxl",
                    endpoint="https://api.together.xyz/v1/images/generations",
                    api_key_env="TOGETHER_API_KEY",
                    cost_per_unit=0.6,  # per 1000 images
                    quality_score=0.85,
                    speed_score=0.8
                ),
                APIProvider(
                    name="huggingface_sdxl",
                    endpoint="https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
                    api_key_env="HUGGINGFACE_API_KEY",
                    cost_per_unit=2.0,  # per 1000 images
                    quality_score=0.85,
                    speed_score=0.7
                ),
                APIProvider(
                    name="replicate_sdxl",
                    endpoint="https://api.replicate.com/v1/predictions",
                    api_key_env="REPLICATE_API_TOKEN",
                    cost_per_unit=3.2,  # per 1000 images
                    quality_score=0.85,
                    speed_score=0.75
                ),
                APIProvider(
                    name="stability_api",
                    endpoint="https://api.stability.ai/v1/generation",
                    api_key_env="STABILITY_API_KEY",
                    cost_per_unit=8.0,  # per 1000 images
                    quality_score=0.9,
                    speed_score=0.85
                )
            ],

            TaskType.VIDEO_GENERATION: [
                APIProvider(
                    name="huggingface_svd",
                    endpoint="https://api-inference.huggingface.co/models/stabilityai/stable-video-diffusion-img2vid",
                    api_key_env="HUGGINGFACE_API_KEY",
                    cost_per_unit=5.0,  # per video
                    quality_score=0.7,
                    speed_score=0.5
                ),
                APIProvider(
                    name="replicate_svd",
                    endpoint="https://api.replicate.com/v1/predictions",
                    api_key_env="REPLICATE_API_TOKEN",
                    cost_per_unit=10.0,  # per video
                    quality_score=0.75,
                    speed_score=0.6
                ),
                APIProvider(
                    name="runwayml_gen2",
                    endpoint="https://api.runwayml.com/v1/generate",
                    api_key_env="RUNWAY_API_KEY",
                    cost_per_unit=50.0,  # per video (more expensive)
                    quality_score=0.9,
                    speed_score=0.7
                )
            ],

            TaskType.AUDIO_GENERATION: [
                APIProvider(
                    name="coqui_xtts",
                    endpoint="https://app.coqui.ai/api/v2/samples",
                    api_key_env="COQUI_API_KEY",
                    cost_per_unit=0.05,  # per 1000 chars
                    quality_score=0.8,
                    speed_score=0.7
                ),
                APIProvider(
                    name="azure_cognitive",
                    endpoint="https://cognitiveservices.azure.com/",
                    api_key_env="AZURE_SPEECH_KEY",
                    cost_per_unit=0.08,  # per 1000 chars
                    quality_score=0.85,
                    speed_score=0.9
                ),
                APIProvider(
                    name="google_tts",
                    endpoint="https://texttospeech.googleapis.com/v1/text:synthesize",
                    api_key_env="GOOGLE_CLOUD_API_KEY",
                    cost_per_unit=0.06,  # per 1000 chars
                    quality_score=0.75,
                    speed_score=0.95
                ),
                APIProvider(
                    name="elevenlabs_free",
                    endpoint="https://api.elevenlabs.io/v1/text-to-speech",
                    api_key_env="ELEVENLABS_API_KEY",
                    cost_per_unit=0.0,  # 10K chars free
                    quality_score=0.95,
                    speed_score=0.8,
                    free_tier_limit=10000  # chars per month
                )
            ]
        }

        return providers

    async def route_request(
        self,
        task_type: TaskType,
        request_data: Dict[str, Any],
        quality_override: Optional[float] = None,
        prefer_free: bool = True
    ) -> Dict[str, Any]:
        """
        Route request to the cheapest suitable API

        Args:
            task_type: Type of task to route
            request_data: Request payload
            quality_override: Override quality threshold
            prefer_free: Prefer free tier APIs when available
        """

        quality_min = quality_override or self.quality_threshold
        providers = self.providers.get(task_type, [])

        # Filter available providers
        available = [
            p for p in providers
            if p.is_available and
            p.quality_score >= quality_min and
            self.failure_counts[p.name] < self.max_failures
        ]

        if not available:
            raise Exception(f"No available providers for {task_type}")

        # Sort by cost (prefer free tier if requested)
        if prefer_free:
            # Check free tier availability
            free_providers = [
                p for p in available
                if p.cost_per_unit == 0 and
                (p.free_tier_limit is None or p.current_usage < p.free_tier_limit)
            ]
            if free_providers:
                available = free_providers

        # Sort by cost, then by quality
        available.sort(key=lambda p: (p.cost_per_unit, -p.quality_score))

        # Try providers in order
        for provider in available:
            try:
                result = await self._execute_request(provider, task_type, request_data)

                # Track usage and cost
                self._track_usage(provider, task_type, request_data)

                # Reset failure count on success
                self.failure_counts[provider.name] = 0

                return {
                    "provider": provider.name,
                    "result": result,
                    "cost": self._calculate_cost(provider, request_data),
                    "quality_score": provider.quality_score
                }

            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                self.failure_counts[provider.name] += 1
                continue

        raise Exception(f"All providers failed for {task_type}")

    async def _execute_request(
        self,
        provider: APIProvider,
        task_type: TaskType,
        request_data: Dict[str, Any]
    ) -> Any:
        """Execute request to specific provider"""

        api_key = os.getenv(provider.api_key_env)
        if not api_key and provider.cost_per_unit > 0:
            raise Exception(f"API key not found for {provider.name}")

        # Provider-specific request formatting
        formatted_request = self._format_request(provider, task_type, request_data)

        async with aiohttp.ClientSession() as session:
            headers = self._get_headers(provider, api_key)

            async with session.post(
                provider.endpoint,
                json=formatted_request,
                headers=headers
            ) as response:
                if response.status != 200:
                    raise Exception(f"API error: {response.status}")

                return await response.json()

    def _format_request(
        self,
        provider: APIProvider,
        task_type: TaskType,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format request for specific provider"""

        # Provider-specific formatting
        if provider.name.startswith("deepseek"):
            return {
                "model": "deepseek-chat",
                "messages": request_data.get("messages", []),
                "temperature": request_data.get("temperature", 0.7)
            }

        elif provider.name.startswith("gemini"):
            return {
                "contents": [{
                    "parts": [{
                        "text": request_data.get("prompt", "")
                    }]
                }],
                "generationConfig": {
                    "temperature": 0.9,
                    "topK": 1,
                    "topP": 1,
                    "maxOutputTokens": 2048
                }
            }

        elif provider.name.startswith("together"):
            if task_type == TaskType.IMAGE_GENERATION:
                return {
                    "model": "stabilityai/stable-diffusion-xl-base-1.0",
                    "prompt": request_data.get("prompt", ""),
                    "width": 1024,
                    "height": 1024,
                    "steps": 20,
                    "n": 1
                }
            else:
                return {
                    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                    "messages": request_data.get("messages", []),
                    "temperature": 0.7,
                    "max_tokens": 2048
                }

        elif provider.name.startswith("huggingface"):
            return {
                "inputs": request_data.get("prompt", ""),
                "parameters": {
                    "max_length": 1000,
                    "temperature": 0.7
                }
            }

        # Default format
        return request_data

    def _get_headers(self, provider: APIProvider, api_key: str) -> Dict[str, str]:
        """Get headers for specific provider"""

        if provider.name.startswith("deepseek"):
            return {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

        elif provider.name.startswith("gemini"):
            return {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }

        elif provider.name.startswith("openrouter"):
            return {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/elliottsax/engineer",
                "X-Title": "YouTube Video Generator"
            }

        elif provider.name.startswith("huggingface"):
            return {
                "Authorization": f"Bearer {api_key}"
            }

        # Default headers
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def _track_usage(
        self,
        provider: APIProvider,
        task_type: TaskType,
        request_data: Dict[str, Any]
    ):
        """Track usage for cost calculation and limits"""

        # Update usage counters
        if task_type == TaskType.SCRIPT_GENERATION:
            # Estimate tokens
            tokens = len(str(request_data).split()) * 1.5
            provider.current_usage += tokens
            self.usage_tracker[f"{provider.name}_tokens"] += tokens

        elif task_type == TaskType.IMAGE_GENERATION:
            provider.current_usage += 1
            self.usage_tracker[f"{provider.name}_images"] += 1

        elif task_type == TaskType.AUDIO_GENERATION:
            chars = len(request_data.get("text", ""))
            provider.current_usage += chars
            self.usage_tracker[f"{provider.name}_chars"] += chars

    def _calculate_cost(
        self,
        provider: APIProvider,
        request_data: Dict[str, Any]
    ) -> float:
        """Calculate cost for specific request"""

        if provider.cost_per_unit == 0:
            return 0.0

        # Calculate based on request type
        if "messages" in request_data or "prompt" in request_data:
            # Text generation - estimate tokens
            text = str(request_data)
            tokens = len(text.split()) * 1.5 / 1000  # Convert to K tokens
            cost = (tokens / 1000) * provider.cost_per_unit  # Most prices are per M tokens
        else:
            # Image/video - per unit
            cost = provider.cost_per_unit

        self.cost_tracker[provider.name] += cost
        return cost

    def get_cost_report(self) -> Dict[str, Any]:
        """Get detailed cost report"""

        total_cost = sum(self.cost_tracker.values())

        return {
            "total_cost": total_cost,
            "by_provider": dict(self.cost_tracker),
            "by_task_type": self._get_costs_by_task(),
            "usage_stats": dict(self.usage_tracker),
            "free_tier_remaining": self._get_free_tier_status()
        }

    def _get_costs_by_task(self) -> Dict[str, float]:
        """Group costs by task type"""

        costs = {
            "script": 0.0,
            "image": 0.0,
            "video": 0.0,
            "audio": 0.0
        }

        for provider_name, cost in self.cost_tracker.items():
            for task_type, providers in self.providers.items():
                if any(p.name == provider_name for p in providers):
                    task_name = task_type.value
                    costs[task_name] += cost
                    break

        return costs

    def _get_free_tier_status(self) -> Dict[str, Dict[str, Any]]:
        """Get remaining free tier quotas"""

        status = {}

        for task_type, providers in self.providers.items():
            for provider in providers:
                if provider.free_tier_limit:
                    remaining = provider.free_tier_limit - provider.current_usage
                    status[provider.name] = {
                        "limit": provider.free_tier_limit,
                        "used": provider.current_usage,
                        "remaining": max(0, remaining),
                        "percentage_used": (provider.current_usage / provider.free_tier_limit) * 100
                    }

        return status

    async def batch_route(
        self,
        requests: List[Dict[str, Any]],
        task_type: TaskType,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Route multiple requests efficiently"""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_request(req):
            async with semaphore:
                return await self.route_request(task_type, req)

        tasks = [process_request(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)


class CostOptimizer:
    """Optimize costs across entire video generation pipeline"""

    def __init__(self, target_cost: float = 0.50):
        self.target_cost = target_cost
        self.router = CheapAPIRouter()

    def optimize_pipeline(
        self,
        num_scenes: int,
        video_duration_minutes: float
    ) -> Dict[str, Any]:
        """
        Optimize cost allocation across pipeline stages
        Returns recommended API selection for each stage
        """

        # Budget allocation (% of target cost)
        budget_allocation = {
            "script": 0.10,  # 10% for script generation
            "images": 0.30,  # 30% for image generation
            "video": 0.40,   # 40% for video effects
            "audio": 0.20    # 20% for TTS
        }

        recommendations = {}

        # Script generation
        script_budget = self.target_cost * budget_allocation["script"]
        recommendations["script"] = {
            "primary": "deepseek_v3",  # Ultra cheap, high quality
            "fallback": "huggingface_zephyr",  # Free tier
            "budget": script_budget
        }

        # Image generation
        image_budget = self.target_cost * budget_allocation["images"]
        images_per_scene = 1
        total_images = num_scenes * images_per_scene

        recommendations["images"] = {
            "primary": "gemini_imagen3",  # FREE!
            "fallback": "together_sdxl",  # $0.0006 per image
            "budget": image_budget,
            "strategy": "Use Gemini for all if under quota, else mix with Together"
        }

        # Video generation
        video_budget = self.target_cost * budget_allocation["video"]
        recommendations["video"] = {
            "strategy": "static_with_transitions",  # Mostly static images
            "ai_video_percentage": 0.1,  # Only 10% AI video
            "primary": "huggingface_svd",
            "budget": video_budget
        }

        # Audio generation
        audio_budget = self.target_cost * budget_allocation["audio"]
        total_chars = video_duration_minutes * 60 * 150  # ~150 chars/min

        recommendations["audio"] = {
            "primary": "elevenlabs_free" if total_chars < 10000 else "coqui_xtts",
            "fallback": "google_tts",
            "budget": audio_budget
        }

        # Calculate expected total cost
        expected_cost = self._calculate_expected_cost(recommendations, num_scenes, video_duration_minutes)

        return {
            "recommendations": recommendations,
            "expected_cost": expected_cost,
            "target_cost": self.target_cost,
            "under_budget": expected_cost <= self.target_cost
        }

    def _calculate_expected_cost(
        self,
        recommendations: Dict,
        num_scenes: int,
        duration_minutes: float
    ) -> float:
        """Calculate expected total cost"""

        cost = 0.0

        # Script: ~2000 tokens
        if recommendations["script"]["primary"] == "deepseek_v3":
            cost += (2 / 1000) * 0.14  # 2K tokens at $0.14/M

        # Images: Gemini is free, fallback costs
        if recommendations["images"]["primary"] != "gemini_imagen3":
            cost += num_scenes * 0.0006  # Together AI price

        # Video: Mostly transitions, minimal AI
        ai_scenes = int(num_scenes * 0.1)
        cost += ai_scenes * 0.005  # Minimal video cost

        # Audio: Check if free tier
        total_chars = duration_minutes * 60 * 150
        if total_chars > 10000:  # Exceeds ElevenLabs free
            cost += (total_chars / 1000) * 0.05  # Coqui price

        return cost