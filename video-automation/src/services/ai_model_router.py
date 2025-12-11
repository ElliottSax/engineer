"""
AI Model Router Service - Inspired by Wondershare Filmora's Multi-Model Architecture

This module routes video generation requests to optimal AI models based on:
- Scene type and importance
- Budget constraints
- Quality requirements
- Available features (auto-audio, etc.)

Combines insights from Wondershare Filmora reverse engineering with the "once" pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from ..models.video_request import Scene, SceneType, VideoQuality

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """AI model providers."""
    GOOGLE = "google"          # Veo 3.0/3.1
    OPENAI = "openai"          # Sora 2
    KELIN = "kelin"            # Proprietary fast model
    PROPRIETARY = "proprietary" # Other internal models


class VideoGenerationType(str, Enum):
    """Types of video generation."""
    TEXT_TO_VIDEO = "text_to_video"
    IMAGE_TO_VIDEO = "image_to_video"
    KEYFRAME_TO_VIDEO = "keyframe_to_video"


@dataclass
class ModelConfig:
    """Configuration for an AI model."""
    workflow_id: str
    point_code: str
    alg_code: str
    model: str
    provider: ModelProvider
    duration: int
    cost_per_clip: float
    features: List[str]
    params: Dict[str, Any]
    resolution: Optional[str] = "720p"
    aspect_ratio: Optional[str] = "16:9"


@dataclass
class ModelSelection:
    """Result of model selection."""
    model_config: ModelConfig
    generation_type: VideoGenerationType
    reason: str
    estimated_cost: float


class AIModelRouter:
    """
    Routes AI generation requests to optimal models.

    Inspired by Wondershare Filmora's modular configuration system,
    this router selects the best AI model for each scene based on:
    - Scene type and importance
    - Budget constraints
    - Quality settings
    - Feature requirements
    """

    def __init__(self, config_path: Path):
        """
        Initialize the AI model router.

        Args:
            config_path: Path to ai_models.json configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.models = self._parse_models()

    def _load_config(self) -> Dict[str, Any]:
        """Load AI model configuration from JSON."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"AI model config not found: {self.config_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            raise

    def _parse_models(self) -> Dict[str, ModelConfig]:
        """Parse model configurations into ModelConfig objects."""
        models = {}

        for gen_type, model_dict in self.config.items():
            if gen_type in ["model_selection_rules", "cost_optimization"]:
                continue  # Skip metadata

            for model_name, model_data in model_dict.items():
                key = f"{gen_type}.{model_name}"
                models[key] = ModelConfig(
                    workflow_id=model_data["workflow_id"],
                    point_code=model_data["point_code"],
                    alg_code=model_data["alg_code"],
                    model=model_data["model"],
                    provider=ModelProvider(model_data["provider"]),
                    duration=model_data["duration"],
                    cost_per_clip=model_data["cost_per_clip"],
                    features=model_data["features"],
                    params=model_data["params"],
                    resolution=model_data.get("resolution"),
                    aspect_ratio=model_data.get("aspect_ratio")
                )

        return models

    def select_model(
        self,
        scene: Scene,
        budget_tier: str = "standard",
        quality: VideoQuality = VideoQuality.STANDARD,
        has_image: bool = False
    ) -> ModelSelection:
        """
        Select the best AI model for a scene.

        Args:
            scene: Scene object to generate video for
            budget_tier: Budget tier (economy, standard, premium)
            quality: Quality setting
            has_image: Whether scene has an associated image

        Returns:
            ModelSelection with chosen model and reasoning
        """
        # Determine generation type
        if has_image:
            gen_type = VideoGenerationType.IMAGE_TO_VIDEO
        else:
            gen_type = VideoGenerationType.TEXT_TO_VIDEO

        # Get selection rules
        rules = self.config.get("model_selection_rules", {})

        # Determine model based on scene type
        model_key = self._get_model_key_for_scene(
            scene, gen_type, budget_tier, quality, rules
        )

        model_config = self.models.get(model_key)
        if not model_config:
            logger.warning(f"Model not found: {model_key}, using fallback")
            model_config = self._get_fallback_model(gen_type)

        # Calculate estimated cost
        estimated_cost = model_config.cost_per_clip

        reason = self._get_selection_reason(scene, model_key, rules)

        return ModelSelection(
            model_config=model_config,
            generation_type=gen_type,
            reason=reason,
            estimated_cost=estimated_cost
        )

    def _get_model_key_for_scene(
        self,
        scene: Scene,
        gen_type: VideoGenerationType,
        budget_tier: str,
        quality: VideoQuality,
        rules: Dict[str, Any]
    ) -> str:
        """Determine model key based on scene characteristics."""

        # Title scenes - use high quality
        if scene.scene_type == SceneType.TITLE:
            if gen_type == VideoGenerationType.TEXT_TO_VIDEO:
                return "text_to_video.veo3"
            else:
                return "image_to_video.sora2"

        # Conclusion scenes - use high quality
        if scene.scene_type == SceneType.CONCLUSION:
            if gen_type == VideoGenerationType.TEXT_TO_VIDEO:
                return "text_to_video.veo3"
            else:
                return "image_to_video.sora2"

        # High importance concept scenes
        if scene.scene_type == SceneType.CONCEPT:
            # Check if marked as important (e.g., keywords length > 3)
            is_important = len(scene.keywords) > 3

            if is_important and quality != VideoQuality.ECONOMY:
                if gen_type == VideoGenerationType.TEXT_TO_VIDEO:
                    return "text_to_video.veo3"
                else:
                    return "image_to_video.sora2"
            else:
                # Standard concept scenes - use cost-effective models
                if gen_type == VideoGenerationType.TEXT_TO_VIDEO:
                    return "text_to_video.kelin"
                else:
                    return "image_to_video.standard_2"

        # Image-to-video transitions
        if gen_type == VideoGenerationType.IMAGE_TO_VIDEO:
            if budget_tier == "premium" or quality == VideoQuality.HD:
                return "image_to_video.sora2"
            else:
                return "image_to_video.standard_2"

        # Default fallback
        if gen_type == VideoGenerationType.TEXT_TO_VIDEO:
            if budget_tier == "economy":
                return "text_to_video.kelin"
            else:
                return "text_to_video.veo3"
        else:
            return "image_to_video.standard_2"

    def _get_fallback_model(self, gen_type: VideoGenerationType) -> ModelConfig:
        """Get fallback model if primary selection fails."""
        if gen_type == VideoGenerationType.TEXT_TO_VIDEO:
            return self.models["text_to_video.kelin"]
        elif gen_type == VideoGenerationType.IMAGE_TO_VIDEO:
            return self.models["image_to_video.standard_2"]
        else:
            raise ValueError(f"No fallback for generation type: {gen_type}")

    def _get_selection_reason(
        self,
        scene: Scene,
        model_key: str,
        rules: Dict[str, Any]
    ) -> str:
        """Get human-readable reason for model selection."""
        if scene.scene_type == SceneType.TITLE:
            return "High quality with auto-audio for impactful opening"
        elif scene.scene_type == SceneType.CONCLUSION:
            return "Strong ending with quality and audio"
        elif "veo3" in model_key:
            return "Key scene requiring high quality and auto-audio"
        elif "sora2" in model_key:
            return "Smooth high-quality image-to-video transition"
        elif "kelin" in model_key:
            return "Cost-effective generation for standard scenes"
        else:
            return "Standard model selection"

    def estimate_video_cost(
        self,
        scenes: List[Scene],
        budget_tier: str = "standard",
        quality: VideoQuality = VideoQuality.STANDARD
    ) -> Dict[str, Any]:
        """
        Estimate total cost for generating all scenes.

        Args:
            scenes: List of scenes to generate
            budget_tier: Budget tier
            quality: Quality setting

        Returns:
            Cost breakdown dictionary
        """
        total_cost = 0.0
        model_usage = {}
        scene_costs = []

        for scene in scenes:
            # Check if scene has image
            has_image = scene.image_path is not None

            selection = self.select_model(
                scene=scene,
                budget_tier=budget_tier,
                quality=quality,
                has_image=has_image
            )

            total_cost += selection.estimated_cost

            # Track model usage
            model_name = f"{selection.generation_type}.{selection.model_config.model}"
            if model_name not in model_usage:
                model_usage[model_name] = {
                    "count": 0,
                    "cost": 0.0
                }
            model_usage[model_name]["count"] += 1
            model_usage[model_name]["cost"] += selection.estimated_cost

            scene_costs.append({
                "scene_id": scene.scene_id,
                "scene_type": scene.scene_type.value,
                "model": selection.model_config.model,
                "cost": selection.estimated_cost,
                "reason": selection.reason
            })

        return {
            "total_cost": round(total_cost, 2),
            "model_usage": model_usage,
            "scene_costs": scene_costs,
            "scenes_count": len(scenes),
            "average_cost_per_scene": round(total_cost / len(scenes), 2) if scenes else 0
        }

    def get_available_models(
        self,
        generation_type: Optional[VideoGenerationType] = None
    ) -> List[ModelConfig]:
        """
        Get list of available models.

        Args:
            generation_type: Optional filter by generation type

        Returns:
            List of ModelConfig objects
        """
        if generation_type:
            prefix = f"{generation_type.value}."
            return [
                model for key, model in self.models.items()
                if key.startswith(prefix)
            ]
        return list(self.models.values())

    def validate_params(
        self,
        model_config: ModelConfig,
        params: Dict[str, Any]
    ) -> tuple[bool, List[str]]:
        """
        Validate parameters against model configuration.

        Args:
            model_config: Model configuration
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        for param_name, param_spec in model_config.params.items():
            value = params.get(param_name)

            # Check required params
            if param_spec.get("required") and value is None:
                errors.append(f"Missing required parameter: {param_name}")
                continue

            # Skip validation if value is None and not required
            if value is None:
                continue

            # Type validation
            param_type = param_spec.get("type")
            if param_type == "string" and not isinstance(value, str):
                errors.append(f"{param_name} must be a string")
            elif param_type == "integer" and not isinstance(value, int):
                errors.append(f"{param_name} must be an integer")

            # String length validation
            if param_type == "string" and isinstance(value, str):
                min_len = param_spec.get("min_length")
                max_len = param_spec.get("max_length")
                if min_len and len(value) < min_len:
                    errors.append(f"{param_name} too short (min: {min_len})")
                if max_len and len(value) > max_len:
                    errors.append(f"{param_name} too long (max: {max_len})")

            # Enum validation
            if "enum" in param_spec:
                if value not in param_spec["enum"]:
                    errors.append(
                        f"{param_name} must be one of: {param_spec['enum']}"
                    )

            # Integer range validation
            if param_type == "integer":
                min_val = param_spec.get("min")
                max_val = param_spec.get("max")
                if min_val is not None and value < min_val:
                    errors.append(f"{param_name} too small (min: {min_val})")
                if max_val is not None and value > max_val:
                    errors.append(f"{param_name} too large (max: {max_val})")

        return (len(errors) == 0, errors)


# Example usage
if __name__ == "__main__":
    from pathlib import Path

    # Initialize router
    config_path = Path(__file__).parent.parent.parent / "config" / "ai_models.json"
    router = AIModelRouter(config_path)

    # Create example scene
    example_scene = Scene(
        scene_id="scene_001",
        scene_type=SceneType.TITLE,
        narration_text="Welcome to AI Video Generation",
        visual_description="Dynamic title animation with particles",
        start_time=0.0,
        duration=8.0,
        keywords=["title", "welcome", "AI", "video"],
        animation_style="dynamic"
    )

    # Select model
    selection = router.select_model(example_scene, budget_tier="standard")

    print(f"Selected model: {selection.model_config.model}")
    print(f"Provider: {selection.model_config.provider}")
    print(f"Reason: {selection.reason}")
    print(f"Estimated cost: ${selection.estimated_cost}")
    print(f"Features: {selection.model_config.features}")
