"""
Hybrid AI Video Generation Pipeline
Combines Filmora AI models with Once automation for YouTube explainer videos
Target: 5-8 minute videos, $3-10 cost, minimalist stick figure style
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
import time
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Cost-based model tiers"""
    ECONOMY = "economy"      # Kelin model - $0.50-1 per scene
    STANDARD = "standard"     # Veo 3.0 - $1-2 per scene
    PREMIUM = "premium"       # Veo 3.1 - $2-3 per scene


class SceneImportance(Enum):
    """Scene importance for model selection"""
    LOW = "low"          # B-roll, transitions
    MEDIUM = "medium"    # Supporting content
    HIGH = "high"        # Key explanations, hooks


@dataclass
class SceneConfig:
    """Configuration for a single scene"""
    scene_number: int
    prompt: str
    visual_prompt: str
    duration: int  # seconds
    importance: SceneImportance
    model_tier: Optional[ModelTier] = None
    narration: Optional[str] = None


class MinimalistPromptGenerator:
    """Generate prompts for minimalist stick figure style"""

    STYLE_TEMPLATE = """Minimal vector illustration, {figures} with thick consistent black outline,
    perfectly round head with two small dot eyes, {pose}, {background} gradient background
    with {lighting}, {composition}, clean professional educational style, no text, 16:9 aspect ratio"""

    BACKGROUNDS = {
        "intro": "blue-to-purple",
        "explanation": "teal-to-navy",
        "example": "warm coral-to-amber",
        "thinking": "violet-to-blue",
        "success": "golden-to-orange",
        "conclusion": "green-to-teal"
    }

    POSES = {
        "explaining": "confident standing pose with one arm extended palm-up in explaining gesture",
        "pointing": "standing in profile view pointing confidently at {target}",
        "thinking": "contemplative pose with one hand raised to chin and slight head tilt",
        "excited": "dynamic pose with arms spread wide and body leaning forward excitedly",
        "teaching": "left figure in teaching pose pointing upward, right figure with hand on chin",
        "working": "seated at simple line-drawn desk with minimal rectangular laptop shape",
        "success": "triumphant pose with both arms raised overhead in victory gesture"
    }

    def generate_visual_prompt(self, scene_type: str, content: str,
                              figures: int = 1, icons: List[str] = None) -> str:
        """Generate a complete visual prompt for stick figure scene"""

        # Select appropriate background
        background = self.BACKGROUNDS.get(scene_type, "blue-to-purple")

        # Select pose based on content
        if "explain" in content.lower():
            pose = self.POSES["explaining"]
        elif "think" in content.lower() or "consider" in content.lower():
            pose = self.POSES["thinking"]
        elif "success" in content.lower() or "achieve" in content.lower():
            pose = self.POSES["success"]
        elif "point" in content.lower() or "show" in content.lower():
            pose = self.POSES["pointing"].format(target="floating chart icon")
        else:
            pose = self.POSES["explaining"]

        # Configure figures
        figure_desc = "single white stick figure" if figures == 1 else f"{figures} white stick figures"

        # Add icons if specified
        icon_desc = ""
        if icons:
            icon_desc = f", {', '.join(icons)} floating nearby in matching black outline style"

        # Configure lighting
        lighting = "soft radial glow behind figure" if scene_type in ["intro", "conclusion"] else "subtle centered lighting"

        # Composition
        composition = "generous negative space, uncluttered composition"
        if icons:
            composition = "balanced composition with figure and icons"

        prompt = self.STYLE_TEMPLATE.format(
            figures=figure_desc,
            pose=pose + icon_desc,
            background=background,
            lighting=lighting,
            composition=composition
        )

        return prompt


class FilmoraModelInterface:
    """Interface to Filmora AI models"""

    def __init__(self):
        self.model_configs = {
            "kelin": {
                "workflow_id": "1804422399892127744",
                "point_code": "klm_text2video",
                "alg_code": "klm_text2video",
                "cost_per_second": 0.10,
                "max_duration": 5
            },
            "veo3": {
                "workflow_id": "46",
                "point_code": "combo_text2video_veo3",
                "alg_code": "google_text2video",
                "model": "veo-3.0-fast-generate-preview",
                "cost_per_second": 0.25,
                "max_duration": 8
            },
            "veo3.1": {
                "workflow_id": "45",
                "point_code": "combo_img2video_veo3",
                "alg_code": "google_img2video",
                "model": "veo-3.1-fast-generate-preview",
                "cost_per_second": 0.35,
                "max_duration": 8
            }
        }

    def select_model_for_scene(self, scene: SceneConfig) -> Tuple[str, Dict]:
        """Select optimal model based on scene importance and cost"""

        # Map importance to model
        if scene.importance == SceneImportance.LOW:
            model_name = "kelin"
        elif scene.importance == SceneImportance.HIGH:
            model_name = "veo3.1"
        else:
            model_name = "veo3"

        config = self.model_configs[model_name].copy()

        # Calculate cost
        duration = min(scene.duration, config["max_duration"])
        cost = duration * config["cost_per_second"]

        return model_name, {
            "config": config,
            "duration": duration,
            "cost": cost
        }

    def generate_api_request(self, model_name: str, prompt: str,
                            duration: int, resolution: str = "1080p") -> Dict:
        """Generate API request for Filmora model"""

        config = self.model_configs[model_name]

        if model_name == "kelin":
            return {
                "workflow_id": config["workflow_id"],
                "point_code": config["point_code"],
                "params": {
                    "alg_code": config["alg_code"],
                    "prompt": prompt,
                    "duration_string": f"{duration}s",
                    "aspect_ratio": "16:9",
                    "mode": "std"
                }
            }
        else:
            return {
                "workflow_id": config["workflow_id"],
                "point_code": config["point_code"],
                "params": {
                    "alg_code": config["alg_code"],
                    "prompt": prompt,
                    "duration": duration,
                    "model": config.get("model"),
                    "resolution": resolution,
                    "aspect_ratio": "16:9"
                }
            }


class HybridVideoGenerator:
    """Main pipeline orchestrator combining Once and Filmora"""

    def __init__(self):
        self.prompt_generator = MinimalistPromptGenerator()
        self.filmora = FilmoraModelInterface()
        self.total_cost = 0.0
        self.scenes = []

    async def generate_video(self, topic: str, target_duration: int = 360,
                            max_budget: float = 10.0) -> Dict:
        """
        Generate complete video from topic
        Args:
            topic: Video topic/title
            target_duration: Target duration in seconds (default 6 minutes)
            max_budget: Maximum budget in dollars
        """

        logger.info(f"Starting video generation for topic: {topic}")
        start_time = time.time()

        # Step 1: Generate script and scene breakdown
        script_data = await self.generate_script(topic, target_duration)

        # Step 2: Analyze scenes and assign importance
        scenes = self.analyze_scene_importance(script_data["scenes"])

        # Step 3: Optimize model selection for budget
        scenes = self.optimize_model_selection(scenes, max_budget)

        # Step 4: Generate visual prompts for each scene
        for scene in scenes:
            scene.visual_prompt = self.prompt_generator.generate_visual_prompt(
                scene_type=self._determine_scene_type(scene),
                content=scene.narration,
                figures=1 if "two" not in scene.narration.lower() else 2,
                icons=self._extract_icons(scene.narration)
            )

        # Step 5: Generate videos for each scene
        video_segments = await self.generate_video_segments(scenes)

        # Step 6: Generate narration
        narration_tracks = await self.generate_narration(scenes)

        # Step 7: Compose final video
        final_video = await self.compose_video(video_segments, narration_tracks)

        # Calculate metrics
        total_time = time.time() - start_time

        return {
            "video_path": final_video["path"],
            "duration": final_video["duration"],
            "total_cost": self.total_cost,
            "generation_time": total_time,
            "scenes": len(scenes),
            "metrics": {
                "cost_per_minute": self.total_cost / (target_duration / 60),
                "time_per_minute": total_time / (target_duration / 60),
                "model_distribution": self._get_model_distribution(scenes)
            }
        }

    async def generate_script(self, topic: str, target_duration: int) -> Dict:
        """Generate script using GPT-4 style approach"""

        # This would integrate with Once's script generation
        # For now, creating a structured example

        scenes_per_minute = 12  # 5-second average scenes
        total_scenes = (target_duration // 60) * scenes_per_minute

        script = {
            "title": f"The Hidden Truth About {topic}",
            "hook": f"What if I told you everything you know about {topic} is wrong?",
            "scenes": []
        }

        # Generate scene structure
        scene_types = ["intro", "problem", "explanation", "example",
                      "deeper_dive", "counterpoint", "conclusion"]

        for i in range(total_scenes):
            scene_type = scene_types[min(i, len(scene_types)-1)]
            scene = {
                "number": i + 1,
                "type": scene_type,
                "narration": f"Scene {i+1} content about {topic}...",
                "duration": 5,
                "visual_notes": "Minimalist stick figure explaining concept"
            }

            # Add signature phrase periodically
            if i % 5 == 0 or i == total_scenes - 1:
                scene["narration"] += " Once you know how this works, everything changes."

            script["scenes"].append(scene)

        return script

    def analyze_scene_importance(self, scenes: List[Dict]) -> List[SceneConfig]:
        """Analyze and assign importance to scenes"""

        scene_configs = []

        for i, scene in enumerate(scenes):
            # Determine importance based on position and type
            if i == 0 or i == len(scenes) - 1:  # First and last scenes
                importance = SceneImportance.HIGH
            elif scene["type"] in ["problem", "explanation"]:
                importance = SceneImportance.HIGH
            elif scene["type"] in ["example", "deeper_dive"]:
                importance = SceneImportance.MEDIUM
            else:
                importance = SceneImportance.LOW

            config = SceneConfig(
                scene_number=scene["number"],
                prompt=scene.get("visual_notes", ""),
                visual_prompt="",  # Will be generated
                duration=scene["duration"],
                importance=importance,
                narration=scene["narration"]
            )
            scene_configs.append(config)

        return scene_configs

    def optimize_model_selection(self, scenes: List[SceneConfig],
                                max_budget: float) -> List[SceneConfig]:
        """Optimize model selection to stay within budget"""

        total_cost = 0.0

        # First pass: assign models based on importance
        for scene in scenes:
            model_name, model_info = self.filmora.select_model_for_scene(scene)
            scene.model_tier = ModelTier[model_name.upper()]
            total_cost += model_info["cost"]

        # If over budget, downgrade some scenes
        if total_cost > max_budget:
            # Sort by importance (low to high)
            sorted_scenes = sorted(scenes,
                                  key=lambda x: x.importance.value)

            for scene in sorted_scenes:
                if total_cost <= max_budget:
                    break

                if scene.importance != SceneImportance.HIGH:
                    # Downgrade to economy
                    old_cost = self.filmora.model_configs[
                        scene.model_tier.value]["cost_per_second"] * scene.duration
                    new_cost = self.filmora.model_configs[
                        "kelin"]["cost_per_second"] * scene.duration

                    total_cost = total_cost - old_cost + new_cost
                    scene.model_tier = ModelTier.ECONOMY

        self.total_cost = total_cost
        logger.info(f"Optimized cost: ${total_cost:.2f} (budget: ${max_budget})")

        return scenes

    async def generate_video_segments(self, scenes: List[SceneConfig]) -> List[Dict]:
        """Generate video segments for all scenes"""

        segments = []

        # Batch processing for efficiency
        batch_size = 5
        for i in range(0, len(scenes), batch_size):
            batch = scenes[i:i+batch_size]
            batch_tasks = []

            for scene in batch:
                task = self._generate_single_segment(scene)
                batch_tasks.append(task)

            batch_results = await asyncio.gather(*batch_tasks)
            segments.extend(batch_results)

        return segments

    async def _generate_single_segment(self, scene: SceneConfig) -> Dict:
        """Generate a single video segment"""

        model_name = scene.model_tier.value if scene.model_tier else "kelin"

        # Generate API request
        request = self.filmora.generate_api_request(
            model_name=model_name,
            prompt=scene.visual_prompt,
            duration=scene.duration
        )

        # Simulate API call (in production, would call actual API)
        await asyncio.sleep(0.1)  # Simulate network delay

        return {
            "scene_number": scene.scene_number,
            "video_path": f"temp/scene_{scene.scene_number}.mp4",
            "duration": scene.duration,
            "model": model_name,
            "cost": self.filmora.model_configs[model_name]["cost_per_second"] * scene.duration
        }

    async def generate_narration(self, scenes: List[SceneConfig]) -> List[Dict]:
        """Generate narration using ElevenLabs style TTS"""

        narration_tracks = []

        for scene in scenes:
            # Would integrate with actual TTS API
            track = {
                "scene_number": scene.scene_number,
                "audio_path": f"temp/narration_{scene.scene_number}.mp3",
                "text": scene.narration,
                "duration": scene.duration
            }
            narration_tracks.append(track)

        return narration_tracks

    async def compose_video(self, video_segments: List[Dict],
                           narration_tracks: List[Dict]) -> Dict:
        """Compose final video using Remotion or FFmpeg"""

        # In production, would use actual video composition
        total_duration = sum(seg["duration"] for seg in video_segments)

        return {
            "path": "output/final_video.mp4",
            "duration": total_duration,
            "resolution": "1920x1080",
            "fps": 30,
            "codec": "h264"
        }

    def _determine_scene_type(self, scene: SceneConfig) -> str:
        """Determine scene type for visual styling"""

        if scene.scene_number == 1:
            return "intro"
        elif "think" in scene.narration.lower():
            return "thinking"
        elif "example" in scene.narration.lower():
            return "example"
        elif "success" in scene.narration.lower():
            return "success"
        elif scene.importance == SceneImportance.HIGH:
            return "explanation"
        else:
            return "explanation"

    def _extract_icons(self, narration: str) -> List[str]:
        """Extract relevant icons from narration"""

        icons = []

        icon_keywords = {
            "dollar": "simple black-outline dollar sign icon",
            "money": "simple black-outline dollar sign icon",
            "idea": "simple black-outline lightbulb icon",
            "think": "simple question mark icon",
            "data": "simple floating bar chart icon",
            "success": "simple checkmark icon",
            "time": "simple clock icon",
            "growth": "simple upward arrow icon"
        }

        narration_lower = narration.lower()
        for keyword, icon in icon_keywords.items():
            if keyword in narration_lower and len(icons) < 3:
                icons.append(icon)

        return icons

    def _get_model_distribution(self, scenes: List[SceneConfig]) -> Dict:
        """Get distribution of models used"""

        distribution = {"economy": 0, "standard": 0, "premium": 0}

        for scene in scenes:
            if scene.model_tier:
                distribution[scene.model_tier.value] += 1

        return distribution


async def main():
    """Main entry point for testing"""

    generator = HybridVideoGenerator()

    # Test generation
    result = await generator.generate_video(
        topic="How AI is Secretly Running the Stock Market",
        target_duration=360,  # 6 minutes
        max_budget=8.0  # $8 budget
    )

    print(f"""
    Video Generation Complete!
    -------------------------
    Video Path: {result['video_path']}
    Duration: {result['duration']} seconds
    Total Cost: ${result['total_cost']:.2f}
    Generation Time: {result['generation_time']:.1f} seconds

    Metrics:
    - Cost per minute: ${result['metrics']['cost_per_minute']:.2f}
    - Time per minute: {result['metrics']['time_per_minute']:.1f}s
    - Model Distribution: {result['metrics']['model_distribution']}
    """)


if __name__ == "__main__":
    asyncio.run(main())