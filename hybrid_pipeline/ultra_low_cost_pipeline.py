#!/usr/bin/env python3
"""
Ultra-Low-Cost Video Generation Pipeline
Complete YouTube video generation for < $0.50
Combines cheapest APIs with intelligent optimization
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import subprocess

# Import our modules
from cheap_api_router import CheapAPIRouter, TaskType, CostOptimizer
from gemini_image_gen import GeminiImageGenerator, StickFigureAssetLibrary
from script_generator import ExplainerScriptGenerator, NarrationOptimizer
from prompt_system import MinimalistSceneGenerator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VideoGenerationResult:
    """Result of video generation pipeline"""
    video_path: str
    total_cost: float
    generation_time: float
    scene_count: int
    providers_used: Dict[str, int]
    cost_breakdown: Dict[str, float]
    success: bool = True
    error: Optional[str] = None


class UltraLowCostVideoPipeline:
    """
    Main orchestrator for ultra-cheap video generation
    Target: < $0.50 per complete 5-8 minute video
    """

    def __init__(self, config_path: Optional[str] = None):
        # Initialize components
        self.api_router = CheapAPIRouter(quality_threshold=0.7)
        self.cost_optimizer = CostOptimizer(target_cost=0.50)
        self.gemini_gen = GeminiImageGenerator()
        self.asset_library = StickFigureAssetLibrary(self.gemini_gen)
        self.script_gen = ExplainerScriptGenerator()
        self.narration_opt = NarrationOptimizer()
        self.scene_gen = MinimalistSceneGenerator()

        # Configuration
        self.config = self._load_config(config_path)

        # Setup directories
        self._setup_directories()

        # Cost tracking
        self.total_cost = 0.0
        self.cost_breakdown = {
            "script": 0.0,
            "images": 0.0,
            "video": 0.0,
            "audio": 0.0,
            "processing": 0.0
        }

        # Performance tracking
        self.start_time = None
        self.stage_times = {}

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration settings"""

        default_config = {
            "target_duration_minutes": 6,
            "target_cost_dollars": 0.50,
            "prefer_free_tier": True,
            "use_cache": True,
            "parallel_processing": True,
            "output_resolution": "1080p",
            "output_fps": 30,
            "video_strategy": "static_with_transitions",  # vs "ai_generated"
            "ai_video_percentage": 0.1,  # Only 10% AI video
            "audio_voice": "coqui_default",
            "background_music_volume": 0.1
        }

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                default_config.update(custom_config)

        return default_config

    def _setup_directories(self):
        """Create necessary directories"""

        dirs = [
            "output/videos",
            "output/scripts",
            "temp/images",
            "temp/audio",
            "temp/video_segments",
            "cache/gemini_images",
            "cache/audio",
            "logs/cost_reports",
            "assets/stick_figures"
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    async def generate_video(
        self,
        topic: str,
        rush_mode: bool = False,
        max_budget: Optional[float] = None
    ) -> VideoGenerationResult:
        """
        Generate complete video using ultra-cheap APIs

        Args:
            topic: Video topic/subject
            rush_mode: Prioritize speed over quality
            max_budget: Maximum budget override (default 0.50)

        Returns:
            VideoGenerationResult with path, cost, and metrics
        """

        self.start_time = time.time()
        max_budget = max_budget or self.config["target_cost_dollars"]

        logger.info("="*60)
        logger.info(f"Ultra-Low-Cost Video Generation: {topic}")
        logger.info(f"Budget: ${max_budget:.2f} | Mode: {'Rush' if rush_mode else 'Normal'}")
        logger.info("="*60)

        try:
            # Stage 1: Generate Script (Target: $0.02)
            script = await self._generate_cheap_script(topic, rush_mode)

            # Stage 2: Generate Images (Target: $0.00 - FREE with Gemini!)
            images = await self._generate_cheap_images(script, rush_mode)

            # Stage 3: Generate Audio (Target: $0.05)
            audio = await self._generate_cheap_audio(script, rush_mode)

            # Stage 4: Create Video Segments (Target: $0.10)
            video_segments = await self._create_cheap_video(images, audio, rush_mode)

            # Stage 5: Compose Final Video (Target: $0.00 - FFmpeg is free)
            final_video = await self._compose_final_cheap(video_segments, audio)

            # Calculate final metrics
            total_time = time.time() - self.start_time

            result = VideoGenerationResult(
                video_path=final_video["path"],
                total_cost=self.total_cost,
                generation_time=total_time,
                scene_count=len(script["segments"]),
                providers_used=self._get_providers_used(),
                cost_breakdown=self.cost_breakdown,
                success=True
            )

            # Log results
            self._log_results(result, topic)

            return result

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return VideoGenerationResult(
                video_path="",
                total_cost=self.total_cost,
                generation_time=time.time() - self.start_time,
                scene_count=0,
                providers_used={},
                cost_breakdown=self.cost_breakdown,
                success=False,
                error=str(e)
            )

    async def _generate_cheap_script(self, topic: str, rush_mode: bool) -> Dict:
        """Generate script using DeepSeek or other cheap LLM"""

        stage_start = time.time()
        logger.info("Stage 1/5: Generating script with cheap LLM...")

        # Generate base script structure
        script = self.script_gen.generate_full_script(
            topic,
            self.config["target_duration_minutes"]
        )

        # Use cheap LLM to enhance script
        messages = [
            {
                "role": "system",
                "content": "You are a YouTube script writer creating engaging educational content. Use the signature phrase 'Once you know how' strategically."
            },
            {
                "role": "user",
                "content": f"Enhance this script outline for a video about {topic}. Make it engaging and educational:\n{json.dumps(script['segments'][:3])}"
            }
        ]

        # Route to cheapest LLM
        response = await self.api_router.route_request(
            TaskType.SCRIPT_GENERATION,
            {"messages": messages, "temperature": 0.7},
            quality_override=0.7 if not rush_mode else 0.6,
            prefer_free=True
        )

        # Track costs
        script_cost = response["cost"]
        self.cost_breakdown["script"] = script_cost
        self.total_cost += script_cost

        self.stage_times["script"] = time.time() - stage_start
        logger.info(f"Script generated: ${script_cost:.4f} using {response['provider']}")

        return script

    async def _generate_cheap_images(self, script: Dict, rush_mode: bool) -> List[Dict]:
        """Generate images using FREE Gemini or cheap fallbacks"""

        stage_start = time.time()
        logger.info("Stage 2/5: Generating images (targeting FREE with Gemini)...")

        images = []
        scenes_to_generate = []

        # Prepare scene data
        for segment in script["segments"]:
            scene_data = {
                "scene_number": segment.scene_number,
                "scene_type": segment.scene_type,
                "pose_description": self._get_pose_from_narration(segment.narration),
                "background": self._get_background_for_scene(segment.scene_type),
                "icons": self._extract_icons(segment.narration)
            }
            scenes_to_generate.append(scene_data)

        # Check asset library first
        for scene in scenes_to_generate:
            library_asset = self.asset_library.get_asset(
                scene["pose_description"][:20],  # Simplified pose key
                scene["background"]
            )

            if library_asset:
                images.append({
                    "path": library_asset,
                    "scene_number": scene["scene_number"],
                    "cost": 0.0,
                    "provider": "asset_library"
                })
            else:
                # Generate with Gemini (FREE!)
                result = await self.gemini_gen.generate_stick_figure(
                    scene["scene_type"],
                    scene["pose_description"],
                    scene["background"],
                    scene.get("icons"),
                    use_cache=True
                )

                images.append({
                    "path": result["path"],
                    "scene_number": scene["scene_number"],
                    "cost": result["cost"],
                    "provider": result["provider"]
                })

                self.total_cost += result["cost"]
                self.cost_breakdown["images"] += result["cost"]

        self.stage_times["images"] = time.time() - stage_start

        # Get quota status
        quota_status = self.gemini_gen.get_quota_status()
        logger.info(f"Images generated: ${self.cost_breakdown['images']:.4f}")
        logger.info(f"Gemini quota: {quota_status['used']}/{quota_status['daily_limit']} used")

        return images

    async def _generate_cheap_audio(self, script: Dict, rush_mode: bool) -> List[Dict]:
        """Generate audio using cheap/free TTS"""

        stage_start = time.time()
        logger.info("Stage 3/5: Generating audio narration...")

        audio_tracks = []
        total_chars = 0

        for segment in script["segments"]:
            # Optimize narration
            optimized_text = self.narration_opt.optimize_for_tts(segment.narration)
            total_chars += len(optimized_text)

        # Decide which TTS to use based on length
        if total_chars < 10000:
            # Use ElevenLabs free tier
            provider = "elevenlabs_free"
            cost_per_char = 0.0
        else:
            # Use Coqui or Google TTS
            provider = "coqui_xtts"
            cost_per_char = 0.00005  # $0.05 per 1000 chars

        # Generate audio for each segment
        for segment in script["segments"]:
            optimized_text = self.narration_opt.optimize_for_tts(segment.narration)

            # Route to TTS provider
            audio_result = await self.api_router.route_request(
                TaskType.AUDIO_GENERATION,
                {"text": optimized_text, "voice": "default"},
                prefer_free=True
            )

            audio_path = f"temp/audio/narration_{segment.scene_number:03d}.mp3"

            audio_tracks.append({
                "path": audio_path,
                "scene_number": segment.scene_number,
                "duration": self.narration_opt.calculate_duration(optimized_text),
                "text": optimized_text,
                "cost": audio_result["cost"],
                "provider": audio_result["provider"]
            })

            self.total_cost += audio_result["cost"]
            self.cost_breakdown["audio"] += audio_result["cost"]

        self.stage_times["audio"] = time.time() - stage_start
        logger.info(f"Audio generated: ${self.cost_breakdown['audio']:.4f} using {provider}")

        return audio_tracks

    async def _create_cheap_video(
        self,
        images: List[Dict],
        audio: List[Dict],
        rush_mode: bool
    ) -> List[Dict]:
        """Create video segments using mostly static images with transitions"""

        stage_start = time.time()
        logger.info("Stage 4/5: Creating video segments...")

        video_segments = []

        # Determine which scenes need AI video (only 10%)
        total_scenes = len(images)
        ai_video_count = max(1, int(total_scenes * self.config["ai_video_percentage"]))

        # Select most important scenes for AI video
        important_indices = [0, total_scenes // 2, total_scenes - 1]  # Start, middle, end
        important_indices = important_indices[:ai_video_count]

        for i, (img, aud) in enumerate(zip(images, audio)):
            if i in important_indices and not rush_mode:
                # Generate AI video for important scene
                logger.info(f"Generating AI video for scene {i+1} (important)")

                # Use cheap video generation
                video_result = await self.api_router.route_request(
                    TaskType.VIDEO_GENERATION,
                    {
                        "image_path": img["path"],
                        "duration": aud["duration"],
                        "motion": "subtle"
                    },
                    quality_override=0.6,
                    prefer_free=True
                )

                segment_path = f"temp/video_segments/scene_{i+1:03d}.mp4"
                video_cost = video_result["cost"]

            else:
                # Use Ken Burns effect on static image (essentially free)
                segment_path = await self._create_ken_burns_video(
                    img["path"],
                    aud["duration"],
                    i + 1
                )
                video_cost = 0.001  # Minimal processing cost

            video_segments.append({
                "path": segment_path,
                "scene_number": i + 1,
                "duration": aud["duration"],
                "type": "ai_video" if i in important_indices else "ken_burns",
                "cost": video_cost
            })

            self.total_cost += video_cost
            self.cost_breakdown["video"] += video_cost

        self.stage_times["video"] = time.time() - stage_start
        logger.info(f"Video segments created: ${self.cost_breakdown['video']:.4f}")
        logger.info(f"AI videos: {ai_video_count}, Ken Burns: {total_scenes - ai_video_count}")

        return video_segments

    async def _create_ken_burns_video(
        self,
        image_path: str,
        duration: float,
        scene_number: int
    ) -> str:
        """Create video with Ken Burns effect using FFmpeg (FREE)"""

        output_path = f"temp/video_segments/scene_{scene_number:03d}.mp4"

        # FFmpeg command for Ken Burns effect
        # Slowly zoom and pan across the image
        scale_factor = 1.2
        zoom_speed = 0.02

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-vf", f"scale=1920*{scale_factor}:-1,zoompan=z='min(zoom+{zoom_speed},1.5)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={int(duration*30)}",
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            output_path
        ]

        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error: {e}")
            # Fallback to simple static video
            return await self._create_static_video(image_path, duration, scene_number)

    async def _create_static_video(
        self,
        image_path: str,
        duration: float,
        scene_number: int
    ) -> str:
        """Create simple static video from image"""

        output_path = f"temp/video_segments/scene_{scene_number:03d}.mp4"

        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", image_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-vf", "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            output_path
        ]

        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return output_path

    async def _compose_final_cheap(
        self,
        video_segments: List[Dict],
        audio_tracks: List[Dict]
    ) -> Dict:
        """Compose final video using FFmpeg (FREE)"""

        stage_start = time.time()
        logger.info("Stage 5/5: Composing final video...")

        # Create concat file
        concat_file = "temp/concat.txt"
        with open(concat_file, 'w') as f:
            for segment in video_segments:
                f.write(f"file '{Path(segment['path']).absolute()}'\n")

        # Create audio concat file
        audio_concat = "temp/audio_concat.txt"
        with open(audio_concat, 'w') as f:
            for track in audio_tracks:
                f.write(f"file '{Path(track['path']).absolute()}'\n")

        # Merge audio tracks
        merged_audio = "temp/merged_audio.mp3"
        ffmpeg_audio = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", audio_concat,
            "-c", "copy",
            merged_audio
        ]
        subprocess.run(ffmpeg_audio, check=True, capture_output=True)

        # Create final video with audio
        timestamp = int(time.time())
        output_path = f"output/videos/video_{timestamp}.mp4"

        ffmpeg_final = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-i", merged_audio,
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            output_path
        ]

        subprocess.run(ffmpeg_final, check=True, capture_output=True)

        self.stage_times["composition"] = time.time() - stage_start
        logger.info(f"Final video composed: {output_path}")

        return {
            "path": output_path,
            "duration": sum(s["duration"] for s in video_segments),
            "format": "mp4",
            "resolution": self.config["output_resolution"]
        }

    def _get_pose_from_narration(self, narration: str) -> str:
        """Extract pose description from narration content"""

        narration_lower = narration.lower()

        if "explain" in narration_lower:
            return "confident standing pose with one arm extended palm-up in explaining gesture"
        elif "think" in narration_lower or "ponder" in narration_lower:
            return "contemplative pose with hand on chin"
        elif "success" in narration_lower or "achieve" in narration_lower:
            return "triumphant pose with arms raised"
        elif "show" in narration_lower or "point" in narration_lower:
            return "standing in profile pointing to the right"
        else:
            return "standing in neutral professional pose"

    def _get_background_for_scene(self, scene_type: str) -> str:
        """Get appropriate background for scene type"""

        backgrounds = {
            "hook": "blue-to-purple",
            "problem_setup": "violet-to-blue",
            "explanation": "teal-to-navy",
            "example": "coral-to-amber",
            "success": "golden-to-orange",
            "conclusion": "green-to-teal"
        }
        return backgrounds.get(scene_type, "blue-to-purple")

    def _extract_icons(self, narration: str) -> List[str]:
        """Extract relevant icons from narration"""

        icons = []
        narration_lower = narration.lower()

        icon_map = {
            "money": "simple dollar sign icon",
            "dollar": "simple dollar sign icon",
            "idea": "simple lightbulb icon",
            "success": "simple checkmark icon",
            "data": "simple chart icon",
            "time": "simple clock icon"
        }

        for keyword, icon in icon_map.items():
            if keyword in narration_lower and len(icons) < 2:
                icons.append(icon)

        return icons

    def _get_providers_used(self) -> Dict[str, int]:
        """Get count of each provider used"""

        # This would aggregate from api_router
        return {
            "gemini": self.gemini_gen.quota.current_usage,
            "deepseek": 1,
            "coqui": 1,
            "ffmpeg": 1
        }

    def _log_results(self, result: VideoGenerationResult, topic: str):
        """Log detailed results and cost breakdown"""

        log_data = asdict(result)
        log_data["topic"] = topic
        log_data["timestamp"] = time.time()
        log_data["stage_times"] = self.stage_times

        # Save to log file
        log_file = f"logs/cost_reports/generation_{int(time.time())}.json"
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

        # Print summary
        print("\n" + "="*60)
        print("ULTRA-LOW-COST VIDEO GENERATION COMPLETE!")
        print("="*60)
        print(f"Video: {result.video_path}")
        print(f"Total Cost: ${result.total_cost:.4f}")
        print(f"Generation Time: {result.generation_time:.1f}s")
        print(f"\nCost Breakdown:")
        for category, cost in result.cost_breakdown.items():
            print(f"  {category}: ${cost:.4f}")
        print(f"\nProviders Used:")
        for provider, count in result.providers_used.items():
            print(f"  {provider}: {count}x")
        print(f"\nCost Savings: ${10 - result.total_cost:.2f} (vs traditional)")
        print("="*60)


async def main():
    """Test the ultra-low-cost pipeline"""

    # Initialize pipeline
    pipeline = UltraLowCostVideoPipeline()

    # Test topics
    topics = [
        "How AI is Secretly Running the Stock Market",
        "The Hidden Psychology Behind TikTok's Algorithm",
        "Why Billionaires Think Differently About Time"
    ]

    # Generate video
    result = await pipeline.generate_video(
        topic=topics[0],
        rush_mode=False,
        max_budget=0.50
    )

    if result.success:
        print(f"\n✅ Success! Video created for ${result.total_cost:.4f}")
        print(f"🎬 Watch at: {result.video_path}")
    else:
        print(f"\n❌ Failed: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())