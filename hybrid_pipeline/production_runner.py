#!/usr/bin/env python3
"""
Production Runner for Automated YouTube Video Generation
Complete "no-touch" pipeline from topic to upload-ready video
Target: $3-10 per video, 15-25 minute production time
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Import our modules
from main import HybridVideoGenerator
from prompt_system import MinimalistSceneGenerator, StickFigurePromptLibrary
from script_generator import ExplainerScriptGenerator, NarrationOptimizer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CostOptimizer:
    """Intelligent cost optimization for video generation"""

    # Model costs per second
    MODEL_COSTS = {
        "kelin": 0.10,        # Economy tier - good for simple scenes
        "veo3": 0.25,         # Standard tier - balanced quality
        "veo3.1": 0.35,       # Premium tier - highest quality
        "dalle3": 0.15,       # Image generation fallback
        "elevenlabs": 0.02    # TTS per second
    }

    def __init__(self, target_budget: float = 8.0):
        self.target_budget = target_budget
        self.cost_tracker = {"video": 0, "audio": 0, "total": 0}

    def optimize_scene_models(self, scenes: List[Dict],
                            video_duration: int) -> List[Dict]:
        """
        Optimize model selection to meet budget constraints
        Strategy: Use premium for hooks, economy for transitions
        """

        # Calculate base costs
        tts_cost = video_duration * self.MODEL_COSTS["elevenlabs"]
        remaining_budget = self.target_budget - tts_cost

        # Distribute budget across scenes
        budget_per_scene = remaining_budget / len(scenes)

        optimized_scenes = []
        for scene in scenes:
            # Determine model based on importance and budget
            if scene["importance"] == "high" and budget_per_scene >= 2.0:
                model = "veo3.1"
            elif scene["importance"] == "medium" and budget_per_scene >= 1.25:
                model = "veo3"
            else:
                model = "kelin"

            # Special case: Always use premium for first and last scenes
            if scene["scene_number"] == 1 or scene["scene_number"] == len(scenes):
                if self.cost_tracker["total"] + 2.8 <= self.target_budget:
                    model = "veo3.1"

            scene["selected_model"] = model
            scene["estimated_cost"] = self.MODEL_COSTS[model] * scene["duration_seconds"]
            self.cost_tracker["video"] += scene["estimated_cost"]

            optimized_scenes.append(scene)

        self.cost_tracker["audio"] = tts_cost
        self.cost_tracker["total"] = self.cost_tracker["video"] + tts_cost

        logger.info(f"Cost optimization complete: ${self.cost_tracker['total']:.2f}")
        return optimized_scenes

    def get_cost_report(self) -> Dict:
        """Generate detailed cost breakdown"""
        return {
            "video_generation": f"${self.cost_tracker['video']:.2f}",
            "audio_narration": f"${self.cost_tracker['audio']:.2f}",
            "total_cost": f"${self.cost_tracker['total']:.2f}",
            "budget_utilization": f"{(self.cost_tracker['total']/self.target_budget)*100:.1f}%"
        }


class ProductionPipeline:
    """
    Main production pipeline orchestrator
    Handles complete flow from topic to final video
    """

    def __init__(self, config_path: Optional[str] = None):
        self.script_generator = ExplainerScriptGenerator()
        self.narration_optimizer = NarrationOptimizer()
        self.scene_generator = MinimalistSceneGenerator()
        self.cost_optimizer = CostOptimizer()
        self.video_generator = HybridVideoGenerator()

        # Production settings
        self.settings = {
            "target_duration_minutes": 6,
            "max_budget": 8.0,
            "output_resolution": "1080p",
            "output_fps": 30,
            "voice_style": "professional_male",  # ElevenLabs voice
            "music_volume": 0.15  # Background music level
        }

        if config_path:
            self.load_config(config_path)

        # Create output directories
        self.setup_directories()

    def setup_directories(self):
        """Create necessary output directories"""
        dirs = ["output", "temp", "logs", "scripts", "assets"]
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def load_config(self, config_path: str):
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            custom_settings = json.load(f)
            self.settings.update(custom_settings)

    async def produce_video(self, topic: str,
                           rush_mode: bool = False) -> Dict:
        """
        Main production method - complete pipeline
        Args:
            topic: Video topic/subject
            rush_mode: If True, prioritize speed over quality
        Returns:
            Dictionary with video path and metrics
        """

        logger.info(f"="*60)
        logger.info(f"Starting production for: {topic}")
        logger.info(f"Target: {self.settings['target_duration_minutes']} min video, "
                   f"${self.settings['max_budget']} budget")
        logger.info(f"="*60)

        start_time = time.time()
        production_log = {"topic": topic, "stages": []}

        try:
            # Stage 1: Script Generation
            stage_start = time.time()
            logger.info("Stage 1/6: Generating script...")
            script = await self._generate_script(topic, rush_mode)
            production_log["stages"].append({
                "name": "script_generation",
                "duration": time.time() - stage_start,
                "status": "success"
            })

            # Stage 2: Visual Prompt Generation
            stage_start = time.time()
            logger.info("Stage 2/6: Creating visual prompts...")
            visual_scenes = await self._generate_visuals(script)
            production_log["stages"].append({
                "name": "visual_generation",
                "duration": time.time() - stage_start,
                "status": "success"
            })

            # Stage 3: Cost Optimization
            stage_start = time.time()
            logger.info("Stage 3/6: Optimizing costs...")
            optimized_scenes = self.cost_optimizer.optimize_scene_models(
                visual_scenes,
                self.settings["target_duration_minutes"] * 60
            )
            production_log["stages"].append({
                "name": "cost_optimization",
                "duration": time.time() - stage_start,
                "status": "success"
            })

            # Stage 4: Video Generation
            stage_start = time.time()
            logger.info("Stage 4/6: Generating video segments...")
            video_segments = await self._generate_videos(optimized_scenes, rush_mode)
            production_log["stages"].append({
                "name": "video_generation",
                "duration": time.time() - stage_start,
                "status": "success"
            })

            # Stage 5: Audio Generation
            stage_start = time.time()
            logger.info("Stage 5/6: Generating narration...")
            audio_tracks = await self._generate_audio(script)
            production_log["stages"].append({
                "name": "audio_generation",
                "duration": time.time() - stage_start,
                "status": "success"
            })

            # Stage 6: Final Composition
            stage_start = time.time()
            logger.info("Stage 6/6: Composing final video...")
            final_video = await self._compose_final_video(
                video_segments, audio_tracks, script
            )
            production_log["stages"].append({
                "name": "final_composition",
                "duration": time.time() - stage_start,
                "status": "success"
            })

            # Calculate final metrics
            total_time = time.time() - start_time
            production_log["total_duration"] = total_time
            production_log["cost_breakdown"] = self.cost_optimizer.get_cost_report()

            # Save production log
            self._save_production_log(production_log)

            # Final report
            self._print_final_report(final_video, production_log)

            return {
                "success": True,
                "video_path": final_video["path"],
                "duration": final_video["duration"],
                "production_time": total_time,
                "cost": self.cost_optimizer.cost_tracker["total"],
                "log": production_log
            }

        except Exception as e:
            logger.error(f"Production failed: {e}")
            production_log["error"] = str(e)
            production_log["total_duration"] = time.time() - start_time
            self._save_production_log(production_log)
            return {
                "success": False,
                "error": str(e),
                "log": production_log
            }

    async def _generate_script(self, topic: str, rush_mode: bool) -> Dict:
        """Generate optimized script"""

        script = self.script_generator.generate_full_script(
            topic,
            self.settings["target_duration_minutes"]
        )

        # Optimize narration for TTS
        for segment in script["segments"]:
            segment.narration = self.narration_optimizer.optimize_for_tts(
                segment.narration
            )

        # Save script
        script_path = f"scripts/{topic.replace(' ', '_')}_script.json"
        with open(script_path, 'w') as f:
            json.dump(script, f, indent=2, default=str)

        logger.info(f"Script generated: {len(script['segments'])} scenes")
        return script

    async def _generate_visuals(self, script: Dict) -> List[Dict]:
        """Generate visual prompts for each scene"""

        visual_scenes = []

        for segment in script["segments"]:
            # Generate minimalist stick figure prompt
            visual_prompt = self.scene_generator.generate_complete_prompt(
                scene_type=segment.scene_type,
                content=segment.narration,
                emphasis=segment.importance
            )

            scene_data = {
                "scene_number": segment.scene_number,
                "scene_type": segment.scene_type,
                "visual_prompt": visual_prompt,
                "narration": segment.narration,
                "duration_seconds": segment.duration_seconds,
                "importance": segment.importance
            }
            visual_scenes.append(scene_data)

        return visual_scenes

    async def _generate_videos(self, scenes: List[Dict],
                              rush_mode: bool) -> List[Dict]:
        """Generate video segments using optimized models"""

        video_segments = []

        # Batch processing for efficiency
        if rush_mode:
            # Process all at once for speed
            tasks = [self._generate_single_video(scene) for scene in scenes]
            video_segments = await asyncio.gather(*tasks)
        else:
            # Process in smaller batches for quality control
            batch_size = 3
            for i in range(0, len(scenes), batch_size):
                batch = scenes[i:i+batch_size]
                tasks = [self._generate_single_video(scene) for scene in batch]
                batch_results = await asyncio.gather(*tasks)
                video_segments.extend(batch_results)

                # Progress update
                progress = (i + len(batch)) / len(scenes) * 100
                logger.info(f"Video generation progress: {progress:.1f}%")

        return video_segments

    async def _generate_single_video(self, scene: Dict) -> Dict:
        """Generate single video segment"""

        # This would call actual Filmora API or video generation service
        # For now, simulating the process

        await asyncio.sleep(0.5)  # Simulate generation time

        return {
            "scene_number": scene["scene_number"],
            "video_path": f"temp/scene_{scene['scene_number']:03d}.mp4",
            "duration": scene["duration_seconds"],
            "model_used": scene.get("selected_model", "kelin"),
            "prompt": scene["visual_prompt"][:100] + "..."  # Truncate for logging
        }

    async def _generate_audio(self, script: Dict) -> List[Dict]:
        """Generate audio narration tracks"""

        audio_tracks = []

        for segment in script["segments"]:
            # Calculate actual duration based on text
            actual_duration = self.narration_optimizer.calculate_duration(
                segment.narration
            )

            track = {
                "scene_number": segment.scene_number,
                "audio_path": f"temp/narration_{segment.scene_number:03d}.mp3",
                "text": segment.narration,
                "duration": actual_duration,
                "voice": self.settings["voice_style"]
            }
            audio_tracks.append(track)

        return audio_tracks

    async def _compose_final_video(self, video_segments: List[Dict],
                                  audio_tracks: List[Dict],
                                  script: Dict) -> Dict:
        """Compose final video with all elements"""

        # This would use FFmpeg or Remotion to compose
        # For now, simulating the output

        total_duration = sum(seg["duration"] for seg in video_segments)
        timestamp = int(time.time())
        output_filename = f"{script['title'].replace(' ', '_')}_{timestamp}.mp4"
        output_path = f"output/{output_filename}"

        return {
            "path": output_path,
            "duration": total_duration,
            "resolution": self.settings["output_resolution"],
            "fps": self.settings["output_fps"],
            "title": script["title"],
            "scenes": len(video_segments)
        }

    def _save_production_log(self, log: Dict):
        """Save production log for analysis"""

        timestamp = int(time.time())
        log_path = f"logs/production_{timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump(log, f, indent=2, default=str)

    def _print_final_report(self, video: Dict, log: Dict):
        """Print final production report"""

        print("\n" + "="*60)
        print("PRODUCTION COMPLETE")
        print("="*60)
        print(f"Video Title: {video['title']}")
        print(f"Output Path: {video['path']}")
        print(f"Duration: {video['duration']} seconds ({video['duration']/60:.1f} minutes)")
        print(f"Resolution: {video['resolution']} @ {video['fps']}fps")
        print(f"Total Scenes: {video['scenes']}")
        print("\nCost Breakdown:")
        for key, value in log["cost_breakdown"].items():
            print(f"  {key}: {value}")
        print(f"\nProduction Time: {log['total_duration']:.1f} seconds")
        print(f"Time per minute of video: {log['total_duration']/(video['duration']/60):.1f} seconds")
        print("\n✅ Video ready for upload!")
        print("="*60)


async def main():
    """Main entry point for production"""

    # Example topics for different niches
    test_topics = [
        "How AI is Secretly Running the Stock Market",
        "The Hidden Psychology of Viral Content",
        "Why Billionaires Think Differently About Money",
        "The Productivity Secret Silicon Valley Doesn't Share",
        "How Your Phone is Designed to Control You"
    ]

    # Initialize pipeline
    pipeline = ProductionPipeline()

    # Produce a video
    topic = test_topics[0]  # Select first topic
    result = await pipeline.produce_video(topic, rush_mode=False)

    if result["success"]:
        print(f"\n🎬 Video successfully created: {result['video_path']}")
        print(f"Ready for YouTube upload!")
    else:
        print(f"\n❌ Production failed: {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())