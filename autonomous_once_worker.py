"""
Autonomous Worker System for Once Project
Continuously analyzes and improves the Once video generation system

Features:
- 24/7 operation
- Quality analysis
- Code improvements
- Self-healing
- Learning from performance data
"""

import asyncio
import structlog
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json

# Add Once project to path
ONCE_PROJECT_PATH = Path("/mnt/e/projects/once")
sys.path.insert(0, str(ONCE_PROJECT_PATH))

logger = structlog.get_logger()


class TaskPriority(Enum):
    """Task priority levels"""
    CRITICAL = 1    # Broken functionality, immediate fix required
    HIGH = 2        # Quality issues affecting production
    MEDIUM = 3      # Improvements and optimizations
    LOW = 4         # Research, exploration, documentation


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkerTask:
    """Represents a task to be executed by workers"""
    task_id: str
    task_type: str
    priority: TaskPriority
    description: str
    execute_fn: callable
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def __lt__(self, other):
        """Enable priority queue sorting"""
        return self.priority.value < other.priority.value


class OnceProjectAnalyzer:
    """Analyzes Once project for quality and improvement opportunities"""

    def __init__(self, project_path: Path):
        self.project_path = project_path

    async def analyze_recent_scripts(self) -> Dict:
        """Analyze recently generated scripts for quality issues"""
        logger.info("analyzing_recent_scripts", path=self.project_path)

        issues_found = []
        cliches_detected = []

        # AI cliché patterns (expanded from basic list)
        CLICHE_PATTERNS = {
            "vague_promises": [
                "unlock", "secret", "game-changer", "revolutionize",
                "transform", "breakthrough"
            ],
            "filler_phrases": [
                "dive into", "delve into", "at the end of the day",
                "to be honest", "essentially", "basically"
            ],
            "ai_signatures": [
                "it's important to note", "it's worth noting",
                "one could argue", "in today's world",
                "in today's landscape", "in today's society"
            ],
            "overused_transitions": [
                "let's dive deeper", "moving forward",
                "with that being said", "having said that"
            ]
        }

        # Check output directory for recent videos
        output_dir = self.project_path / "output"
        if output_dir.exists():
            # Find recent video directories
            recent_dirs = sorted(
                [d for d in output_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:10]  # Last 10 videos

            for video_dir in recent_dirs:
                script_file = video_dir / "script.json"
                if script_file.exists():
                    try:
                        with open(script_file, 'r', encoding='utf-8') as f:
                            script = json.load(f)

                        # Analyze scenes for clichés
                        for scene in script.get("scenes", []):
                            narration = scene.get("narration", "").lower()

                            # Check for clichés
                            for category, patterns in CLICHE_PATTERNS.items():
                                for pattern in patterns:
                                    if pattern in narration:
                                        cliches_detected.append({
                                            "video": video_dir.name,
                                            "scene": scene.get("scene_number"),
                                            "category": category,
                                            "pattern": pattern,
                                            "narration_preview": narration[:100]
                                        })

                            # Check scene duration appropriateness
                            duration = scene.get("duration", 4.0)
                            word_count = len(narration.split())
                            words_per_second = word_count / duration if duration > 0 else 0

                            if words_per_second > 4:  # Too fast
                                issues_found.append({
                                    "video": video_dir.name,
                                    "scene": scene.get("scene_number"),
                                    "issue": "pacing_too_fast",
                                    "words_per_second": words_per_second,
                                    "recommendation": "Increase duration or simplify narration"
                                })
                            elif words_per_second < 1.5:  # Too slow
                                issues_found.append({
                                    "video": video_dir.name,
                                    "scene": scene.get("scene_number"),
                                    "issue": "pacing_too_slow",
                                    "words_per_second": words_per_second,
                                    "recommendation": "Decrease duration or add more content"
                                })

                    except json.JSONDecodeError as e:
                        logger.error("script_json_invalid", file=str(script_file), error=str(e))
                    except IOError as e:
                        logger.error("script_read_failed", file=str(script_file), error=str(e))

        return {
            "videos_analyzed": len(recent_dirs) if output_dir.exists() else 0,
            "cliches_detected": len(cliches_detected),
            "cliche_details": cliches_detected[:20],  # Top 20
            "pacing_issues": len(issues_found),
            "issue_details": issues_found[:20],
            "recommendations": self._generate_recommendations(cliches_detected, issues_found)
        }

    def _generate_recommendations(self, cliches: List[Dict], issues: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []

        if cliches:
            most_common_cliche = max(
                set(c["pattern"] for c in cliches),
                key=lambda x: sum(1 for c in cliches if c["pattern"] == x)
            )
            recommendations.append(
                f"Update prompt to explicitly avoid '{most_common_cliche}' "
                f"(detected {sum(1 for c in cliches if c['pattern'] == most_common_cliche)} times)"
            )

        if issues:
            pacing_issues = [i for i in issues if "pacing" in i.get("issue", "")]
            if pacing_issues:
                recommendations.append(
                    f"Review scene duration calculation - {len(pacing_issues)} scenes have pacing issues"
                )

        if not recommendations:
            recommendations.append("No major issues detected - system performing well!")

        return recommendations

    async def analyze_image_quality(self) -> Dict:
        """Analyze generated images for quality and coherence"""
        logger.info("analyzing_image_quality")

        issues = []
        output_dir = self.project_path / "output"

        if output_dir.exists():
            recent_dirs = sorted(
                [d for d in output_dir.iterdir() if d.is_dir()],
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )[:10]

            for video_dir in recent_dirs:
                images_dir = video_dir / "images"
                if images_dir.exists():
                    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))

                    for img_file in image_files:
                        try:
                            # Check file size (potential quality indicator)
                            size_mb = img_file.stat().st_size / (1024 * 1024)

                            if size_mb < 0.1:  # Less than 100KB might be low quality
                                issues.append({
                                    "video": video_dir.name,
                                    "image": img_file.name,
                                    "issue": "low_file_size",
                                    "size_mb": round(size_mb, 2),
                                    "recommendation": "Investigate image generation quality"
                                })
                            elif size_mb > 10:  # Over 10MB might be inefficient
                                issues.append({
                                    "video": video_dir.name,
                                    "image": img_file.name,
                                    "issue": "large_file_size",
                                    "size_mb": round(size_mb, 2),
                                    "recommendation": "Consider compression or lower quality setting"
                                })

                        except (OSError, IOError) as e:
                            logger.error("image_analysis_failed", file=str(img_file), error=str(e))

        return {
            "videos_analyzed": len(recent_dirs) if output_dir.exists() else 0,
            "issues_found": len(issues),
            "issue_details": issues[:20],
            "summary": f"Analyzed images from {len(recent_dirs) if output_dir.exists() else 0} recent videos"
        }

    async def find_code_improvement_opportunities(self) -> Dict:
        """Identify opportunities to improve Once codebase"""
        logger.info("analyzing_code_for_improvements")

        opportunities = []
        src_dir = self.project_path / "src"

        if src_dir.exists():
            # Find Python files
            py_files = list(src_dir.rglob("*.py"))

            for py_file in py_files:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for potential improvements
                    lines = content.split('\n')

                    # Find TODO comments
                    todos = [
                        (i + 1, line.strip())
                        for i, line in enumerate(lines)
                        if 'TODO' in line or 'FIXME' in line or 'HACK' in line
                    ]

                    if todos:
                        for line_num, todo_text in todos:
                            opportunities.append({
                                "file": str(py_file.relative_to(self.project_path)),
                                "line": line_num,
                                "type": "todo_comment",
                                "content": todo_text,
                                "priority": "medium"
                            })

                    # Find hardcoded values that should be configurable
                    if 'magic number' in content.lower():
                        opportunities.append({
                            "file": str(py_file.relative_to(self.project_path)),
                            "type": "magic_numbers",
                            "recommendation": "Extract hardcoded values to configuration",
                            "priority": "low"
                        })

                    # Check for missing docstrings
                    if 'def ' in content and '"""' not in content[:500]:
                        # Heuristic: if file has functions but no docstrings in first 500 chars
                        opportunities.append({
                            "file": str(py_file.relative_to(self.project_path)),
                            "type": "missing_documentation",
                            "recommendation": "Add module and function docstrings",
                            "priority": "low"
                        })

                except (IOError, UnicodeDecodeError) as e:
                    logger.error("code_analysis_failed", file=str(py_file), error=str(e))

        return {
            "files_analyzed": len(py_files) if src_dir.exists() else 0,
            "opportunities_found": len(opportunities),
            "details": opportunities[:30],  # Top 30
            "summary": f"Found {len(opportunities)} improvement opportunities"
        }


class AutonomousWorkerSystem:
    """
    24/7 Autonomous worker system for continuous improvement
    """

    def __init__(
        self,
        num_workers: int = 4,
        once_project_path: Path = ONCE_PROJECT_PATH
    ):
        self.num_workers = num_workers
        self.once_path = once_project_path
        self.task_queue = asyncio.PriorityQueue()
        self.running = False
        self.workers = []
        self.task_history = []
        self.analyzer = OnceProjectAnalyzer(once_project_path)

        # Results directory
        self.results_dir = Path("/mnt/e/projects/code/autonomous_worker_results")
        self.results_dir.mkdir(exist_ok=True)

        logger.info(
            "worker_system_initialized",
            workers=num_workers,
            once_path=str(once_project_path)
        )

    async def start(self):
        """Start the autonomous worker system"""
        logger.info("🚀 Starting autonomous worker system", workers=self.num_workers)

        self.running = True

        # Start worker coroutines
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.num_workers)
        ]

        # Start task generator (runs continuously)
        asyncio.create_task(self._task_generator())

        # Start progress monitor
        asyncio.create_task(self._monitor_progress())

        # Generate initial tasks
        await self._generate_initial_tasks()

        logger.info("✅ Worker system started successfully")

        # Keep running
        try:
            await asyncio.gather(*self.workers)
        except asyncio.CancelledError:
            logger.info("Worker system cancelled")

    async def _worker(self, worker_id: int):
        """Individual worker coroutine"""
        logger.info("👷 Worker started", worker_id=worker_id)

        while self.running:
            try:
                # Get next task from priority queue (blocks if empty)
                _, task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=60.0  # 1 minute timeout
                )

                logger.info(
                    "🔨 Worker executing task",
                    worker_id=worker_id,
                    task_id=task.task_id,
                    task_type=task.task_type,
                    priority=task.priority.name
                )

                # Execute task
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

                try:
                    result = await task.execute_fn(task.metadata)
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.now()

                    duration = (task.completed_at - task.started_at).total_seconds()

                    logger.info(
                        "✅ Task completed",
                        worker_id=worker_id,
                        task_id=task.task_id,
                        duration=f"{duration:.2f}s"
                    )

                    # Save result
                    self._save_task_result(task)

                except Exception as e:
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()

                    logger.error(
                        "❌ Task failed",
                        worker_id=worker_id,
                        task_id=task.task_id,
                        error=str(e)
                    )

                # Add to history
                self.task_history.append(task)

                # Mark task as done in queue
                self.task_queue.task_done()

            except asyncio.TimeoutError:
                # No tasks available, worker idles
                await asyncio.sleep(5)
                continue

            except Exception as e:
                logger.error("worker_error", worker_id=worker_id, error=str(e))
                await asyncio.sleep(5)

    async def _task_generator(self):
        """Continuously generate new tasks"""
        logger.info("📋 Task generator started")

        iteration = 0

        while self.running:
            try:
                iteration += 1
                logger.info("🔄 Generating tasks", iteration=iteration)

                # Quality analysis tasks (every 30 minutes)
                if iteration % 6 == 1:  # Every 30 min (if 5 min cycle)
                    await self._generate_quality_analysis_tasks()

                # Code improvement tasks (every hour)
                if iteration % 12 == 1:
                    await self._generate_code_improvement_tasks()

                # Research tasks (every 2 hours)
                if iteration % 24 == 1:
                    await self._generate_research_tasks()

                # Daily deep analysis (every 24 hours)
                if iteration % 288 == 1:  # Every 24 hours
                    await self._generate_deep_analysis_tasks()

            except Exception as e:
                logger.error("task_generation_error", error=str(e))

            # Wait before next cycle (5 minutes)
            await asyncio.sleep(300)

    async def _generate_initial_tasks(self):
        """Generate initial set of tasks on startup"""
        logger.info("📝 Generating initial tasks")

        # Immediate quality check
        await self._add_task(
            task_type="quality_check",
            priority=TaskPriority.HIGH,
            description="Initial quality check of recent Once videos",
            execute_fn=self._task_analyze_scripts
        )

        # Code analysis
        await self._add_task(
            task_type="code_analysis",
            priority=TaskPriority.MEDIUM,
            description="Analyze Once codebase for improvements",
            execute_fn=self._task_analyze_code
        )

        # Image quality check
        await self._add_task(
            task_type="image_quality",
            priority=TaskPriority.MEDIUM,
            description="Analyze generated images for quality issues",
            execute_fn=self._task_analyze_images
        )

    async def _generate_quality_analysis_tasks(self):
        """Generate quality analysis tasks"""
        await self._add_task(
            task_type="quality_check",
            priority=TaskPriority.HIGH,
            description="Analyze recent videos for quality issues",
            execute_fn=self._task_analyze_scripts
        )

    async def _generate_code_improvement_tasks(self):
        """Generate code improvement tasks"""
        await self._add_task(
            task_type="code_analysis",
            priority=TaskPriority.MEDIUM,
            description="Find code improvement opportunities",
            execute_fn=self._task_analyze_code
        )

    async def _generate_research_tasks(self):
        """Generate research and exploration tasks"""
        await self._add_task(
            task_type="research",
            priority=TaskPriority.LOW,
            description="Research best practices for educational videos",
            execute_fn=self._task_research_best_practices
        )

    async def _generate_deep_analysis_tasks(self):
        """Generate comprehensive deep analysis tasks"""
        await self._add_task(
            task_type="deep_analysis",
            priority=TaskPriority.MEDIUM,
            description="Comprehensive system analysis and reporting",
            execute_fn=self._task_comprehensive_analysis
        )

    async def _add_task(
        self,
        task_type: str,
        priority: TaskPriority,
        description: str,
        execute_fn: callable,
        metadata: Dict = None
    ):
        """Add task to queue"""
        task = WorkerTask(
            task_id=f"{task_type}_{int(datetime.now().timestamp())}",
            task_type=task_type,
            priority=priority,
            description=description,
            execute_fn=execute_fn,
            metadata=metadata or {}
        )

        await self.task_queue.put((priority.value, task))
        logger.info("➕ Task added", task_id=task.task_id, priority=priority.name)

    # Task execution functions

    async def _task_analyze_scripts(self, metadata: Dict) -> Dict:
        """Analyze scripts for quality"""
        return await self.analyzer.analyze_recent_scripts()

    async def _task_analyze_images(self, metadata: Dict) -> Dict:
        """Analyze images for quality"""
        return await self.analyzer.analyze_image_quality()

    async def _task_analyze_code(self, metadata: Dict) -> Dict:
        """Analyze code for improvements"""
        return await self.analyzer.find_code_improvement_opportunities()

    async def _task_research_best_practices(self, metadata: Dict) -> Dict:
        """Research educational video best practices"""
        logger.info("researching_best_practices")

        # Placeholder - would integrate with web search, YouTube API, etc.
        return {
            "sources_analyzed": 0,
            "best_practices": [
                "Hook viewers in first 5 seconds",
                "Use visual metaphors for complex concepts",
                "Maintain consistent pacing",
                "Include call-to-action at end"
            ],
            "recommendations": [
                "Add engagement metrics tracking",
                "Implement A/B testing for hooks"
            ]
        }

    async def _task_comprehensive_analysis(self, metadata: Dict) -> Dict:
        """Comprehensive system analysis"""
        logger.info("running_comprehensive_analysis")

        # Run all analyzers
        script_analysis = await self.analyzer.analyze_recent_scripts()
        image_analysis = await self.analyzer.analyze_image_quality()
        code_analysis = await self.analyzer.find_code_improvement_opportunities()

        return {
            "timestamp": datetime.now().isoformat(),
            "script_analysis": script_analysis,
            "image_analysis": image_analysis,
            "code_analysis": code_analysis,
            "overall_health": self._calculate_health_score(
                script_analysis, image_analysis, code_analysis
            )
        }

    def _calculate_health_score(
        self, script_analysis: Dict, image_analysis: Dict, code_analysis: Dict
    ) -> Dict:
        """Calculate overall system health score"""

        # Simple scoring based on issues found
        script_issues = (
            script_analysis.get("cliches_detected", 0) +
            script_analysis.get("pacing_issues", 0)
        )
        image_issues = image_analysis.get("issues_found", 0)
        code_issues = code_analysis.get("opportunities_found", 0)

        total_issues = script_issues + image_issues + code_issues

        # Score out of 100
        health_score = max(0, 100 - (total_issues * 2))

        return {
            "score": health_score,
            "grade": self._score_to_grade(health_score),
            "script_quality": max(0, 100 - (script_issues * 5)),
            "image_quality": max(0, 100 - (image_issues * 5)),
            "code_quality": max(0, 100 - (code_issues * 2))
        }

    def _score_to_grade(self, score: int) -> str:
        """Convert score to letter grade"""
        if score >= 90: return "A"
        if score >= 80: return "B"
        if score >= 70: return "C"
        if score >= 60: return "D"
        return "F"

    def _save_task_result(self, task: WorkerTask):
        """Save task result to file"""
        try:
            result_file = self.results_dir / f"{task.task_id}_result.json"

            result_data = {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "description": task.description,
                "priority": task.priority.name,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "duration_seconds": (
                    (task.completed_at - task.started_at).total_seconds()
                    if task.started_at and task.completed_at else None
                ),
                "result": task.result,
                "error": task.error
            }

            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2)

            logger.info("💾 Task result saved", file=str(result_file))

        except (IOError, OSError, TypeError) as e:
            logger.error("failed_to_save_result", task_id=task.task_id, error=str(e))

    async def _monitor_progress(self):
        """Monitor and report worker progress"""
        logger.info("📊 Progress monitor started")

        while self.running:
            await asyncio.sleep(300)  # Every 5 minutes

            # Calculate statistics
            total_tasks = len(self.task_history)
            completed = sum(1 for t in self.task_history if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in self.task_history if t.status == TaskStatus.FAILED)

            # Recent tasks (last hour)
            one_hour_ago = datetime.now() - timedelta(hours=1)
            recent_tasks = [
                t for t in self.task_history
                if t.completed_at and t.completed_at > one_hour_ago
            ]

            logger.info(
                "📈 Worker statistics",
                total_tasks=total_tasks,
                completed=completed,
                failed=failed,
                queue_size=self.task_queue.qsize(),
                recent_tasks_last_hour=len(recent_tasks)
            )

            # Generate summary report
            if total_tasks > 0 and total_tasks % 10 == 0:
                self._generate_progress_report()

    def _generate_progress_report(self):
        """Generate progress report"""
        report_file = self.results_dir / f"progress_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        completed = [t for t in self.task_history if t.status == TaskStatus.COMPLETED]
        failed = [t for t in self.task_history if t.status == TaskStatus.FAILED]

        # Task type breakdown
        task_type_counts = {}
        for task in self.task_history:
            task_type_counts[task.task_type] = task_type_counts.get(task.task_type, 0) + 1

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(self.task_history),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self.task_history) if self.task_history else 0,
            "task_type_breakdown": task_type_counts,
            "recent_results": [
                {
                    "task_id": t.task_id,
                    "type": t.task_type,
                    "status": t.status.value,
                    "duration": (
                        (t.completed_at - t.started_at).total_seconds()
                        if t.started_at and t.completed_at else None
                    )
                }
                for t in self.task_history[-10:]  # Last 10 tasks
            ]
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info("📄 Progress report generated", file=str(report_file))

    async def stop(self):
        """Stop the worker system"""
        logger.info("🛑 Stopping worker system")
        self.running = False

        # Cancel all workers
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)

        logger.info("✅ Worker system stopped")


async def main():
    """Main entry point"""
    print("="*60)
    print("🤖 Once Project - Autonomous Worker System")
    print("="*60)
    print()
    print("This system will continuously analyze and improve the Once project.")
    print("Workers: 4")
    print("Running: 24/7")
    print("Results: /mnt/e/projects/code/autonomous_worker_results/")
    print()
    print("Press Ctrl+C to stop")
    print("="*60)
    print()

    worker_system = AutonomousWorkerSystem(num_workers=4)

    try:
        await worker_system.start()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping workers...")
        await worker_system.stop()
        print("✅ Workers stopped cleanly")


if __name__ == "__main__":
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer()
        ]
    )

    # Run the system
    asyncio.run(main())
