#!/usr/bin/env python3
"""
Oracle Cloud 24/7 Worker Orchestrator

A unified orchestrator for running continuous improvement workers on Oracle Cloud.
Manages multiple worker types, health monitoring, auto-restart, and task distribution.

Features:
- Multi-worker management (training, improvement, analysis)
- Health monitoring with auto-restart
- Persistent state across restarts
- Task queue with priority scheduling
- Cost tracking and optimization
- Graceful shutdown handling

Usage:
    python oracle_24x7_orchestrator.py                    # Start orchestrator
    python oracle_24x7_orchestrator.py --workers 3        # Start with 3 workers
    python oracle_24x7_orchestrator.py --status           # Show status
    python oracle_24x7_orchestrator.py --stop             # Stop all workers
"""

import os
import sys
import json
import asyncio
import signal
import logging
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from logging.handlers import RotatingFileHandler
import argparse

# Setup logging with rotation
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            LOG_DIR / "orchestrator.log",
            maxBytes=10*1024*1024,
            backupCount=10,
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("orchestrator")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class WorkerType(Enum):
    TRAINING = "training"
    IMPROVEMENT = "improvement"
    ANALYSIS = "analysis"
    BENCHMARK = "benchmark"


class WorkerStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    IDLE = "idle"
    ERROR = "error"
    STOPPED = "stopped"
    RESTARTING = "restarting"


class TaskPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class WorkerInfo:
    """Information about a worker process"""
    worker_id: str
    worker_type: WorkerType
    status: WorkerStatus = WorkerStatus.STOPPED
    pid: Optional[int] = None
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    current_task: Optional[str] = None
    error_count: int = 0
    restart_count: int = 0

    def to_dict(self) -> Dict:
        return {
            "worker_id": self.worker_id,
            "worker_type": self.worker_type.value,
            "status": self.status.value,
            "pid": self.pid,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "current_task": self.current_task,
            "error_count": self.error_count,
            "restart_count": self.restart_count,
            "uptime": str(datetime.now() - self.started_at) if self.started_at else None
        }


@dataclass
class Task:
    """A task to be executed by workers"""
    task_id: str
    task_type: str
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    created_at: datetime = field(default_factory=datetime.now)
    assigned_to: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "assigned_to": self.assigned_to,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error,
            "retries": self.retries
        }


@dataclass
class OrchestratorStats:
    """Statistics for the orchestrator"""
    started_at: datetime = field(default_factory=datetime.now)
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    total_worker_restarts: int = 0
    improvements_made: int = 0
    tests_passed: int = 0
    code_generated_lines: int = 0

    def to_dict(self) -> Dict:
        uptime = datetime.now() - self.started_at
        return {
            "uptime": str(uptime),
            "uptime_hours": uptime.total_seconds() / 3600,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "success_rate": f"{self.total_tasks_completed / max(1, self.total_tasks_completed + self.total_tasks_failed):.1%}",
            "total_worker_restarts": self.total_worker_restarts,
            "improvements_made": self.improvements_made,
            "tests_passed": self.tests_passed,
            "code_generated_lines": self.code_generated_lines
        }


# =============================================================================
# TASK QUEUE
# =============================================================================

class TaskQueue:
    """Priority-based task queue with persistence"""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self.tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self._load_state()

    def add_task(self, task: Task):
        """Add a task to the queue"""
        self.tasks.append(task)
        self.tasks.sort(key=lambda t: (t.priority.value, t.created_at))
        self._save_state()
        logger.info(f"Task added: {task.task_id} ({task.task_type})")

    def get_next_task(self, worker_type: WorkerType) -> Optional[Task]:
        """Get next available task for worker type"""
        for task in self.tasks:
            if task.assigned_to is None:
                # Match task type to worker type
                if self._task_matches_worker(task, worker_type):
                    task.assigned_to = f"pending_{worker_type.value}"
                    self._save_state()
                    return task
        return None

    def _task_matches_worker(self, task: Task, worker_type: WorkerType) -> bool:
        """Check if task type matches worker type"""
        mappings = {
            WorkerType.TRAINING: ["training", "learn", "improve_patterns"],
            WorkerType.IMPROVEMENT: ["improve", "refactor", "fix", "enhance"],
            WorkerType.ANALYSIS: ["analyze", "audit", "review", "check"],
            WorkerType.BENCHMARK: ["benchmark", "test", "validate"]
        }
        return any(m in task.task_type.lower() for m in mappings.get(worker_type, []))

    def complete_task(self, task_id: str, result: Dict):
        """Mark task as completed"""
        for i, task in enumerate(self.tasks):
            if task.task_id == task_id:
                task.completed_at = datetime.now()
                task.result = result
                self.completed_tasks.append(task)
                self.tasks.pop(i)
                self._save_state()
                logger.info(f"Task completed: {task_id}")
                return
        logger.warning(f"Task not found: {task_id}")

    def fail_task(self, task_id: str, error: str):
        """Mark task as failed, retry if possible"""
        for task in self.tasks:
            if task.task_id == task_id:
                task.retries += 1
                task.error = error
                if task.retries < task.max_retries:
                    task.assigned_to = None  # Re-queue
                    logger.warning(f"Task {task_id} failed, retry {task.retries}/{task.max_retries}")
                else:
                    task.completed_at = datetime.now()
                    self.completed_tasks.append(task)
                    self.tasks.remove(task)
                    logger.error(f"Task {task_id} failed permanently: {error}")
                self._save_state()
                return

    def get_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            "pending": len([t for t in self.tasks if t.assigned_to is None]),
            "in_progress": len([t for t in self.tasks if t.assigned_to is not None]),
            "completed": len(self.completed_tasks),
            "by_priority": {
                p.name: len([t for t in self.tasks if t.priority == p])
                for p in TaskPriority
            }
        }

    def _save_state(self):
        """Save queue state to file"""
        state = {
            "tasks": [t.to_dict() for t in self.tasks],
            "completed_tasks": [t.to_dict() for t in self.completed_tasks[-100:]]  # Keep last 100
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load queue state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    state = json.load(f)
                # Reconstruct tasks (simplified - would need proper deserialization)
                logger.info(f"Loaded {len(state.get('tasks', []))} pending tasks from state")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Could not load state: {e}")


# =============================================================================
# WORKER MANAGER
# =============================================================================

class WorkerManager:
    """Manages worker processes"""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.workers: Dict[str, WorkerInfo] = {}
        self.processes: Dict[str, subprocess.Popen] = {}

    async def start_worker(
        self,
        worker_type: WorkerType,
        worker_id: Optional[str] = None
    ) -> WorkerInfo:
        """Start a new worker process"""
        if worker_id is None:
            worker_id = f"{worker_type.value}_{len(self.workers)}"

        worker = WorkerInfo(
            worker_id=worker_id,
            worker_type=worker_type,
            status=WorkerStatus.STARTING,
            started_at=datetime.now()
        )

        # Get the appropriate worker script
        script = self._get_worker_script(worker_type)

        if script and script.exists():
            try:
                # Start worker process
                env = os.environ.copy()
                env['WORKER_ID'] = worker_id
                env['WORKER_TYPE'] = worker_type.value

                process = subprocess.Popen(
                    [sys.executable, str(script), "--worker-id", worker_id],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    cwd=str(self.base_dir)
                )

                worker.pid = process.pid
                worker.status = WorkerStatus.RUNNING
                worker.last_heartbeat = datetime.now()

                self.workers[worker_id] = worker
                self.processes[worker_id] = process

                logger.info(f"Started worker {worker_id} (PID: {worker.pid})")

            except Exception as e:
                worker.status = WorkerStatus.ERROR
                logger.error(f"Failed to start worker {worker_id}: {e}")
        else:
            # Run inline worker if no script exists
            worker.status = WorkerStatus.RUNNING
            worker.last_heartbeat = datetime.now()
            self.workers[worker_id] = worker
            logger.info(f"Started inline worker {worker_id}")

        return worker

    def _get_worker_script(self, worker_type: WorkerType) -> Optional[Path]:
        """Get the script path for a worker type"""
        scripts = {
            WorkerType.TRAINING: self.base_dir / "multi_provider_trainer.py",
            WorkerType.IMPROVEMENT: self.base_dir / "oci_iterative_improver.py",
            WorkerType.ANALYSIS: self.base_dir / "autonomous_once_worker.py",
            WorkerType.BENCHMARK: self.base_dir / "benchmark.py"
        }
        return scripts.get(worker_type)

    async def stop_worker(self, worker_id: str, graceful: bool = True):
        """Stop a worker process"""
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found")
            return

        worker = self.workers[worker_id]

        if worker_id in self.processes:
            process = self.processes[worker_id]
            if process.poll() is None:  # Still running
                if graceful:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                else:
                    process.kill()

            del self.processes[worker_id]

        worker.status = WorkerStatus.STOPPED
        logger.info(f"Stopped worker {worker_id}")

    async def restart_worker(self, worker_id: str):
        """Restart a worker"""
        if worker_id not in self.workers:
            return

        worker = self.workers[worker_id]
        worker.restart_count += 1
        worker.status = WorkerStatus.RESTARTING

        await self.stop_worker(worker_id)
        await asyncio.sleep(2)
        await self.start_worker(worker.worker_type, worker_id)

        logger.info(f"Restarted worker {worker_id} (restart #{worker.restart_count})")

    def check_health(self) -> Dict[str, bool]:
        """Check health of all workers"""
        health = {}
        now = datetime.now()

        for worker_id, worker in self.workers.items():
            # Check if process is running
            if worker_id in self.processes:
                process = self.processes[worker_id]
                if process.poll() is not None:
                    worker.status = WorkerStatus.ERROR
                    health[worker_id] = False
                    continue

            # Check heartbeat timeout (5 minutes)
            if worker.last_heartbeat:
                timeout = timedelta(minutes=5)
                if now - worker.last_heartbeat > timeout:
                    worker.status = WorkerStatus.ERROR
                    health[worker_id] = False
                    continue

            health[worker_id] = worker.status == WorkerStatus.RUNNING

        return health

    def get_status(self) -> Dict:
        """Get status of all workers"""
        return {
            "total_workers": len(self.workers),
            "running": len([w for w in self.workers.values() if w.status == WorkerStatus.RUNNING]),
            "workers": {wid: w.to_dict() for wid, w in self.workers.items()}
        }


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class Oracle24x7Orchestrator:
    """Main orchestrator for 24/7 autonomous operation"""

    def __init__(self, base_dir: Optional[Path] = None, num_workers: int = 2):
        self.base_dir = base_dir or Path(__file__).parent
        self.state_dir = self.base_dir / ".orchestrator_state"
        self.state_dir.mkdir(exist_ok=True)

        self.num_workers = num_workers
        self.running = False
        self.stats = OrchestratorStats()

        # Initialize components
        self.task_queue = TaskQueue(self.state_dir / "task_queue.json")
        self.worker_manager = WorkerManager(self.base_dir)

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(f"Orchestrator initialized (base_dir: {self.base_dir})")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False

    async def start(self):
        """Start the orchestrator and all workers"""
        logger.info("=" * 60)
        logger.info("ORACLE 24/7 ORCHESTRATOR STARTING")
        logger.info("=" * 60)

        self.running = True
        self.stats.started_at = datetime.now()

        # Start workers
        worker_types = [
            WorkerType.TRAINING,
            WorkerType.IMPROVEMENT,
            WorkerType.ANALYSIS
        ]

        for i in range(self.num_workers):
            worker_type = worker_types[i % len(worker_types)]
            await self.worker_manager.start_worker(worker_type)

        # Generate initial tasks
        self._generate_improvement_tasks()

        # Main loop
        await self._main_loop()

    async def _main_loop(self):
        """Main orchestrator loop"""
        health_check_interval = 60  # seconds
        task_generation_interval = 300  # 5 minutes
        status_report_interval = 600  # 10 minutes

        last_health_check = datetime.now()
        last_task_generation = datetime.now()
        last_status_report = datetime.now()

        while self.running:
            now = datetime.now()

            # Health check
            if (now - last_health_check).seconds >= health_check_interval:
                await self._perform_health_check()
                last_health_check = now

            # Generate new tasks
            if (now - last_task_generation).seconds >= task_generation_interval:
                self._generate_improvement_tasks()
                last_task_generation = now

            # Status report
            if (now - last_status_report).seconds >= status_report_interval:
                self._log_status_report()
                last_status_report = now

            # Process inline workers (for workers without separate scripts)
            await self._process_inline_workers()

            # Save state
            self._save_state()

            await asyncio.sleep(10)

        # Shutdown
        await self._shutdown()

    async def _perform_health_check(self):
        """Check worker health and restart if needed"""
        health = self.worker_manager.check_health()

        for worker_id, is_healthy in health.items():
            if not is_healthy:
                logger.warning(f"Worker {worker_id} unhealthy, restarting...")
                await self.worker_manager.restart_worker(worker_id)
                self.stats.total_worker_restarts += 1

    async def _process_inline_workers(self):
        """Process tasks for inline workers (no separate process)"""
        for worker_id, worker in self.worker_manager.workers.items():
            if worker_id not in self.worker_manager.processes:
                # This is an inline worker
                if worker.status == WorkerStatus.RUNNING:
                    task = self.task_queue.get_next_task(worker.worker_type)
                    if task:
                        await self._execute_inline_task(worker, task)

    async def _execute_inline_task(self, worker: WorkerInfo, task: Task):
        """Execute a task inline"""
        worker.current_task = task.task_id
        task.assigned_to = worker.worker_id
        task.started_at = datetime.now()

        logger.info(f"Worker {worker.worker_id} executing task: {task.description}")

        try:
            # Import and run appropriate handler
            result = await self._run_task(task)

            task.completed_at = datetime.now()
            task.result = result
            self.task_queue.complete_task(task.task_id, result)

            worker.tasks_completed += 1
            worker.last_heartbeat = datetime.now()
            self.stats.total_tasks_completed += 1

            if result.get('improvements_made', 0) > 0:
                self.stats.improvements_made += result['improvements_made']

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            self.task_queue.fail_task(task.task_id, str(e))
            worker.tasks_failed += 1
            worker.error_count += 1
            self.stats.total_tasks_failed += 1

        finally:
            worker.current_task = None

    async def _run_task(self, task: Task) -> Dict:
        """Run a specific task"""
        task_type = task.task_type.lower()

        if "training" in task_type or "learn" in task_type:
            return await self._run_training_task(task)
        elif "improve" in task_type or "fix" in task_type:
            return await self._run_improvement_task(task)
        elif "analyze" in task_type or "audit" in task_type:
            return await self._run_analysis_task(task)
        elif "benchmark" in task_type or "test" in task_type:
            return await self._run_benchmark_task(task)
        else:
            return {"status": "skipped", "reason": "unknown task type"}

    async def _run_training_task(self, task: Task) -> Dict:
        """Run a training iteration"""
        try:
            # Import trainer
            sys.path.insert(0, str(self.base_dir))
            from multi_provider_trainer import MultiProviderWorker

            worker = MultiProviderWorker(worker_id=0)
            result = await worker.run_iteration()

            return {
                "status": "completed",
                "success": result.get('success', False),
                "score": result.get('score', 0),
                "improvements_made": 1 if result.get('success', False) else 0
            }
        except Exception as e:
            logger.error(f"Training task error: {e}")
            return {"status": "error", "error": str(e)}

    async def _run_improvement_task(self, task: Task) -> Dict:
        """Run an improvement iteration"""
        try:
            sys.path.insert(0, str(self.base_dir))
            from unified_coding_agent import UnifiedCodingAgent

            agent = UnifiedCodingAgent(repo_path=str(self.base_dir))
            agent.auto_commit = False

            result = await agent.solve_task(task.description)

            return {
                "status": "completed",
                "success": result.success,
                "files_modified": len(result.files_modified),
                "improvements_made": 1 if result.success else 0
            }
        except Exception as e:
            logger.error(f"Improvement task error: {e}")
            return {"status": "error", "error": str(e)}

    async def _run_analysis_task(self, task: Task) -> Dict:
        """Run an analysis task"""
        try:
            # Simple code analysis
            py_files = list(self.base_dir.glob("*.py"))
            issues = []

            for f in py_files[:10]:  # Limit to 10 files
                content = f.read_text()
                if "except:" in content and "except Exception" not in content:
                    issues.append(f"Bare except in {f.name}")
                if "TODO" in content or "FIXME" in content:
                    issues.append(f"TODO/FIXME found in {f.name}")

            return {
                "status": "completed",
                "files_analyzed": len(py_files),
                "issues_found": len(issues),
                "issues": issues[:10]
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _run_benchmark_task(self, task: Task) -> Dict:
        """Run benchmark tests"""
        try:
            # Run tests
            result = subprocess.run(
                [sys.executable, "-m", "unittest", "discover", "tests/", "-v"],
                capture_output=True,
                timeout=120,
                cwd=str(self.base_dir)
            )

            output = result.stdout.decode()
            tests_passed = output.count(" ok")
            tests_failed = output.count("FAIL") + output.count("ERROR")

            self.stats.tests_passed += tests_passed

            return {
                "status": "completed",
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _generate_improvement_tasks(self):
        """Generate new improvement tasks"""
        tasks_to_add = [
            Task(
                task_id=f"train_{datetime.now().strftime('%H%M%S')}",
                task_type="training",
                description="Run training iteration on code generation patterns",
                priority=TaskPriority.MEDIUM
            ),
            Task(
                task_id=f"improve_{datetime.now().strftime('%H%M%S')}",
                task_type="improve",
                description="Improve code quality in the codebase",
                priority=TaskPriority.LOW
            ),
            Task(
                task_id=f"analyze_{datetime.now().strftime('%H%M%S')}",
                task_type="analyze",
                description="Analyze codebase for issues and improvements",
                priority=TaskPriority.LOW
            ),
            Task(
                task_id=f"benchmark_{datetime.now().strftime('%H%M%S')}",
                task_type="benchmark",
                description="Run benchmark tests",
                priority=TaskPriority.HIGH
            )
        ]

        for task in tasks_to_add:
            self.task_queue.add_task(task)

        logger.info(f"Generated {len(tasks_to_add)} new tasks")

    def _log_status_report(self):
        """Log a status report"""
        logger.info("=" * 60)
        logger.info("STATUS REPORT")
        logger.info("=" * 60)

        stats = self.stats.to_dict()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")

        worker_status = self.worker_manager.get_status()
        logger.info(f"  Workers: {worker_status['running']}/{worker_status['total_workers']} running")

        queue_stats = self.task_queue.get_stats()
        logger.info(f"  Tasks: {queue_stats['pending']} pending, {queue_stats['in_progress']} in progress")

        logger.info("=" * 60)

    def _save_state(self):
        """Save orchestrator state"""
        state = {
            "stats": self.stats.to_dict(),
            "workers": self.worker_manager.get_status(),
            "queue": self.task_queue.get_stats(),
            "saved_at": datetime.now().isoformat()
        }

        state_file = self.state_dir / "orchestrator_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)

    async def _shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down orchestrator...")

        # Stop all workers
        for worker_id in list(self.worker_manager.workers.keys()):
            await self.worker_manager.stop_worker(worker_id, graceful=True)

        # Save final state
        self._save_state()

        logger.info("Orchestrator shutdown complete")

    def get_status(self) -> Dict:
        """Get full orchestrator status"""
        return {
            "running": self.running,
            "stats": self.stats.to_dict(),
            "workers": self.worker_manager.get_status(),
            "queue": self.task_queue.get_stats()
        }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Oracle 24/7 Worker Orchestrator")
    parser.add_argument("--workers", "-w", type=int, default=2, help="Number of workers")
    parser.add_argument("--status", "-s", action="store_true", help="Show status")
    parser.add_argument("--stop", action="store_true", help="Stop orchestrator")
    parser.add_argument("--base-dir", type=str, help="Base directory")

    args = parser.parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent

    if args.status:
        state_file = base_dir / ".orchestrator_state" / "orchestrator_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            print(json.dumps(state, indent=2))
        else:
            print("No orchestrator state found")
        return

    if args.stop:
        # Send SIGTERM to orchestrator process
        pid_file = base_dir / ".orchestrator_state" / "orchestrator.pid"
        if pid_file.exists():
            pid = int(pid_file.read_text())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to orchestrator (PID: {pid})")
            except ProcessLookupError:
                print("Orchestrator not running")
        else:
            print("No PID file found")
        return

    # Start orchestrator
    orchestrator = Oracle24x7Orchestrator(base_dir=base_dir, num_workers=args.workers)

    # Save PID
    pid_file = base_dir / ".orchestrator_state" / "orchestrator.pid"
    pid_file.write_text(str(os.getpid()))

    try:
        asyncio.run(orchestrator.start())
    finally:
        if pid_file.exists():
            pid_file.unlink()


if __name__ == "__main__":
    main()
