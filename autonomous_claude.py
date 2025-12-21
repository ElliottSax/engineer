#!/usr/bin/env python3
"""
Autonomous Claude Code Headless Automation
Runs Claude Code in headless mode to work on tasks autonomously
"""

import subprocess
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import sys

# Import utilities for safe file operations
try:
    from utils.file_locking import safe_read_json, safe_write_json, StateFileManager
    from utils.config_loader import get_config
    FILE_LOCKING_AVAILABLE = True
except ImportError:
    FILE_LOCKING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_claude.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a task for Claude to work on"""
    id: str
    prompt: str
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    status: str = "pending"  # pending, in_progress, completed, failed
    session_id: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = None
    completed_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class Session:
    """Manages a Claude Code conversation session"""
    session_id: str
    tasks_completed: List[str]
    created_at: str
    last_activity: str


class AutonomousClaude:
    """Manages autonomous Claude Code execution"""

    def __init__(self,
                 task_file: str = "tasks.json",
                 state_file: str = "autonomous_state.json",
                 output_format: str = "json",
                 working_dir: str = None):
        self.task_file = Path(task_file)
        self.state_file = Path(state_file)
        self.output_format = output_format
        self.working_dir = working_dir or str(Path.cwd())

        self.tasks: List[Task] = []
        self.current_session: Optional[Session] = None
        self.total_tasks_completed = 0

        self.load_state()

    def load_state(self) -> None:
        """Load saved state from file with proper locking."""
        if not self.state_file.exists():
            return

        try:
            if FILE_LOCKING_AVAILABLE:
                state = safe_read_json(self.state_file, default={})
            else:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)

            self.tasks = [Task(**t) for t in state.get('tasks', [])]
            self.total_tasks_completed = state.get('total_tasks_completed', 0)

            session_data = state.get('current_session')
            if session_data:
                self.current_session = Session(**session_data)

            logger.info(f"Loaded state: {len(self.tasks)} tasks")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in state file: {e}")
        except (KeyError, TypeError) as e:
            logger.error(f"Invalid state file structure: {e}")
        except IOError as e:
            logger.error(f"IO error loading state: {e}")

    def save_state(self) -> bool:
        """Save current state to file with proper locking."""
        state = {
            'tasks': [asdict(t) for t in self.tasks],
            'current_session': asdict(self.current_session) if self.current_session else None,
            'total_tasks_completed': self.total_tasks_completed,
            'last_updated': datetime.now().isoformat()
        }

        try:
            if FILE_LOCKING_AVAILABLE:
                return safe_write_json(self.state_file, state, backup=True)
            else:
                with open(self.state_file, 'w', encoding='utf-8') as f:
                    json.dump(state, f, indent=2)
                return True

        except json.JSONEncodeError as e:
            logger.error(f"Failed to serialize state: {e}")
            return False
        except IOError as e:
            logger.error(f"IO error saving state: {e}")
            return False
        except OSError as e:
            logger.error(f"OS error saving state: {e}")
            return False

    def load_tasks(self) -> None:
        """Load tasks from task file with proper error handling."""
        if not self.task_file.exists():
            return

        try:
            if FILE_LOCKING_AVAILABLE:
                task_data = safe_read_json(self.task_file, default={'tasks': []})
            else:
                with open(self.task_file, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)

            tasks_added = 0
            for item in task_data.get('tasks', []):
                try:
                    task = Task(
                        id=item['id'],
                        prompt=item['prompt'],
                        priority=item.get('priority', 0)
                    )
                    self.add_task(task)
                    tasks_added += 1
                except KeyError as e:
                    logger.warning(f"Skipping malformed task (missing {e}): {item.get('id', 'unknown')}")
                except TypeError as e:
                    logger.warning(f"Invalid task data type: {e}")

            logger.info(f"Loaded {tasks_added} tasks from {self.task_file}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in task file: {e}")
        except IOError as e:
            logger.error(f"IO error loading tasks: {e}")

    def add_task(self, task: Task):
        """Add a new task to the queue"""
        # Check if task already exists
        if not any(t.id == task.id for t in self.tasks):
            self.tasks.append(task)
            self.tasks.sort(key=lambda t: t.priority, reverse=True)
            logger.info(f"Added task: {task.id}")
            self.save_state()

    def get_next_task(self) -> Optional[Task]:
        """Get the next pending task"""
        for task in self.tasks:
            if task.status == "pending":
                return task
        return None

    def execute_claude_command(self, prompt: str, session_id: str = None, continue_session: bool = False) -> Dict:
        """Execute a Claude Code headless command"""
        try:
            cmd = ["claude", "-p", prompt, "--output-format", self.output_format]

            if continue_session:
                cmd.extend(["--continue"])
            elif session_id:
                cmd.extend(["--resume", session_id])

            logger.info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                if self.output_format == "json":
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        return {"output": result.stdout, "error": None}
                else:
                    return {"output": result.stdout, "error": None}
            else:
                return {"output": None, "error": result.stderr}

        except subprocess.TimeoutExpired:
            logger.error("Command timed out")
            return {"output": None, "error": "Command timed out"}
        except Exception as e:
            logger.error(f"Error executing command: {e}")
            return {"output": None, "error": str(e)}

    def process_task(self, task: Task) -> bool:
        """Process a single task"""
        logger.info(f"Processing task {task.id}: {task.prompt[:50]}...")

        task.status = "in_progress"
        self.save_state()

        # Execute the task
        result = self.execute_claude_command(
            task.prompt,
            session_id=self.current_session.session_id if self.current_session else None,
            continue_session=bool(self.current_session)
        )

        if result.get('error'):
            logger.error(f"Task {task.id} failed: {result['error']}")
            task.error = result['error']
            task.retry_count += 1

            if task.retry_count >= task.max_retries:
                task.status = "failed"
                logger.error(f"Task {task.id} failed permanently after {task.retry_count} retries")
            else:
                task.status = "pending"
                logger.info(f"Task {task.id} will be retried ({task.retry_count}/{task.max_retries})")

            self.save_state()
            return False
        else:
            logger.info(f"Task {task.id} completed successfully")
            task.status = "completed"
            task.result = json.dumps(result.get('output'))
            task.completed_at = datetime.now().isoformat()

            # Update session
            if self.current_session:
                self.current_session.tasks_completed.append(task.id)
                self.current_session.last_activity = datetime.now().isoformat()

            self.total_tasks_completed += 1
            self.save_state()
            return True

    def start_new_session(self):
        """Start a new Claude conversation session"""
        self.current_session = Session(
            session_id=f"auto_{int(time.time())}",
            tasks_completed=[],
            created_at=datetime.now().isoformat(),
            last_activity=datetime.now().isoformat()
        )
        logger.info(f"Started new session: {self.current_session.session_id}")

    def run_once(self) -> bool:
        """Process one task from the queue"""
        task = self.get_next_task()

        if not task:
            logger.info("No pending tasks")
            return False

        if not self.current_session:
            self.start_new_session()

        success = self.process_task(task)
        return success

    def run_continuous(self, interval: int = 10, max_iterations: int = None):
        """Run continuously, processing tasks as they arrive"""
        logger.info("Starting continuous mode")
        iterations = 0

        try:
            while True:
                # Load new tasks from file
                self.load_tasks()

                # Process next task
                self.run_once()

                iterations += 1
                if max_iterations and iterations >= max_iterations:
                    logger.info(f"Reached max iterations: {max_iterations}")
                    break

                # Check if there are more tasks
                if not self.get_next_task():
                    logger.info(f"No more tasks. Waiting {interval} seconds...")
                    time.sleep(interval)
                else:
                    # Small delay between tasks
                    time.sleep(2)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.save_state()
            self.print_summary()

    def run_all(self):
        """Process all pending tasks"""
        logger.info("Processing all pending tasks")

        if not self.current_session:
            self.start_new_session()

        while True:
            task = self.get_next_task()
            if not task:
                break

            self.process_task(task)
            time.sleep(2)  # Small delay between tasks

        self.save_state()
        self.print_summary()

    def print_summary(self):
        """Print execution summary"""
        pending = sum(1 for t in self.tasks if t.status == "pending")
        in_progress = sum(1 for t in self.tasks if t.status == "in_progress")
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")

        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total tasks: {len(self.tasks)}")
        logger.info(f"Pending: {pending}")
        logger.info(f"In Progress: {in_progress}")
        logger.info(f"Completed: {completed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total completed (all time): {self.total_tasks_completed}")
        logger.info("=" * 60)

    def get_status(self) -> Dict:
        """Get current status"""
        return {
            'total_tasks': len(self.tasks),
            'pending': sum(1 for t in self.tasks if t.status == "pending"),
            'in_progress': sum(1 for t in self.tasks if t.status == "in_progress"),
            'completed': sum(1 for t in self.tasks if t.status == "completed"),
            'failed': sum(1 for t in self.tasks if t.status == "failed"),
            'total_completed_all_time': self.total_tasks_completed,
            'current_session': self.current_session.session_id if self.current_session else None
        }


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Autonomous Claude Code Automation')
    parser.add_argument('mode', choices=['once', 'all', 'continuous', 'status'],
                       help='Execution mode')
    parser.add_argument('--task-file', default='tasks.json',
                       help='Task queue file (default: tasks.json)')
    parser.add_argument('--state-file', default='autonomous_state.json',
                       help='State file (default: autonomous_state.json)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Check interval in continuous mode (seconds)')
    parser.add_argument('--max-iterations', type=int,
                       help='Max iterations in continuous mode')
    parser.add_argument('--working-dir',
                       help='Working directory for Claude commands')

    args = parser.parse_args()

    claude = AutonomousClaude(
        task_file=args.task_file,
        state_file=args.state_file,
        working_dir=args.working_dir
    )

    # Load tasks from file
    claude.load_tasks()

    if args.mode == 'once':
        claude.run_once()
        claude.print_summary()
    elif args.mode == 'all':
        claude.run_all()
    elif args.mode == 'continuous':
        claude.run_continuous(interval=args.interval, max_iterations=args.max_iterations)
    elif args.mode == 'status':
        status = claude.get_status()
        print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
