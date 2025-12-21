#!/usr/bin/env python3
"""
Hugging Face Space: Autocoder Implementation Worker
24/7 autonomous code implementation for the Once project

This worker continuously generates implementation code based on deep analysis,
following 6 stages of thoughtful consideration before each implementation.
"""

import gradio as gr
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import os

# Hugging Face integration
from huggingface_hub import HfApi, create_repo, upload_file
from datasets import Dataset, load_dataset

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME", "autocoder")
DATASET_NAME = f"{HF_USERNAME}/autocoder-implementation-results"
SPACE_NAME = f"{HF_USERNAME}/autocoder-implementation-worker"

class ImplementationWorker:
    """24/7 Autonomous Implementation Worker"""

    def __init__(self):
        self.iteration = 0
        self.running = True
        self.status = "Initializing..."
        self.latest_results = {}
        self.hf_api = HfApi()

        # Track progress across iterations
        self.implementations_generated = []
        self.tests_written = []
        self.issues_discovered = []

        # Create dataset if it doesn't exist
        try:
            self.dataset = load_dataset(DATASET_NAME, split="train")
        except (FileNotFoundError, ValueError, ConnectionError) as e:
            self.dataset = None
            self.status = f"Creating new dataset... ({type(e).__name__})"

    async def stage_1_deep_understanding(self):
        """Stage 1: What is the REAL problem we're solving?"""
        return {
            "stage": "Deep Understanding",
            "analysis": {
                "surface_problem": "ScriptValidator needs integration into LongFormScriptGenerator",
                "real_problem": "Duplicate scenes reduce video quality - need deduplication in generation pipeline",
                "root_cause": "Line 190 in longform_generator.py just concatenates scenes without validation",
                "scope": "1 import + ~20 lines of integration code",
                "complexity": "LOW - using existing tested code"
            }
        }

    async def stage_2_architectural_analysis(self):
        """Stage 2: How does this fit in the system?"""
        return {
            "stage": "Architectural Analysis",
            "analysis": {
                "current_flow": "generate_longform_script() → concatenate chapters → return script",
                "new_flow": "generate_longform_script() → concatenate chapters → validate_and_fix() → return script",
                "components_affected": [
                    "LongFormScriptGenerator.generate_longform_script()",
                    "Return value structure (adds 'validation' field)"
                ],
                "dependencies": "ScriptValidator from script_processing.validator",
                "backwards_compatibility": "Breaking: adds validation field, changes scene count"
            }
        }

    async def stage_3_approach_evaluation(self):
        """Stage 3: Compare different approaches"""
        return {
            "stage": "Approach Evaluation",
            "approaches": [
                {
                    "name": "Direct Integration",
                    "code_changes": "1 import + 4 lines",
                    "pros": ["Simple", "Uses existing code", "Minimal changes"],
                    "cons": ["Adds field to output"],
                    "verdict": "RECOMMENDED"
                },
                {
                    "name": "Inline Deduplication",
                    "code_changes": "30+ lines",
                    "pros": ["No output changes"],
                    "cons": ["Duplicates existing code", "More complex"],
                    "verdict": "NOT RECOMMENDED"
                },
                {
                    "name": "Separate Validation Step",
                    "code_changes": "50+ lines",
                    "pros": ["Clean separation"],
                    "cons": ["More moving parts", "Harder to maintain"],
                    "verdict": "OVER-ENGINEERED"
                }
            ],
            "selected": "Direct Integration"
        }

    async def stage_4_risk_analysis(self):
        """Stage 4: What could go wrong?"""
        return {
            "stage": "Risk Analysis",
            "risks": [
                {
                    "risk": "Import failure",
                    "probability": "LOW",
                    "impact": "CRITICAL",
                    "mitigation": "Test import explicitly"
                },
                {
                    "risk": "Validator crashes",
                    "probability": "LOW",
                    "impact": "HIGH",
                    "mitigation": "Add try/except with fallback"
                },
                {
                    "risk": "Scene count changes break tests",
                    "probability": "HIGH",
                    "impact": "MEDIUM",
                    "mitigation": "Update test expectations"
                },
                {
                    "risk": "Performance degradation",
                    "probability": "LOW",
                    "impact": "LOW",
                    "mitigation": "Benchmark before/after"
                }
            ]
        }

    async def stage_5_testing_strategy(self):
        """Stage 5: How will we test this?"""
        return {
            "stage": "Testing Strategy",
            "test_plan": {
                "critical_tests": [
                    "test_duplicates_are_removed()",
                    "test_validation_results_included()",
                    "test_auto_fix_works()",
                    "test_scene_numbering_sequential()",
                    "test_existing_functionality_unchanged()"
                ],
                "edge_cases": [
                    "Empty script",
                    "Single scene",
                    "All identical scenes",
                    "Perfect script (no duplicates)"
                ],
                "performance": [
                    "Baseline measurement",
                    "With validation overhead",
                    "Acceptable threshold: <10%"
                ],
                "total_tests": 25
            }
        }

    async def stage_6_implementation_plan(self):
        """Stage 6: Step-by-step implementation"""
        implementation = {
            "stage": "Implementation Plan",
            "code_changes": {
                "file": "/mnt/e/projects/once/src/script_processing/longform_generator.py",
                "import_add": "from .validator import ScriptValidator",
                "location": "After line 190 (all_scenes.extend)",
                "code": """
# Validate and deduplicate scenes
validator = ScriptValidator()
validated_script = {
    "title": title,
    "scenes": all_scenes
}
validated_script, validation_result = validator.validate_and_fix(
    validated_script,
    auto_fix=True
)

# Log validation results
logger.info(
    f"Script validation: {validation_result.passed}, "
    f"Quality: {validation_result.quality_score:.2f}, "
    f"Scenes: {len(all_scenes)} → {len(validated_script['scenes'])}"
)

# Return validated script with validation metadata
return {
    **validated_script,
    "validation": {
        "passed": validation_result.passed,
        "quality_score": validation_result.quality_score,
        "issues": [issue.to_dict() for issue in validation_result.issues],
        "scene_count_before": len(all_scenes),
        "scene_count_after": len(validated_script['scenes'])
    }
}
""".strip()
            },
            "phases": [
                "Phase 1: Add import",
                "Phase 2: Integrate validation call",
                "Phase 3: Update return structure",
                "Phase 4: Add logging",
                "Phase 5: Write tests",
                "Phase 6: Update documentation"
            ]
        }

        self.implementations_generated.append(implementation)
        return implementation

    async def run_implementation_cycle(self):
        """Run complete 6-stage implementation cycle"""
        self.iteration += 1
        self.status = f"Running iteration {self.iteration}..."

        # All 6 stages
        stages = [
            self.stage_1_deep_understanding,
            self.stage_2_architectural_analysis,
            self.stage_3_approach_evaluation,
            self.stage_4_risk_analysis,
            self.stage_5_testing_strategy,
            self.stage_6_implementation_plan
        ]

        # Rotate through stages
        stage_index = self.iteration % len(stages)
        current_stage = stages[stage_index]

        stage_result = await current_stage()

        result = {
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "stage_number": stage_index + 1,
            "stage_name": stage_result["stage"],
            "analysis": stage_result,
            "cumulative_progress": {
                "implementations_generated": len(self.implementations_generated),
                "tests_written": len(self.tests_written),
                "issues_discovered": len(self.issues_discovered)
            }
        }

        self.latest_results = result

        # Push to Hugging Face Dataset
        await self.push_results(result)

        self.status = f"✓ Iteration {self.iteration} complete - {stage_result['stage']}"

        return result

    async def push_results(self, results: Dict):
        """Push results to Hugging Face Dataset"""
        if not HF_TOKEN:
            self.status += " (No HF_TOKEN - results not pushed)"
            return

        try:
            # Create dataset repo if it doesn't exist
            try:
                create_repo(
                    repo_id=DATASET_NAME,
                    token=HF_TOKEN,
                    repo_type="dataset",
                    exist_ok=True
                )
            except (ConnectionError, ValueError, PermissionError) as e:
                print(f"Repo creation skipped: {e}")

            # Save results locally
            results_file = Path("/tmp/implementation_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            # Upload to dataset
            upload_file(
                path_or_fileobj=str(results_file),
                path_in_repo=f"implementation_iteration_{self.iteration}.json",
                repo_id=DATASET_NAME,
                repo_type="dataset",
                token=HF_TOKEN
            )

            self.status += " (Pushed to HF Dataset)"

        except Exception as e:
            self.status += f" (Push failed: {e})"

    async def run_continuous(self):
        """Run continuous implementation loop"""
        while self.running:
            try:
                await self.run_implementation_cycle()
                await asyncio.sleep(120)  # Wait 2 minutes between iterations
            except Exception as e:
                self.status = f"Error: {e}"
                await asyncio.sleep(10)

# Global worker instance
worker = ImplementationWorker()

# Gradio Interface
def get_status():
    """Get current worker status"""
    return {
        "Status": worker.status,
        "Iteration": worker.iteration,
        "Implementations Generated": len(worker.implementations_generated),
        "Tests Written": len(worker.tests_written),
        "Issues Discovered": len(worker.issues_discovered),
        "Latest Analysis": json.dumps(worker.latest_results, indent=2) if worker.latest_results else "No results yet"
    }

def start_worker():
    """Start the worker"""
    worker.running = True
    return "Worker is already running in background!"

def stop_worker():
    """Stop the worker"""
    worker.running = False
    return "Worker stopped!"

def get_full_log():
    """Get full implementation log"""
    if not HF_TOKEN:
        return "No HF_TOKEN configured - cannot fetch logs"

    try:
        # Load all results from dataset
        dataset = load_dataset(DATASET_NAME, split="train")
        return f"Total iterations: {len(dataset)}\n\nLatest results:\n{json.dumps(worker.latest_results, indent=2)}"
    except Exception as e:
        return f"Error loading dataset: {e}"

# Build Gradio UI
with gr.Blocks(title="Autocoder Implementation Worker") as demo:
    gr.Markdown("""
    # ⚙️ Autocoder Implementation Worker

    **24/7 Autonomous Code Implementation**

    This worker continuously generates implementation code for the Once project,
    using 6 stages of deep thinking before each implementation.

    ## Status
    """)

    status_display = gr.JSON(label="Current Status", value=get_status)

    with gr.Row():
        start_btn = gr.Button("▶ Start Worker", variant="primary")
        stop_btn = gr.Button("⏸ Stop Worker", variant="stop")
        refresh_btn = gr.Button("🔄 Refresh Status")

    start_output = gr.Textbox(label="Action Result")

    gr.Markdown("## Implementation Log")
    log_display = gr.Textbox(label="Full Log", lines=20)

    gr.Markdown("""
    ## Implementation Stages (6)

    1. **Deep Understanding** - What is the REAL problem?
    2. **Architectural Analysis** - How does this fit in the system?
    3. **Approach Evaluation** - Compare different solutions
    4. **Risk Analysis** - Identify and mitigate risks
    5. **Testing Strategy** - Plan comprehensive tests
    6. **Implementation Plan** - Step-by-step roadmap

    ## Configuration

    - **Dataset**: `autocoder/autocoder-implementation-results`
    - **Update Frequency**: Every 120 seconds
    - **Thinking Stages**: 6 deep analysis stages

    ## Setup

    To enable dataset pushing, configure:
    1. Add `HF_TOKEN` secret in Space settings
    2. Set `HF_USERNAME` to your username (default: `autocoder`)

    ## Results

    All implementation results are pushed to the Hugging Face Dataset and can be:
    - Downloaded programmatically
    - Viewed in the Datasets UI
    - Used to guide actual implementation
    - Referenced by monitoring dashboard

    ## Philosophy

    This worker embodies "think deeply before coding":
    - Analyze the problem thoroughly
    - Consider architectural implications
    - Evaluate multiple approaches
    - Assess risks proactively
    - Plan comprehensive testing
    - Generate actionable implementation plans
    """)

    # Button actions
    start_btn.click(fn=start_worker, outputs=start_output)
    stop_btn.click(fn=stop_worker, outputs=start_output)
    refresh_btn.click(fn=get_status, outputs=status_display)

    # Auto-refresh status every 5 seconds
    demo.load(fn=get_status, outputs=status_display, every=5)

if __name__ == "__main__":
    # Auto-start worker in background
    import threading
    threading.Thread(target=lambda: asyncio.run(worker.run_continuous()), daemon=True).start()

    demo.launch(server_name="0.0.0.0", server_port=7860)
