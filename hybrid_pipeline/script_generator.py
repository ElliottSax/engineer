"""
Advanced Script Generation System for YouTube Explainer Videos
Targets high-CPM niches with "hidden knowledge" angle
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
import re


@dataclass
class ScriptSegment:
    """Individual script segment with timing"""
    scene_number: int
    scene_type: str
    narration: str
    duration_seconds: int
    visual_description: str
    importance: str  # high, medium, low
    includes_hook: bool = False


class ExplainerScriptGenerator:
    """
    Generate engaging explainer video scripts
    Style: David Attenborough's authority + VSauce intrigue
    """

    # High-CPM topic templates
    HIGH_CPM_TOPICS = {
        "finance": [
            "How {subject} Secretly Controls Your Money",
            "The Hidden {subject} Strategy Banks Don't Want You to Know",
            "Why Rich People Use {subject} (And You Should Too)"
        ],
        "business": [
            "How {subject} Built a Billion-Dollar Empire",
            "The {subject} Method That's Disrupting Industries",
            "What {subject} Knows That You Don't"
        ],
        "ai": [
            "How AI is Secretly {action}",
            "The {subject} AI Revolution Nobody's Talking About",
            "Why {subject} Will Be Replaced by AI (And When)"
        ],
        "productivity": [
            "The {subject} Technique That 10x'd My Output",
            "How Top Performers Use {subject}",
            "The Hidden Science of {subject}"
        ]
    }

    # Signature phrases to weave throughout
    SIGNATURE_PHRASES = [
        "Once you know how this works, everything changes.",
        "Once you know how to spot this, you'll see it everywhere.",
        "Once you know how they do it, you can't unsee it.",
        "Once you know how simple it is, you'll wonder why everyone doesn't do it.",
        "Once you know how powerful this is, there's no going back."
    ]

    def __init__(self):
        self.scene_duration = 5  # Average scene duration in seconds

    def generate_full_script(self, topic: str, video_duration_minutes: int = 6) -> Dict:
        """
        Generate complete script for video
        Args:
            topic: Main topic/title of video
            video_duration_minutes: Target duration (5-8 minutes typically)
        """

        total_seconds = video_duration_minutes * 60
        num_scenes = total_seconds // self.scene_duration

        script = {
            "title": self._generate_title(topic),
            "duration_seconds": total_seconds,
            "segments": []
        }

        # Script structure for 6-minute video
        structure = self._get_script_structure(num_scenes)

        for i, scene_type in enumerate(structure):
            segment = self._generate_segment(
                scene_number=i + 1,
                scene_type=scene_type,
                topic=topic,
                is_final=(i == len(structure) - 1)
            )
            script["segments"].append(segment)

        return script

    def _generate_title(self, topic: str) -> str:
        """Generate compelling title with high CTR potential"""

        templates = [
            f"The Hidden Truth About {topic}",
            f"What Nobody Tells You About {topic}",
            f"How {topic} Actually Works (It's Not What You Think)",
            f"{topic}: The Secret Everyone Should Know",
            f"The {topic} Trick That Changes Everything"
        ]

        # Select template based on topic keywords
        if "ai" in topic.lower() or "tech" in topic.lower():
            return f"How {topic} is Secretly Changing Everything"
        elif "money" in topic.lower() or "finance" in topic.lower():
            return f"The {topic} Secret That Could Make You Rich"
        else:
            return templates[0]

    def _get_script_structure(self, num_scenes: int) -> List[str]:
        """Get optimal script structure for engagement"""

        if num_scenes <= 60:  # 5 minutes or less
            structure = [
                "hook",
                "problem_setup",
                "teaser",
                "explanation_1",
                "example_1",
                "deeper_dive",
                "explanation_2",
                "example_2",
                "counterpoint",
                "resolution",
                "call_to_action",
                "conclusion"
            ]
        else:  # 6-8 minutes
            structure = [
                "hook",
                "problem_setup",
                "teaser",
                "background",
                "explanation_1",
                "example_1",
                "mini_hook",
                "deeper_dive",
                "explanation_2",
                "example_2",
                "data_proof",
                "counterpoint",
                "advanced_insight",
                "practical_application",
                "resolution",
                "call_to_action",
                "conclusion"
            ]

        # Extend or trim to match exact scene count
        while len(structure) < num_scenes:
            # Add more examples and explanations
            structure.insert(-2, "additional_example")

        return structure[:num_scenes]

    def _generate_segment(self, scene_number: int, scene_type: str,
                         topic: str, is_final: bool) -> ScriptSegment:
        """Generate individual script segment"""

        narration = self._get_narration(scene_type, topic, is_final)
        visual = self._get_visual_description(scene_type)
        importance = self._get_importance(scene_type)

        # Add signature phrase strategically
        includes_hook = False
        if scene_type in ["hook", "mini_hook", "resolution", "conclusion"]:
            includes_hook = True
            if not is_final:
                narration += " " + self.SIGNATURE_PHRASES[0]
            else:
                narration += " " + self.SIGNATURE_PHRASES[0].replace("changes", "makes sense")

        return ScriptSegment(
            scene_number=scene_number,
            scene_type=scene_type,
            narration=narration,
            duration_seconds=self._get_duration(scene_type),
            visual_description=visual,
            importance=importance,
            includes_hook=includes_hook
        )

    def _get_narration(self, scene_type: str, topic: str, is_final: bool) -> str:
        """Get narration text for scene type"""

        narrations = {
            "hook": f"What if I told you that {topic} isn't what you think it is? In fact, what you're about to learn might completely change how you see it.",

            "problem_setup": f"Most people think {topic} works one way. They follow the conventional wisdom, do what everyone else does, and get mediocre results. But there's something they're missing.",

            "teaser": f"In the next few minutes, I'm going to show you exactly how {topic} really works, why almost everyone gets it wrong, and how you can use this knowledge to your advantage.",

            "background": f"To understand {topic}, we need to go back to where it all started. You see, this isn't a new phenomenon - it's been hiding in plain sight for years.",

            "explanation_1": f"Here's how {topic} actually works: It's not about complexity, it's about understanding the fundamental principles that drive everything. Think of it like this...",

            "example_1": f"Let me give you a real example. Last year, someone discovered this about {topic} and the results were extraordinary. They went from zero to significant impact in just months.",

            "mini_hook": f"But here's where it gets really interesting. What I'm about to show you next is something 99% of people never realize about {topic}.",

            "deeper_dive": f"Now, let's go deeper. The real power of {topic} isn't in what you can see on the surface. It's in the hidden mechanisms that operate behind the scenes.",

            "explanation_2": f"This works because of a simple principle: {topic} follows patterns. Once you recognize these patterns, you can predict and leverage them. It's almost like having a superpower.",

            "example_2": f"Consider this scenario: Two people approach {topic} differently. One follows conventional wisdom, the other uses what you're learning now. The difference in results? Night and day.",

            "data_proof": f"The numbers back this up. Studies show that understanding {topic} this way leads to dramatically better outcomes. We're talking about improvements of 200%, 300%, sometimes even more.",

            "counterpoint": f"Now, some people will tell you {topic} doesn't work this way. They'll give you dozens of reasons why the conventional approach is better. But here's what they're not considering...",

            "advanced_insight": f"For those ready to take this further, there's an advanced technique with {topic}. It's not for everyone, but if you're serious about mastering this, it's game-changing.",

            "practical_application": f"So how do you actually use this? Start small. Apply these principles to {topic} in your own situation. Test it, measure the results, and iterate.",

            "resolution": f"Everything we've covered about {topic} boils down to this: It's not about working harder, it's about understanding the system and working smarter.",

            "call_to_action": f"Now that you understand how {topic} really works, you have a choice. You can go back to the old way, or you can apply what you've learned and see real results.",

            "conclusion": f"Remember, {topic} is everywhere once you know how to see it. The question isn't whether this works - it's whether you're ready to use it.",

            "additional_example": f"Here's another fascinating aspect of {topic} that most people miss. It's subtle, but once you see it, everything clicks into place."
        }

        return narrations.get(scene_type, f"Let's explore another aspect of {topic}.")

    def _get_visual_description(self, scene_type: str) -> str:
        """Get visual description for scene"""

        visuals = {
            "hook": "Single figure with questioning pose, question mark icon",
            "problem_setup": "Figure thinking deeply, gradient shift to darker tones",
            "teaser": "Figure pointing upward excitedly, lightbulb appearing",
            "background": "Two figures, one teaching, one learning",
            "explanation_1": "Figure explaining with hand gestures, floating icons",
            "example_1": "Figure pointing at chart showing upward trend",
            "mini_hook": "Figure with realization pose, bright background",
            "deeper_dive": "Figure working at desk, focused concentration",
            "explanation_2": "Figure presenting with confidence, data visualization",
            "example_2": "Split scene with two figures showing contrast",
            "data_proof": "Figure with bar chart, impressive numbers visible",
            "counterpoint": "Figure in thoughtful pose, considering alternatives",
            "advanced_insight": "Figure in triumphant pose, multiple icons",
            "practical_application": "Figure demonstrating with hand gestures",
            "resolution": "Figure in welcoming pose, warm lighting",
            "call_to_action": "Figure pointing forward, inviting viewer",
            "conclusion": "Figure in confident final pose, checkmark icon",
            "additional_example": "Figure explaining with supporting visual elements"
        }

        return visuals.get(scene_type, "Figure in explanatory pose")

    def _get_duration(self, scene_type: str) -> int:
        """Get optimal duration for scene type"""

        # Vary durations for pacing
        durations = {
            "hook": 7,  # Longer for impact
            "mini_hook": 4,  # Quick punch
            "explanation_1": 6,
            "explanation_2": 6,
            "example_1": 5,
            "example_2": 5,
            "conclusion": 6,
            "data_proof": 5,
            "deeper_dive": 7
        }

        return durations.get(scene_type, 5)  # Default 5 seconds

    def _get_importance(self, scene_type: str) -> str:
        """Determine scene importance for model selection"""

        high_importance = ["hook", "mini_hook", "resolution", "conclusion",
                          "explanation_1", "explanation_2"]
        low_importance = ["additional_example", "background"]

        if scene_type in high_importance:
            return "high"
        elif scene_type in low_importance:
            return "low"
        else:
            return "medium"


class NarrationOptimizer:
    """Optimize narration for TTS and pacing"""

    def __init__(self):
        self.words_per_minute = 150  # Optimal for clarity
        self.pause_markers = {
            ".": 0.5,  # Period pause
            ",": 0.2,  # Comma pause
            "...": 1.0,  # Dramatic pause
            ":": 0.3,  # Colon pause
            "—": 0.4  # Dash pause
        }

    def optimize_for_tts(self, text: str) -> str:
        """Optimize text for TTS processing"""

        # Add emphasis markers
        text = self._add_emphasis_markers(text)

        # Optimize sentence length
        text = self._optimize_sentence_length(text)

        # Add pause markers
        text = self._add_pause_markers(text)

        return text

    def _add_emphasis_markers(self, text: str) -> str:
        """Add markers for TTS emphasis"""

        emphasis_words = ["secret", "hidden", "revealed", "truth", "actually",
                         "really", "never", "always", "everything", "nothing"]

        for word in emphasis_words:
            text = text.replace(f" {word} ", f" *{word}* ")

        return text

    def _optimize_sentence_length(self, text: str) -> str:
        """Break long sentences for better pacing"""

        sentences = text.split(". ")
        optimized = []

        for sentence in sentences:
            if len(sentence.split()) > 20:
                # Break into smaller chunks
                parts = sentence.split(", ")
                if len(parts) > 1:
                    mid = len(parts) // 2
                    sentence = ", ".join(parts[:mid]) + ". " + ", ".join(parts[mid:])
            optimized.append(sentence)

        return ". ".join(optimized)

    def _add_pause_markers(self, text: str) -> str:
        """Add pause markers for dramatic effect"""

        # Add pauses before reveals
        text = text.replace("But here's", "... But here's")
        text = text.replace("The truth is", "... The truth is")
        text = text.replace("What if", "... What if")

        return text

    def calculate_duration(self, text: str) -> float:
        """Calculate speaking duration for text"""

        word_count = len(text.split())
        base_duration = (word_count / self.words_per_minute) * 60

        # Add pause time
        pause_time = 0
        for marker, duration in self.pause_markers.items():
            pause_time += text.count(marker) * duration

        return base_duration + pause_time