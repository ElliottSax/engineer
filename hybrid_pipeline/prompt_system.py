"""
Advanced Minimalist Stick Figure Prompt Generation System
Specifically designed for the exact visual style requirements
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random


class FigurePose(Enum):
    """Detailed pose specifications for stick figures"""
    EXPLAINING = "confident standing pose with one arm extended palm-up in explaining gesture"
    POINTING_RIGHT = "standing in profile view pointing confidently to the right"
    POINTING_UP = "standing with arm raised pointing upward enthusiastically"
    THINKING = "contemplative pose with one hand raised to chin and slight head tilt"
    EXCITED = "dynamic pose with arms spread wide and body leaning forward excitedly"
    TEACHING = "authoritative stance with one arm gesturing outward in teaching motion"
    LISTENING = "attentive pose with head tilted and arms relaxed at sides"
    SUCCESS = "triumphant pose with both arms raised overhead in victory gesture"
    WORKING = "seated at simple line-drawn desk with minimal rectangular laptop shape, focused forward-leaning pose"
    PRESENTING = "standing confidently beside floating chart, one arm extended toward it"
    REALIZATION = "sudden insight pose with finger raised and slight backward lean"
    WELCOMING = "open stance with both arms slightly spread in welcoming gesture"


class BackgroundGradient(Enum):
    """Professional gradient color schemes"""
    BLUE_PURPLE = "blue-to-purple"
    TEAL_NAVY = "teal-to-navy"
    CORAL_AMBER = "warm coral-to-amber"
    VIOLET_BLUE = "violet-to-blue"
    GOLDEN_ORANGE = "golden-to-orange"
    GREEN_TEAL = "green-to-teal"
    MAGENTA_VIOLET = "magenta-to-violet"
    SLATE_INDIGO = "slate-to-indigo"
    ROSE_PINK = "soft rose-to-pink"
    CYAN_BLUE = "cyan-to-deep-blue"


class IconType(Enum):
    """Minimalist icon specifications"""
    DOLLAR = "simple black-outline dollar sign icon"
    LIGHTBULB = "simple black-outline lightbulb icon glowing softly"
    QUESTION = "simple question mark icon"
    CHECKMARK = "simple checkmark icon"
    CHART_BAR = "simple floating bar chart icon drawn in matching black outline style"
    CHART_LINE = "simple line graph icon with upward trend"
    CLOCK = "simple clock icon showing time"
    ARROW_UP = "simple upward arrow icon"
    GEAR = "simple gear icon"
    BRAIN = "simple brain icon in outline style"
    BOOK = "simple open book icon"
    TARGET = "simple target/bullseye icon"


@dataclass
class SceneComposition:
    """Complete scene composition details"""
    figures: int
    primary_pose: FigurePose
    secondary_pose: Optional[FigurePose]
    background: BackgroundGradient
    icons: List[IconType]
    lighting: str
    composition_notes: str


class MinimalistSceneGenerator:
    """Generate complete scene prompts matching exact visual requirements"""

    BASE_TEMPLATE = """Minimal vector illustration, {figure_desc} with thick consistent black outline,
perfectly round head with {eye_desc}, {pose_desc}, {icon_desc}{background} gradient background
with {lighting}, {composition}, clean professional educational style, no text, 16:9 aspect ratio"""

    def __init__(self):
        self.scene_memory = []  # Track recent scenes to ensure variety

    def generate_complete_prompt(self, scene_type: str, content: str,
                                emphasis: str = "explanation") -> str:
        """Generate a complete, production-ready prompt"""

        composition = self._determine_composition(scene_type, content, emphasis)

        # Build figure description
        if composition.figures == 1:
            figure_desc = "single white stick figure"
            eye_desc = "two small dot eyes"
            pose_desc = composition.primary_pose.value
        else:
            figure_desc = f"{composition.figures} white stick figures"
            eye_desc = "dot eyes"
            pose_desc = self._combine_poses(composition.primary_pose,
                                           composition.secondary_pose)

        # Build icon description
        icon_desc = self._format_icons(composition.icons)

        # Format background
        background = composition.background.value

        # Ensure variety in recent backgrounds
        if len(self.scene_memory) > 0:
            recent_bg = self.scene_memory[-1].get("background")
            if recent_bg == background:
                # Switch to alternative
                background = self._get_alternative_background(background)

        # Create final prompt
        prompt = self.BASE_TEMPLATE.format(
            figure_desc=figure_desc,
            eye_desc=eye_desc,
            pose_desc=pose_desc,
            icon_desc=icon_desc,
            background=background,
            lighting=composition.lighting,
            composition=composition.composition_notes
        )

        # Store in memory
        self.scene_memory.append({
            "background": background,
            "pose": composition.primary_pose
        })

        # Keep memory limited
        if len(self.scene_memory) > 5:
            self.scene_memory.pop(0)

        return prompt

    def _determine_composition(self, scene_type: str, content: str,
                              emphasis: str) -> SceneComposition:
        """Determine optimal scene composition"""

        # Scene type mappings
        scene_configs = {
            "intro": {
                "figures": 1,
                "pose": FigurePose.WELCOMING,
                "background": BackgroundGradient.BLUE_PURPLE,
                "lighting": "soft radial glow behind figure",
                "icons": []
            },
            "hook": {
                "figures": 1,
                "pose": FigurePose.POINTING_UP,
                "background": BackgroundGradient.MAGENTA_VIOLET,
                "lighting": "dramatic centered spotlight effect",
                "icons": [IconType.QUESTION]
            },
            "problem": {
                "figures": 1,
                "pose": FigurePose.THINKING,
                "background": BackgroundGradient.VIOLET_BLUE,
                "lighting": "subtle ambient lighting",
                "icons": [IconType.QUESTION]
            },
            "explanation": {
                "figures": 1,
                "pose": FigurePose.EXPLAINING,
                "background": BackgroundGradient.TEAL_NAVY,
                "lighting": "even professional lighting",
                "icons": self._select_explanation_icons(content)
            },
            "example": {
                "figures": 1,
                "pose": FigurePose.POINTING_RIGHT,
                "background": BackgroundGradient.CORAL_AMBER,
                "lighting": "warm centered glow",
                "icons": self._select_example_icons(content)
            },
            "data": {
                "figures": 1,
                "pose": FigurePose.PRESENTING,
                "background": BackgroundGradient.SLATE_INDIGO,
                "lighting": "focused lighting on figure and chart",
                "icons": [IconType.CHART_BAR]
            },
            "realization": {
                "figures": 1,
                "pose": FigurePose.REALIZATION,
                "background": BackgroundGradient.GOLDEN_ORANGE,
                "lighting": "bright radial burst behind figure",
                "icons": [IconType.LIGHTBULB]
            },
            "success": {
                "figures": 1,
                "pose": FigurePose.SUCCESS,
                "background": BackgroundGradient.GOLDEN_ORANGE,
                "lighting": "celebratory bright ambient glow",
                "icons": [IconType.CHECKMARK]
            },
            "teaching": {
                "figures": 2,
                "pose": FigurePose.TEACHING,
                "secondary_pose": FigurePose.LISTENING,
                "background": BackgroundGradient.GREEN_TEAL,
                "lighting": "soft even lighting across both figures",
                "icons": []
            },
            "conclusion": {
                "figures": 1,
                "pose": FigurePose.WELCOMING,
                "background": BackgroundGradient.GREEN_TEAL,
                "lighting": "warm concluding glow",
                "icons": [IconType.CHECKMARK]
            }
        }

        # Get base configuration
        config = scene_configs.get(scene_type, scene_configs["explanation"])

        # Build composition
        composition = SceneComposition(
            figures=config["figures"],
            primary_pose=config["pose"],
            secondary_pose=config.get("secondary_pose"),
            background=config["background"],
            icons=config.get("icons", []),
            lighting=config["lighting"],
            composition_notes=self._get_composition_notes(config["figures"],
                                                         len(config.get("icons", [])))
        )

        return composition

    def _select_explanation_icons(self, content: str) -> List[IconType]:
        """Select appropriate icons for explanation scenes"""

        icons = []
        content_lower = content.lower()

        if "money" in content_lower or "dollar" in content_lower or "cost" in content_lower:
            icons.append(IconType.DOLLAR)
        elif "idea" in content_lower or "insight" in content_lower:
            icons.append(IconType.LIGHTBULB)
        elif "data" in content_lower or "analytics" in content_lower:
            icons.append(IconType.CHART_BAR)
        elif "growth" in content_lower or "increase" in content_lower:
            icons.append(IconType.ARROW_UP)
        elif "process" in content_lower or "system" in content_lower:
            icons.append(IconType.GEAR)
        elif "learn" in content_lower or "education" in content_lower:
            icons.append(IconType.BOOK)

        # Limit to 1-2 icons for clean composition
        return icons[:2]

    def _select_example_icons(self, content: str) -> List[IconType]:
        """Select icons for example scenes"""

        content_lower = content.lower()

        if "target" in content_lower or "goal" in content_lower:
            return [IconType.TARGET]
        elif "time" in content_lower or "schedule" in content_lower:
            return [IconType.CLOCK]
        elif "success" in content_lower:
            return [IconType.CHECKMARK]
        else:
            return []

    def _format_icons(self, icons: List[IconType]) -> str:
        """Format icon descriptions for prompt"""

        if not icons:
            return ""

        if len(icons) == 1:
            return f"{icons[0].value} floating nearby, "
        else:
            icon_desc = ", ".join([icon.value for icon in icons])
            return f"{icon_desc} arranged thoughtfully around figure, "

    def _combine_poses(self, primary: FigurePose,
                      secondary: Optional[FigurePose]) -> str:
        """Combine poses for multi-figure scenes"""

        if not secondary:
            return f"both figures in {primary.value}"

        return f"left figure in {primary.value}, right figure in {secondary.value}"

    def _get_composition_notes(self, figures: int, icons: int) -> str:
        """Generate composition notes based on elements"""

        if figures == 2:
            return "balanced composition with comfortable spacing between figures, generous negative space"
        elif icons > 0:
            return "figure positioned left-of-center with icons on right, maintaining visual balance"
        else:
            return "centered figure with generous negative space, uncluttered professional composition"

    def _get_alternative_background(self, current: str) -> str:
        """Get alternative background to avoid repetition"""

        alternatives = {
            "blue-to-purple": "teal-to-navy",
            "teal-to-navy": "slate-to-indigo",
            "warm coral-to-amber": "golden-to-orange",
            "violet-to-blue": "magenta-to-violet",
            "golden-to-orange": "warm coral-to-amber",
            "green-to-teal": "cyan-to-deep-blue",
            "magenta-to-violet": "violet-to-blue",
            "slate-to-indigo": "blue-to-purple"
        }

        return alternatives.get(current, "blue-to-purple")


class StickFigurePromptLibrary:
    """Pre-built prompts for common scenarios - exactly matching requirements"""

    PROMPTS = {
        "narrator_intro": """Minimal vector illustration, single white stick figure with thick black outline, perfectly round head with two small dot eyes, confident standing pose with one arm extended palm-up in explaining gesture, blue-to-purple gradient background with soft radial glow behind figure, generous negative space, clean professional educational style, no text, 16:9 aspect ratio""",

        "explaining_concept": """Minimal vector illustration, white stick figure with thick consistent black outline, round head with dot eyes, standing with both arms raised in open explanatory gesture, three simple black-outline icons floating at arm level, teal-to-navy gradient background with subtle centered lighting, clean flat vector style, uncluttered composition, no text, 16:9 aspect ratio""",

        "two_figures_discussion": """Minimal vector illustration, two white stick figures with matching thick black outlines and round heads with dot eyes, left figure in teaching pose pointing upward, right figure with hand on chin in thoughtful stance, warm coral-to-amber gradient background, soft even lighting, figures positioned with comfortable spacing, simple professional style, no text, 16:9 aspect ratio""",

        "aha_moment": """Minimal vector illustration, single white stick figure with thick black outline, round head with dot eyes, dynamic pose with arms spread wide and body leaning forward excitedly, simple black-outline lightbulb icon glowing softly above head, purple-to-magenta gradient background with radial lighting centered on figure, energetic but clean composition, no text, 16:9 aspect ratio""",

        "working_focused": """Minimal vector illustration, white stick figure with thick black outline seated at simple line-drawn desk with minimal rectangular laptop shape, focused forward-leaning pose suggesting concentration, deep blue-to-teal gradient background, calm professional lighting, plenty of negative space, no clutter, no text, 16:9 aspect ratio""",

        "presenting_data": """Minimal vector illustration, white stick figure with thick black outline and round head with dot eyes, standing in profile view pointing confidently at simple floating bar chart icon drawn in matching black outline style, green-to-teal gradient background, clean educational aesthetic, balanced composition with figure on left third and chart on right, no text, 16:9 aspect ratio""",

        "pondering": """Minimal vector illustration, single white stick figure with thick black outline, round head with dot eyes, contemplative pose with one hand raised to chin and slight head tilt, small simple question mark icon floating nearby, soft violet-to-blue gradient background with gentle ambient lighting, introspective but approachable mood, no text, 16:9 aspect ratio""",

        "celebration": """Minimal vector illustration, white stick figure with thick black outline, round head with dot eyes, triumphant pose with both arms raised overhead in victory gesture, simple checkmark icon floating above, warm golden-to-orange gradient background with bright centered glow, positive energetic composition, clean vector style, no text, 16:9 aspect ratio"""
    }

    @classmethod
    def get_prompt(cls, scene_name: str) -> str:
        """Get a pre-built prompt by name"""
        return cls.PROMPTS.get(scene_name, cls.PROMPTS["explaining_concept"])