"""
Wondershare API Service - Access Veo3, Sora2 via your paid Filmora subscription

This service uses your legitimate Filmora subscription to access:
- Google Veo 3.0/3.1 (text-to-video with auto-audio)
- OpenAI Sora 2 (image-to-video)
- Kelin model (cost-effective text-to-video)

Authentication uses credentials from your paid account.
"""

import requests
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class AIModel(str, Enum):
    """Available AI models via Wondershare."""
    VEO3 = "veo-3.0-fast-generate-preview"
    VEO31 = "veo-3.1-fast-generate-preview"
    SORA2 = "sora-2"
    KELIN = "video_model"


class TaskStatus(str, Enum):
    """AI task processing statuses."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class WondershareAPIService:
    """
    Service to interact with Wondershare's AI video generation API.

    Uses your paid Filmora subscription credentials.
    """

    def __init__(
        self,
        client_sign: str,
        wsid: str,
        session_token: Optional[str] = None
    ):
        """
        Initialize Wondershare API service.

        Args:
            client_sign: Client signature from Filmora
            wsid: Wondershare user ID from your account
            session_token: Optional session/bearer token
        """
        self.base_url = "https://prod-web.wondershare.cc/api/v1"
        self.client_sign = client_sign
        self.wsid = wsid
        self.session_token = session_token

        # Filmora product info
        self.pid = "1901"  # Filmora product ID
        self.pver = "15.0.12.16430"  # Product version
        self.pname = "filmora"

    def _get_headers(self, content_type: str = "application/json") -> Dict[str, str]:
        """Get standard headers for API requests."""
        headers = {
            "User-Agent": f"Filmora/{self.pver}",
            "Content-Type": content_type,
            "X-Client-Sign": self.client_sign,
            "X-PID": self.pid,
            "X-PVER": self.pver,
        }

        if self.wsid:
            headers["X-WSID"] = self.wsid

        if self.session_token:
            headers["Authorization"] = f"Bearer {self.session_token}"

        return headers

    def text_to_video(
        self,
        prompt: str,
        model: AIModel = AIModel.VEO3,
        duration: int = 8,
        resolution: str = "720p",
        aspect_ratio: str = "16:9"
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt using Veo3 or Kelin.

        Args:
            prompt: Text description of the video (1-1500 chars)
            model: AI model to use (VEO3, VEO31, or KELIN)
            duration: Video duration in seconds (fixed 8 for Veo3, 5 for Kelin)
            resolution: "720p" or "1080p"
            aspect_ratio: "16:9" or "9:16"

        Returns:
            Response dict with task_id and status
        """
        endpoint = f"{self.base_url}/aigc/text_to_video"

        payload = {
            "prompt": prompt,
            "model": model.value,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "workflow_id": "46" if model == AIModel.VEO3 else "1804422399892127744"
        }

        logger.info(f"Requesting text-to-video: {prompt[:50]}... (model: {model.value})")

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Task created: {result.get('task_id', 'unknown')}")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Text-to-video request failed: {e}")
            raise

    def image_to_video(
        self,
        image_path: Path,
        prompt: str,
        model: AIModel = AIModel.SORA2,
        duration: int = 8
    ) -> Dict[str, Any]:
        """
        Generate video from image using Sora2 or Veo3.1.

        Args:
            image_path: Path to input image
            prompt: Motion/animation description
            model: AI model (SORA2 or VEO31)
            duration: Fixed 8 seconds

        Returns:
            Response dict with task_id and status
        """
        endpoint = f"{self.base_url}/aigc/image_to_video"

        # Upload image first (may need separate endpoint)
        # For now, assume image is uploaded separately

        payload = {
            "init_image": str(image_path),  # or upload URL
            "prompt": prompt,
            "model": model.value,
            "duration": duration,
            "workflow_id": "51" if model == AIModel.SORA2 else "45"
        }

        logger.info(f"Requesting image-to-video (model: {model.value})")

        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers=self._get_headers(),
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"Task created: {result.get('task_id', 'unknown')}")
            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"Image-to-video request failed: {e}")
            raise

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Poll for task completion status.

        Args:
            task_id: Task ID from generation request

        Returns:
            Status dict with task state and video URL if completed
        """
        endpoint = f"{self.base_url}/aigc/task_status/{task_id}"

        try:
            response = requests.get(
                endpoint,
                headers=self._get_headers(),
                timeout=10
            )
            response.raise_for_status()

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Status check failed: {e}")
            raise

    def wait_for_completion(
        self,
        task_id: str,
        timeout: int = 600,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """
        Wait for task to complete with polling.

        Args:
            task_id: Task ID to wait for
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between status checks

        Returns:
            Final task status with video URL

        Raises:
            TimeoutError: If task doesn't complete in time
        """
        start_time = time.time()

        logger.info(f"Waiting for task {task_id} to complete...")

        while time.time() - start_time < timeout:
            status = self.get_task_status(task_id)

            task_status = status.get("status", "unknown")
            logger.debug(f"Task status: {task_status}")

            if task_status == TaskStatus.COMPLETED:
                logger.info(f"Task {task_id} completed!")
                return status

            elif task_status == TaskStatus.FAILED:
                error_msg = status.get("error", "Unknown error")
                logger.error(f"Task failed: {error_msg}")
                raise RuntimeError(f"Task failed: {error_msg}")

            time.sleep(poll_interval)

        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")

    def download_video(self, video_url: str, output_path: Path) -> Path:
        """
        Download generated video.

        Args:
            video_url: URL from completed task
            output_path: Where to save video

        Returns:
            Path to downloaded video
        """
        logger.info(f"Downloading video to {output_path}")

        try:
            response = requests.get(video_url, stream=True, timeout=60)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Video downloaded: {output_path}")
            return output_path

        except requests.exceptions.RequestException as e:
            logger.error(f"Download failed: {e}")
            raise


# Example usage
if __name__ == "__main__":
    # Configure with your Filmora subscription credentials
    # These will be extracted from monitoring/database

    # Example (REPLACE WITH YOUR ACTUAL CREDENTIALS):
    api = WondershareAPIService(
        client_sign="REDACTED_WONDERSHARE_SIGN",  # Found in logs
        wsid="YOUR_WSID_HERE",  # From monitoring or database
        session_token="YOUR_TOKEN_HERE"  # From monitoring (optional)
    )

    # Generate video from text using Veo3
    print("Requesting Veo3 video generation...")
    result = api.text_to_video(
        prompt="A cat eating crispy fried chicken with ASMR sounds",
        model=AIModel.VEO3,
        duration=8,
        resolution="720p"
    )

    task_id = result.get("task_id")
    print(f"Task ID: {task_id}")

    # Wait for completion
    print("Waiting for video generation...")
    final_status = api.wait_for_completion(task_id, timeout=600)

    # Download video
    video_url = final_status.get("video_url")
    if video_url:
        output_path = Path("/mnt/e/wondershare/engineer/output/test_video.mp4")
        api.download_video(video_url, output_path)
        print(f"✅ Video saved to: {output_path}")
    else:
        print("❌ No video URL in response")
        print(final_status)
