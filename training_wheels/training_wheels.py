# training_wheels.py - Complete enhanced implementation with improvements

import os
import json
import uuid
import time
import hashlib
import threading
import logging
import requests
import sys
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
import pyautogui
import boto3
from PIL import Image, ImageDraw, ImageFont
import imagehash
import tempfile
import base64
from io import BytesIO
from dotenv import load_dotenv
import dotenv

# Load environment variables from .env file if present
load_dotenv()
dotenv.load_dotenv()  # Load environment variables from .env file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Configuration
class Config:
    # S3 configuration
    S3_ENABLED = True  # Enable S3 uploads
    S3_BUCKET_NAME = "learnchain-training-wheels"  # Use env var or fallback
    S3_FOLDER = "sessions"

    # AWS credentials from environment variables with fallbacks
    AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "AKIAYKFQQ4HCH64UYOQF")
    AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "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")
    AWS_DEFAULT_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

    # OpenAI API - get key from environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")

    # AWS Polly - Text-to-Speech
    POLLY_ENABLED = os.environ.get("POLLY_ENABLED", "false").lower() == "true"
    POLLY_VOICE_ID = os.environ.get("POLLY_VOICE_ID", "Ruth")  # Default voice

    # Screenshot settings
    SCREENSHOT_INTERVAL = float(os.environ.get("SCREENSHOT_INTERVAL", "0.1"))  # Seconds between screenshot checks
    CHANGE_THRESHOLD = int(os.environ.get("CHANGE_THRESHOLD", "8"))  # Perceptual hash difference threshold
    MAX_CONSECUTIVE_NO_CHANGES = int(os.environ.get("MAX_CONSECUTIVE_NO_CHANGES", "10"))  # Max checks with no change

    # Screenshot optimization
    MAX_SCREENSHOT_WIDTH = int(os.environ.get("MAX_SCREENSHOT_WIDTH", "1200"))  # Maximum width for optimization
    SCREENSHOT_QUALITY = int(os.environ.get("SCREENSHOT_QUALITY", "75"))  # JPEG quality for compression

    # Session timeout (in minutes)
    SESSION_TIMEOUT_MINUTES = int(os.environ.get("SESSION_TIMEOUT_MINUTES", "30"))

    # Explanation mode - default to disabled
    DEFAULT_EXPLANATION_MODE = os.environ.get("DEFAULT_EXPLANATION_MODE", "false").lower() == "true"

    # Storage paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    STORAGE_DIR = os.path.join(BASE_DIR, "training_data")
    AUDIO_DIR = os.path.join(STORAGE_DIR, "audio")

    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist"""
        os.makedirs(cls.STORAGE_DIR, exist_ok=True)
        os.makedirs(cls.AUDIO_DIR, exist_ok=True)

    @classmethod
    def get_session_dir(cls, session_id: str) -> str:
        """Get directory for a specific session"""
        session_dir = os.path.join(cls.STORAGE_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    @classmethod
    def validate(cls):
        """Validate critical configuration settings"""
        if cls.S3_ENABLED and (not cls.AWS_ACCESS_KEY_ID or not cls.AWS_SECRET_ACCESS_KEY):
            logger.warning("S3 is enabled but AWS credentials are missing. Check your environment variables.")
            return False

        if not cls.OPENAI_API_KEY:
            logger.warning("OpenAI API key is missing. Check your environment variables.")
            return False

        return True


# Audio Manager for Text-to-Speech
class AudioManager:
    """Manages text-to-speech functionality using AWS Polly"""

    def __init__(self, s3_manager=None):
        self.enabled = Config.POLLY_ENABLED
        self.polly_client = None
        self.s3_manager = s3_manager

        # Try to initialize Polly client if enabled
        if self.enabled:
            try:
                self.polly_client = boto3.client(
                    'polly',
                    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                    region_name=Config.AWS_DEFAULT_REGION
                )
                # Test connection
                self.polly_client.describe_voices(LanguageCode='en-US')
                logger.info("AWS Polly connection successful")
            except Exception as e:
                logger.error(f"AWS Polly initialization failed: {e}")
                self.enabled = False

    def synthesize_speech(self, text: str, session_id: str, step_index: int) -> Optional[str]:
        """Synthesize speech from text and return the file path"""
        if not self.enabled or not self.polly_client:
            return None

        try:
            # Clean text for speech synthesis (remove formatting, etc.)
            clean_text = self._clean_text_for_speech(text)

            # Prepare output path
            session_dir = Config.get_session_dir(session_id)
            audio_dir = os.path.join(session_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)

            filename = f"step_{step_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
            file_path = os.path.join(audio_dir, filename)

            # Call Polly to synthesize speech
            response = self.polly_client.synthesize_speech(
                Text=clean_text,
                OutputFormat='mp3',
                VoiceId=Config.POLLY_VOICE_ID,
                Engine='neural'  # Use neural engine for better quality
            )

            # Save the audio to a file
            with open(file_path, 'wb') as f:
                f.write(response['AudioStream'].read())

            logger.info(f"Text-to-speech audio saved to {file_path}")

            # Also upload to S3 if enabled
            if self.s3_manager and self.s3_manager.enabled:
                url = self.s3_manager.upload_file(
                    file_path,
                    session_id,
                    "audio",
                    filename
                )
                logger.info(f"Audio uploaded to S3: {url}")

            return file_path

        except Exception as e:
            logger.error(f"Text-to-speech synthesis failed: {e}")
            return None

    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis"""
        # Remove code blocks, formatting symbols, etc.
        # This is a simple implementation - could be expanded
        import re

        # Replace code blocks with placeholder
        text = re.sub(r'```.*?```', 'Here is some code.', text, flags=re.DOTALL)

        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'__(.*)__', r'\1', text)  # Underline
        text = re.sub(r'_(.*)_', r'\1', text)  # Italic

        # Replace bullet points with pauses
        text = re.sub(r'^\s*[-*]\s*', 'Bullet point: ', text, flags=re.MULTILINE)

        # Replace numbered lists
        text = re.sub(r'^\s*(\d+)\.', r'Step \1:', text, flags=re.MULTILINE)

        # Add pauses for readability
        text = text.replace(".", ". ")
        text = text.replace("!", "! ")
        text = text.replace("?", "? ")

        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# S3 Manager
class S3Manager:
    """Manages S3 operations with graceful fallback to local storage"""

    def __init__(self):
        self.enabled = Config.S3_ENABLED
        self.s3_client = None

        # Try to initialize S3 client if enabled
        if self.enabled:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                    aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                    region_name=Config.AWS_DEFAULT_REGION
                )
                # Test connection
                self.s3_client.list_buckets()
                logger.info("S3 connection successful")

                # Test bucket
                try:
                    self.s3_client.head_bucket(Bucket=Config.S3_BUCKET_NAME)
                    logger.info(f"S3 bucket '{Config.S3_BUCKET_NAME}' exists")
                except Exception as e:
                    logger.warning(f"S3 bucket test failed: {e}")
                    self.enabled = False
            except Exception as e:
                logger.error(f"S3 initialization failed: {e}")
                self.enabled = False

    def upload_file(self, local_path: str, session_id: str, file_type: str, filename: str) -> str:
        """Upload file to S3 and return URL with better file existence checks"""
        if not self.enabled or not self.s3_client:
            # Force S3 to be enabled in Config if it wasn't
            if not Config.S3_ENABLED:
                logger.warning("S3 was disabled in Config, but is required. Enabling.")
                Config.S3_ENABLED = True
                # Re-initialize the client
                self.__init__()

                # If still not enabled after re-init, this is a critical error
                if not self.enabled or not self.s3_client:
                    logger.error("Critical error: S3 is required but could not be enabled")
                    return local_path  # Return local path as fallback

        # Check if file exists before attempting upload
        if not os.path.exists(local_path):
            logger.error(f"File does not exist: {local_path}")

            # Generate a placeholder error image if this is a screenshot
            if file_type == "screenshots":
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    # Create a simple error image
                    error_img = Image.new('RGB', (800, 600), color=(255, 255, 255))
                    draw = ImageDraw.Draw(error_img)
                    draw.text((50, 50), f"Error: Screenshot file not found\nTimestamp: {datetime.now()}",
                              fill=(255, 0, 0))
                    error_img.save(local_path)
                    logger.info(f"Created placeholder error image: {local_path}")
                except Exception as e:
                    logger.error(f"Failed to create placeholder image: {e}")
                    return local_path  # Return local path as fallback

        try:
            key = f"{Config.S3_FOLDER}/{session_id}/{file_type}/{filename}"

            # Upload without ACL since the bucket doesn't support them
            self.s3_client.upload_file(
                local_path,
                Config.S3_BUCKET_NAME,
                key
            )

            url = f"https://{Config.S3_BUCKET_NAME}.s3.amazonaws.com/{key}"
            logger.info(f"Uploaded to S3: {url}")
            return url
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            # Try one more time with a short delay
            time.sleep(1.0)
            try:
                key = f"{Config.S3_FOLDER}/{session_id}/{file_type}/{filename}"
                self.s3_client.upload_file(
                    local_path,
                    Config.S3_BUCKET_NAME,
                    key
                )
                url = f"https://{Config.S3_BUCKET_NAME}.s3.amazonaws.com/{key}"
                logger.info(f"Retry successful - uploaded to S3: {url}")
                return url
            except Exception as e:
                # Log the error but return the local path as fallback
                logger.error(f"S3 upload retry failed: {e}")
                return local_path


# Screenshot Manager
class ScreenshotManager:
    """Manages screen capture, change detection, and image processing"""

    def __init__(self, s3_manager: S3Manager):
        self.region: Optional[Tuple[int, int, int, int]] = None
        self.last_image_hash = None
        self.s3_manager = s3_manager
        self.no_change_counter = 0
        self.previous_screenshots = []  # Store recent screenshots for comparison

    def set_region(self, region: Tuple[int, int, int, int]) -> bool:
        """Set the screen region to capture and take test screenshot"""
        # Ensure region is valid (all values > 0)
        if not all(val > 0 for val in region[2:]):
            logger.error(f"Invalid region dimensions: {region}")
            return False

        self.region = region
        logger.info(f"Screen capture region set to {region}")

        # Take test screenshot
        success, test_path = self.take_test_screenshot()
        return success

    def take_test_screenshot(self) -> Tuple[bool, Optional[str]]:
        """Take a test screenshot to verify the region works"""
        if not self.region:
            logger.error("Cannot take test screenshot: No region set")
            return False, None

        try:
            # Create test directory
            test_dir = os.path.join(Config.STORAGE_DIR, "test_screenshots")
            os.makedirs(test_dir, exist_ok=True)
            test_filename = f"test_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            test_path = os.path.join(test_dir, test_filename)

            # First try with PyAutoGUI
            try:
                screenshot = pyautogui.screenshot(region=self.region)
                # Optimize the screenshot
                screenshot = self._optimize_image(screenshot)
                screenshot.save(test_path)
                logger.info(f"Test screenshot successful with PyAutoGUI: {test_path}")
            except Exception as e:
                logger.warning(f"PyAutoGUI screenshot failed: {e}, trying alternative methods")

                # Try using system tools on Linux
                if sys.platform.startswith('linux'):
                    try:
                        # Try using 'import' command from ImageMagick
                        import subprocess
                        x, y, width, height = self.region
                        subprocess.run([
                            'import',
                            '-window', 'root',
                            '-crop', f'{width}x{height}+{x}+{y}',
                            test_path
                        ], check=True)
                        logger.info(f"Test screenshot successful with ImageMagick: {test_path}")

                        # Optimize the saved image
                        try:
                            img = Image.open(test_path)
                            img = self._optimize_image(img)
                            img.save(test_path)
                        except Exception as opt_error:
                            logger.warning(f"Failed to optimize ImageMagick screenshot: {opt_error}")

                    except Exception as img_error:
                        logger.warning(f"ImageMagick screenshot failed: {img_error}, trying gnome-screenshot")

                        # Try gnome-screenshot as a last resort
                        try:
                            x, y, width, height = self.region
                            subprocess.run([
                                'gnome-screenshot',
                                '--area',
                                f'{x},{y},{width},{height}',
                                '--file',
                                test_path
                            ], check=True)
                            logger.info(f"Test screenshot successful with gnome-screenshot: {test_path}")

                            # Optimize the saved image
                            try:
                                img = Image.open(test_path)
                                img = self._optimize_image(img)
                                img.save(test_path)
                            except Exception as opt_error:
                                logger.warning(f"Failed to optimize gnome-screenshot: {opt_error}")

                        except Exception as gnome_error:
                            logger.error(f"All screenshot methods failed: {gnome_error}")
                            return False, None
                else:
                    # If not on Linux and PyAutoGUI failed, we fail
                    logger.error(f"Screenshot failed and no alternatives available on this platform")
                    return False, None

            # Upload to S3
            self.s3_manager.upload_file(
                test_path,
                "test",
                "screenshots",
                test_filename
            )

            return True, test_path
        except Exception as e:
            logger.error(f"Test screenshot failed: {e}")
            return False, None

    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image size and quality for faster uploads and processing"""
        try:
            # Check if resizing is needed
            if image.width > Config.MAX_SCREENSHOT_WIDTH:
                # Calculate new height maintaining aspect ratio
                ratio = Config.MAX_SCREENSHOT_WIDTH / image.width
                new_height = int(image.height * ratio)

                # Resize the image
                image = image.resize((Config.MAX_SCREENSHOT_WIDTH, new_height), Image.LANCZOS)
                logger.debug(f"Resized image to {Config.MAX_SCREENSHOT_WIDTH}x{new_height}")

            # Convert to RGB if needed (in case of RGBA)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image
        except Exception as e:
            logger.error(f"Error optimizing image: {e}")
            return image  # Return original image on error

    def capture_screenshot(self, session_id: str, step_index: int, force: bool = False) -> Optional[str]:
        """Capture screenshot and return the path if it represents a change"""
        if not self.region:
            logger.error("Cannot capture screenshot: No region set")
            return None

        try:
            # Prepare the path
            session_dir = Config.get_session_dir(session_id)
            screenshots_dir = os.path.join(session_dir, "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)

            filename = f"step_{step_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"  # Use JPG for better compression
            file_path = os.path.join(screenshots_dir, filename)

            # Take the screenshot
            try:
                screenshot = pyautogui.screenshot(region=self.region)

                # Optimize the screenshot
                screenshot = self._optimize_image(screenshot)

                # Save with compression
                screenshot.save(file_path, "JPEG", quality=Config.SCREENSHOT_QUALITY)

                # Verify the file was saved correctly
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    logger.error(f"Screenshot file missing or empty after capture: {file_path}")
                    return None

                logger.debug(f"Screenshot saved to {file_path}")
            except Exception as e:
                logger.error(f"Failed to capture screenshot: {e}")
                return None

            # Use a file lock to prevent deletion during upload
            lock_path = file_path + ".lock"
            with open(lock_path, 'w') as lock_file:
                lock_file.write("1")

            # Check if it represents a change or if we should force capture
            if force:
                self.no_change_counter = 0
                self.last_image_hash = imagehash.phash(screenshot)
                logger.info(f"Forced screenshot captured: {file_path}")

                # Add to previous screenshots for context
                self.previous_screenshots.append(file_path)
                if len(self.previous_screenshots) > 5:  # Keep only the 5 most recent
                    self.previous_screenshots.pop(0)

                # Upload to S3 - this happens synchronously
                url = self.s3_manager.upload_file(
                    file_path,
                    session_id,
                    "screenshots",
                    filename
                )

                # Delete the lock file after upload
                if os.path.exists(lock_path):
                    os.remove(lock_path)

                return file_path

            # Check for significant change
            is_change = self._is_significant_change(screenshot)

            if is_change:
                # Significant change detected
                self.no_change_counter = 0
                logger.info(f"Screenshot captured due to significant change: {file_path}")

                # Add to previous screenshots for context
                self.previous_screenshots.append(file_path)
                if len(self.previous_screenshots) > 5:  # Keep only the 5 most recent
                    self.previous_screenshots.pop(0)

                # Upload to S3
                url = self.s3_manager.upload_file(
                    file_path,
                    session_id,
                    "screenshots",
                    filename
                )

                # Delete the lock file after upload
                if os.path.exists(lock_path):
                    os.remove(lock_path)

                return file_path
            else:
                # No significant change
                self.no_change_counter += 1

                # Check if we've had too many consecutive no-changes
                if self.no_change_counter >= Config.MAX_CONSECUTIVE_NO_CHANGES:
                    self.no_change_counter = 0
                    logger.info(
                        f"Screenshot captured after {Config.MAX_CONSECUTIVE_NO_CHANGES} consecutive no-changes: {file_path}")

                    # Add to previous screenshots for context
                    self.previous_screenshots.append(file_path)
                    if len(self.previous_screenshots) > 5:  # Keep only the 5 most recent
                        self.previous_screenshots.pop(0)

                    # Upload to S3
                    url = self.s3_manager.upload_file(
                        file_path,
                        session_id,
                        "screenshots",
                        filename
                    )

                    # Delete the lock file after upload
                    if os.path.exists(lock_path):
                        os.remove(lock_path)

                    return file_path
                else:
                    # No change and not enough consecutive no-changes
                    # Don't delete the file if it has a lock
                    if not os.path.exists(lock_path):
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        logger.info(
                            f"No significant change detected ({self.no_change_counter}/{Config.MAX_CONSECUTIVE_NO_CHANGES}), screenshot discarded")
                    else:
                        logger.warning(f"File has lock, not deleting: {file_path}")
                        # Clean up the lock file
                        os.remove(lock_path)
                    return None

        except Exception as e:
            logger.error(f"Screenshot capture failed: {e}")
            return None

    def _is_significant_change(self, image: Image.Image) -> bool:
        """Check if the image represents a significant change"""
        try:
            current_hash = imagehash.phash(image)

            # First screenshot is always significant
            if self.last_image_hash is None:
                self.last_image_hash = current_hash
                return True

            # Calculate difference
            difference = self.last_image_hash - current_hash
            logger.debug(f"Image hash difference: {difference}")

            if difference > Config.CHANGE_THRESHOLD:
                self.last_image_hash = current_hash
                return True

            return False
        except Exception as e:
            logger.error(f"Error checking for significant change: {e}")
            return True  # Default to assuming change on error


# LLM Manager
class LLMManager:
    """Manages interactions with LLM for step guidance"""

    def __init__(self, s3_manager=None):
        self.api_key = Config.OPENAI_API_KEY
        self.model = Config.OPENAI_MODEL
        self.audio_manager = AudioManager(s3_manager)
        self.s3_manager = s3_manager

    def get_next_instruction(self,
                             session_id: str,
                             goal: str,
                             step_index: int,
                             screenshot_path: str,
                             knowledge: List[Dict[str, Any]],
                             previous_steps: List[Dict[str, Any]] = None,
                             include_explanations: bool = False,
                             chat_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get next instruction from LLM with robust error handling"""
        try:
            # Get screenshot URL - this will throw an exception if S3 upload fails
            screenshot_url = self.s3_manager.upload_file(
                screenshot_path,
                session_id,
                "screenshots",
                os.path.basename(screenshot_path)
            )

            # Build the system message
            system_message = self._build_system_message(step_index, include_explanations)

            # Build the user message text
            user_message_text = self._build_user_message_text(
                goal=goal,
                step_index=step_index,
                knowledge=knowledge,
                previous_steps=previous_steps or [],
                chat_history=chat_history or []
            )

            # Call OpenAI API with the S3 URL
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message_text},
                            {"type": "image_url", "image_url": {"url": screenshot_url}}
                        ]
                    }
                ],
                "response_format": {"type": "json_object"}
            }

            logger.info(f"Sending request to LLM with screenshot: {screenshot_url}")
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )

            if response.status_code != 200:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return {
                    "instruction": f"Error getting step {step_index} guidance. Please try again.",
                    "format": "text",
                    "error": True
                }

            # Parse response
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content")

            # Check if content is empty or None
            if not content:
                logger.error(f"Empty content from LLM: {result}")
                return {
                    "instruction": f"Error generating step {step_index}. Please try again.",
                    "format": "text",
                    "error": True
                }

            # Save the raw response
            session_dir = Config.get_session_dir(session_id)
            responses_dir = os.path.join(session_dir, "llm_responses")
            os.makedirs(responses_dir, exist_ok=True)

            response_filename = f"step_{step_index}_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            response_path = os.path.join(responses_dir, response_filename)

            with open(response_path, 'w') as f:
                json.dump({
                    "step_index": step_index,
                    "timestamp": datetime.now().isoformat(),
                    "request": {
                        "system_message": system_message,
                        "user_message": user_message_text,
                        "screenshot_url": screenshot_url
                    },
                    "response": result
                }, f, indent=2)

            # Upload to S3
            self.s3_manager.upload_file(response_path, session_id, "llm_responses", response_filename)

            # Parse the content as JSON
            try:
                guidance = json.loads(content)
                instruction_data = {
                    "instruction": guidance.get("instruction", f"Step {step_index}: No specific instruction provided"),
                    "format": guidance.get("format", "text"),
                    "details": guidance.get("details", {})
                }

                # Add explanation if present
                if "explanation" in guidance:
                    instruction_data["explanation"] = guidance.get("explanation", "")

                # Generate audio if enabled
                if Config.POLLY_ENABLED:
                    audio_text = instruction_data["instruction"]
                    if include_explanations and "explanation" in instruction_data:
                        audio_text += f" Here's why: {instruction_data['explanation']}"

                    audio_path = self.audio_manager.synthesize_speech(
                        audio_text,
                        session_id,
                        step_index
                    )

                    if audio_path:
                        instruction_data["audio_path"] = audio_path

                return instruction_data

            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse LLM response as JSON: {content} - Error: {e}")
                # Fallback to using raw text or a default message
                if isinstance(content, str) and content.strip():
                    return {
                        "instruction": f"Step {step_index}: {content}",
                        "format": "text"
                    }
                else:
                    return {
                        "instruction": f"Error generating step {step_index}. Please try again.",
                        "format": "text",
                        "error": True
                    }

        except Exception as e:
            logger.error(f"Error getting next instruction: {e}")
            return {
                "instruction": f"Error generating step {step_index}. Please try again.",
                "format": "text",
                "error": True
            }

    def _build_system_message(self, step_index: int, include_explanations: bool = False) -> str:
        """Build the system message for the LLM"""
        explanation_instruction = """
    When providing instructions, also include an 'explanation' field in your JSON response that explains the reasoning behind this step. This explanation should help the user understand WHY they are performing this action and how it relates to the overall goal."""

        base_message = f"""You are an expert assistant guiding users through completing tasks step by step.

    You are currently providing Step {step_index} of the process.

    CRITICAL: First, carefully analyze the screenshot to verify if the user has completed ALL previous steps before providing the next instruction.

    - Sound like a friendly instructor. If you feel like you're repeating a step or if you sense the user wandering in confusion, predict what could be the possible reasons and give them additional instructions.
    - For example, if the previous instruction was to open an application, in the current step if you see that the user has NOT YET opened the required application, your instruction could be to open that application.
    - If you see that previous steps haven't been completed, instruct/remind the user to complete those steps first if it might still matter. Sometimes, the user might be just faster than your instructions that they went faster than you could see. In those cases ask them to ensure and give the current instruction or ask them to go back to the respective screen and show you that previous step.
    - If the knowledge (context) doesn't have an info, feel free to use your common-sense and general outside knowledge. Don't have a hard reliance on the instructions/knowledge/provided context. You are allowed to be intuitive.
    - Sometimes, the context/knowledge might miss a step/even have unwanted steps. Handle these.
    - sometimes, the button name/screen/app name might be different than what you see on screen, at these times, do an intuitive verification/check. 
    - Try to visually confirm if the user has completed a previous step, you are allowed to ask them to go back and show you the completion. 
    - Give explanatory instructions that can enable the user to learn/understand why and what they are doing. 

    Analyze the screenshot carefully and provide clear, specific instructions for what the user should do next.

    Important: You cannot provide instruction for a step the user is not yet at. Always prioritize the current step based on what you see in the screen.

    {explanation_instruction if include_explanations else ""}

    Your response must be a JSON object with these fields:
    - instruction: A list of instruction blocks.
    - explanation: A list of explanation blocks{" (required)" if include_explanations else " (optional)"}

    Each block should contain:
    - type: One of "text", "link", or "code"
    - content: The text, code, or URL
    - label (optional): For "link" blocks only, a user-friendly label

    Be specific about what to click, type, select, etc.

    Example response:
    {{
      "instruction": [
        {{
          "type": "text",
          "content": "Open the AWS EC2 Console in your browser."
        }},
        {{
          "type": "link",
          "label": "EC2 Console",
          "content": "https://console.aws.amazon.com/ec2/"
        }}
      ],
      "explanation": [
        {{
          "type": "text",
          "content": "The EC2 Console is where you will launch and configure your virtual server."
        }}
      ]
    }}

    Keep your instructions focused on ONE specific action at a time. Always base your guidance on what you can see in the screenshot."""

        return base_message

    def _build_user_message_text(self,
                                 goal: str,
                                 step_index: int,
                                 knowledge: List[Dict[str, Any]],
                                 previous_steps: List[Dict[str, Any]],
                                 chat_history: List[Dict[str, Any]] = None) -> str:
        """Build the user message text for the LLM with improved context"""
        # Format previous steps - Show MORE of the previous steps for better context
        steps_text = ""
        if previous_steps:
            steps_text = "Previous steps:\n"
            for i, step in enumerate(previous_steps[-10:]):  # Show last 10 steps for better context
                # Get the step index correctly
                displayed_index = step.get('step_index', i)
                instruction = step.get('instruction', 'Unknown step')
                timestamp = step.get('timestamp', '')

                # Convert timestamp to a readable format if it's a string
                if isinstance(timestamp, str):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        friendly_time = dt.strftime("%H:%M:%S")
                    except:
                        friendly_time = ""
                else:
                    friendly_time = ""

                steps_text += f"{displayed_index}. [{friendly_time}] {instruction}\n"

        # Format knowledge - include more context from knowledge items
        knowledge_text = ""
        if knowledge:
            knowledge_text = "Relevant knowledge:\n"
            for i, item in enumerate(knowledge[:5]):  # Show top 5 knowledge items
                content = item.get('content', '')
                source = item.get('type', 'general')
                knowledge_text += f"{i + 1}. [{source}] {content[:500]}\n\n"

        # Format chat history if available
        chat_text = ""
        if chat_history and len(chat_history) > 0:
            chat_text = "Recent chat interactions:\n"
            # Only include the 5 most recent chat interactions
            recent_chats = chat_history[-5:] if len(chat_history) > 5 else chat_history
            for i, chat in enumerate(recent_chats):
                role = chat.get('role', 'unknown')
                content = chat.get('content', '')
                chat_text += f"{role}: {content[:200]}...\n"
            chat_text += "\n"

        # Build complete message text
        return f"""GOAL: {goal}

CURRENT STEP: Step {step_index}

IMPORTANT: First verify if the respective application or browser tab to do the task is open before proceeding to any other instruction. If not open, instruct the user to open it.

{steps_text}

{knowledge_text}

{chat_text}

Based on the screenshot and the goal, what specific action should the user take next?
Provide your answer as a JSON object following the format in my instructions."""


# Session state management
class SessionState:
    """Manages the state of a training session"""

    def __init__(self, session_id: str, user_id: str, course_id: str, goal: str, include_explanations: bool = False, s3_manager=None):
        self.session_id = session_id
        self.user_id = user_id
        self.course_id = course_id
        self.goal = goal
        self.step_index = 0
        self.user_step_index = 1  # Tracks user-visible step numbers starting at 1
        self.steps = []
        self.knowledge = []
        self.chat_history = []  # Added to store chat interactions
        self.status = "initializing"  # initializing, in_progress, paused, completed
        self.start_time = datetime.now().isoformat()
        self.screenshots = []
        self.next_step_requested = False  # Flag for user-requested next step
        self.include_explanations = include_explanations  # Whether to include explanations
        self.audio_enabled = Config.POLLY_ENABLED  # Whether to generate audio for steps
        self.s3_manager = s3_manager

        # Create session directory
        self.session_dir = Config.get_session_dir(session_id)
        self.state_file = os.path.join(self.session_dir, "session_state.json")

        # Initialize state file
        self._save_state()
        logger.info(f"Session initialized: {session_id}")

    def _save_state(self):
        """Save the current state to disk"""
        state = {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "course_id": self.course_id,
            "goal": self.goal,
            "step_index": self.step_index,
            "steps": self.steps,
            "status": self.status,
            "start_time": self.start_time,
            "knowledge_count": len(self.knowledge),
            "screenshots": self.screenshots,
            "include_explanations": self.include_explanations,
            "audio_enabled": self.audio_enabled,
            "chat_history_count": len(self.chat_history)
        }

        if hasattr(self, 'end_time'):
            state["end_time"] = self.end_time

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            # Also save to S3
            if self.s3_manager:
                self.s3_manager.upload_file(self.state_file, self.session_id, "state", "session_state.json")


        except Exception as e:
            logger.error(f"Error saving session state: {e}")

    def add_step(self, instruction: Dict[str, Any], screenshot_path: str = None):
        """Add a step to the session"""
        is_system_step = instruction.get("system_step", False)

        step = {
            "step_index": self.step_index,
            "timestamp": datetime.now().isoformat(),
            "instruction": instruction.get("instruction", ""),
            "format": instruction.get("format", "text"),
            "details": instruction.get("details", {}),
            "system_step": is_system_step
        }

        # Include explanation if present and explanations are enabled
        if self.include_explanations and "explanation" in instruction:
            step["explanation"] = instruction.get("explanation", "")

        # Include audio path if present
        if "audio_path" in instruction:
            step["audio_path"] = instruction.get("audio_path", "")

        if screenshot_path:
            # Save the screenshot path and upload URL
            if self.s3_manager:
                url = self.s3_manager.upload_file(screenshot_path, self.session_id, "screenshots",os.path.basename(screenshot_path))

            step["screenshot_path"] = screenshot_path
            step["screenshot_url"] = url

            self.screenshots.append({
                "step_index": self.step_index,
                "path": screenshot_path,
                "url": url
            })

        self.steps.append(step)
        self.step_index += 1

        # Only increment user step for non-system steps
        if not is_system_step:
            self.user_step_index += 1

        self._save_state()

        # Reset the next_step_requested flag
        self.next_step_requested = False

        logger.info(f"Step {self.step_index - 1} added")
        return self.step_index - 1

    def add_knowledge(self, knowledge_items: List[Dict[str, Any]]):
        """Add knowledge items to the session"""
        if not knowledge_items:
            return

        self.knowledge.extend(knowledge_items)

        # Save knowledge to file
        knowledge_dir = os.path.join(self.session_dir, "knowledge")
        os.makedirs(knowledge_dir, exist_ok=True)

        knowledge_file = os.path.join(knowledge_dir, f"knowledge_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        try:
            with open(knowledge_file, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "goal": self.goal,
                    "timestamp": datetime.now().isoformat(),
                    "items": knowledge_items
                }, f, indent=2)

            # Upload to S3
            # CHANGE: Use self.s3_manager instead
            if self.s3_manager:
                self.s3_manager.upload_file(knowledge_file, self.session_id, "knowledge",
                                            os.path.basename(knowledge_file))


            logger.info(f"Added {len(knowledge_items)} knowledge items to session")
            self._save_state()
        except Exception as e:
            logger.error(f"Error saving knowledge: {e}")

    def add_chat_interaction(self, role: str, content: str):
        """Add a chat interaction to the session"""
        chat_item = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        self.chat_history.append(chat_item)

        # Save chat history to file
        chat_dir = os.path.join(self.session_dir, "chat")
        os.makedirs(chat_dir, exist_ok=True)

        chat_file = os.path.join(chat_dir, f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

        try:
            with open(chat_file, 'w') as f:
                json.dump({
                    "session_id": self.session_id,
                    "timestamp": datetime.now().isoformat(),
                    "interaction": chat_item
                }, f, indent=2)

            # Upload to S3

            if self.s3_manager:
                self.s3_manager.upload_file(chat_file, self.session_id, "chat", os.path.basename(chat_file))

            logger.info(f"Added chat interaction: {role}")
            self._save_state()
        except Exception as e:
            logger.error(f"Error saving chat interaction: {e}")

    def request_next_step(self):
        """Set flag for user-requested next step"""
        self.next_step_requested = True
        logger.info("User requested next step")

    def toggle_explanation_mode(self, enabled: bool):
        """Toggle explanation mode"""
        self.include_explanations = enabled
        logger.info(f"Explanation mode set to: {enabled}")
        self._save_state()

    def toggle_audio_mode(self, enabled: bool):
        """Toggle audio mode"""
        self.audio_enabled = enabled
        logger.info(f"Audio mode set to: {enabled}")
        self._save_state()

    def update_status(self, status: str):
        """Update the session status"""
        self.status = status

        if status == "completed":
            self.end_time = datetime.now().isoformat()

        self._save_state()
        logger.info(f"Session status updated to: {status}")

    def get_current_progress(self) -> Dict[str, Any]:
        """Get the current session progress"""
        # Calculate time elapsed
        if hasattr(self, 'end_time'):
            end_time = datetime.fromisoformat(self.end_time)
        else:
            end_time = datetime.now()

        start_time = datetime.fromisoformat(self.start_time)
        time_elapsed = (end_time - start_time).total_seconds()

        # Count non-system steps
        user_steps = len([s for s in self.steps if not s.get('system_step', False)])

        return {
            "session_id": self.session_id,
            "status": self.status,
            "goal": self.goal,
            "step_count": user_steps,
            "time_elapsed_seconds": time_elapsed,
            "knowledge_count": len(self.knowledge),
            "chat_count": len(self.chat_history)
        }


# Main Training Wheels class
class TrainingWheels:
    """Main class to coordinate training wheels functionality"""

    def __init__(self):
        self.s3_manager = S3Manager()
        self.screenshot_manager = ScreenshotManager(self.s3_manager)
        #self.llm_manager = LLMManager()
        self.llm_manager = LLMManager(self.s3_manager)
        self.session = None
        self.monitoring_thread = None
        self.monitoring_active = False
        self.callback = None
        self.status_callback = None

        # Ensure directories exist
        Config.ensure_dirs()

        # Validate configuration
        Config.validate()

        logger.info("Training Wheels initialized")

    def set_region(self, region: Tuple[int, int, int, int]) -> bool:
        """Set the screen region to monitor"""
        return self.screenshot_manager.set_region(region)

    def start_session(self, user_id: str, course_id: str, goal: str, callback: Callable,
                      status_callback: Callable | None = None,
                      include_explanations: bool = Config.DEFAULT_EXPLANATION_MODE) -> str:
        """Start a new training session"""
        # Generate session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        # Create session
        #self.session = SessionState(session_id, user_id, course_id, goal, include_explanations)
        #self.session = SessionState(session_id, user_id, course_id, goal, include_explanations, self.s3_manager)
        self.session = SessionState(session_id, user_id, course_id, goal, include_explanations, self.s3_manager)
        self.callback = callback
        self.status_callback = status_callback

        # Update status
        self.session.update_status("initializing")

        # Take initial screenshot and add first step (step 0 - system initialization)
        screenshot_path = self.screenshot_manager.capture_screenshot(
            session_id,
            0,
            force=True
        )

        # Add initial step (placeholder until knowledge is fetched)
        if screenshot_path:
            # This is a system step, not shown to user
            init_step = {
                "instruction": f"Initializing session for goal: {goal}",
                "format": "text",
                "system_step": True  # Mark as system step
            }
            self.session.add_step(init_step, screenshot_path)

        logger.info(f"Session started: {session_id}")
        self._notify_status("session_started", {"session_id": session_id})
        return session_id

    def _notify_status(self, status_type: str, data: Any | None = None):
        """Tell the UI that something noteworthy happened."""
        if self.status_callback:
            try:
                self.status_callback(status_type, data)
            except Exception as e:
                logger.error(f"Status-callback error: {e}")

    def add_knowledge(self, knowledge: List[Dict[str, Any]]):
        """Add knowledge to the session"""
        if not self.session:
            logger.error("Cannot add knowledge: No active session")
            return False

        self.session.add_knowledge(knowledge)
        return True

    def add_chat_interaction(self, role: str, content: str):
        """Add a chat interaction to the session knowledge"""
        if not self.session:
            logger.error("Cannot add chat interaction: No active session")
            return False

        self.session.add_chat_interaction(role, content)
        return True

    def toggle_explanation_mode(self, enabled: bool):
        """Toggle explanation mode"""
        if not self.session:
            logger.error("Cannot toggle explanation mode: No active session")
            return False

        self.session.toggle_explanation_mode(enabled)
        return True

    def toggle_audio_mode(self, enabled: bool):
        """Toggle audio mode"""
        if not self.session:
            logger.error("Cannot toggle audio mode: No active session")
            return False

        self.session.toggle_audio_mode(enabled)
        return True

    def start_monitoring(self) -> bool:
        """Start the monitoring thread"""
        if not self.session:
            logger.error("Cannot start monitoring: No active session")
            return False

        if not self.screenshot_manager.region:
            logger.error("Cannot start monitoring: No screen region set")
            return False

        # Update session status
        self.session.update_status("in_progress")

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        logger.info("Screenshot monitoring started")
        return True

    def _monitoring_loop(self):
        """Background thread to monitor screenshots and generate steps"""
        logger.info("Monitoring loop started")

        while self.monitoring_active:
            try:
                # Skip if session is not in progress
                if not self.session or self.session.status != "in_progress":
                    logger.info("Monitoring loop stopping: Session not active")
                    break

                # Check if user requested next step
                if self.session.next_step_requested:
                    logger.info("Processing user-requested next step")
                    self.process_next_step(force=True)
                else:
                    # Capture screenshot and process if changed
                    self.process_next_step(force=False)

                # Sleep before next check
                time.sleep(Config.SCREENSHOT_INTERVAL)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(2.0)  # Longer sleep on error

        logger.info("Monitoring loop stopped")

    def process_next_step(self, force: bool = False):
        """Process the next step based on current screen"""
        if not self.session:
            logger.error("Cannot process step: No active session")
            return

        if not self.screenshot_manager.region:
            logger.error("Cannot process step: No screen region set")
            return

        try:
            # Capture screenshot
            screenshot_path = self.screenshot_manager.capture_screenshot(
                self.session.session_id,
                self.session.step_index,
                force=force
            )

            # If no screenshot (no change) and not forced, do nothing
            if not screenshot_path and not force:
                return

            # If forced but no screenshot, create an error step
            if not screenshot_path and force:
                logger.error("Failed to capture screenshot for forced step")

                if self.callback:
                    self.callback({
                        "instruction": "Error capturing screenshot. Please check your application window.",
                        "format": "text",
                        "error": True
                    })
                return

            # Get LLM guidance if we have knowledge
            if self.session.knowledge:
                instruction = self.llm_manager.get_next_instruction(
                    session_id=self.session.session_id,
                    goal=self.session.goal,
                    step_index=self.session.step_index,
                    screenshot_path=screenshot_path,
                    knowledge=self.session.knowledge,
                    previous_steps=self.session.steps,
                    include_explanations=self.session.include_explanations,
                    chat_history=self.session.chat_history
                )
            else:
                # If no knowledge yet, just use a placeholder
                instruction = {
                    "instruction": "Fetching relevant information for your goal...",
                    "format": "text"
                }

            # Add the step
            self.session.add_step(instruction, screenshot_path)

            # Call the callback function
            if self.callback:
                self.callback(instruction)

        except Exception as e:
            logger.error(f"Error processing step: {e}")
            # Try to inform the user
            if hasattr(self, 'callback') and self.callback:
                try:
                    self.callback({
                        "instruction": f"Error processing step: {e}",
                        "format": "text",
                        "error": True
                    })
                except:
                    pass

    def request_next_step(self) -> bool:
        """Force a new step on user request"""
        if not self.session or self.session.status != "in_progress":
            logger.warning("Cannot request next step: Session not in progress")
            return False

        # Set the flag in the session state
        self.session.request_next_step()

        # If monitoring is not active, process immediately
        if not self.monitoring_active or not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.process_next_step(force=True)

        return True

    def pause_session(self) -> bool:
        """Pause the monitoring session"""
        if not self.session:
            logger.warning("Cannot pause: No active session")
            return False

        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)

        self.session.update_status("paused")
        logger.info("Session paused")
        return True

    def resume_session(self) -> bool:
        """Resume a paused session"""
        if not self.session:
            logger.warning("Cannot resume: No active session")
            return False

        if self.session.status != "paused":
            logger.warning(f"Cannot resume: Session is {self.session.status}")
            return False

        self.session.update_status("in_progress")
        self.start_monitoring()

        logger.info("Session resumed")
        return True

    def end_session(self, success: bool = True) -> bool:
        """End the current session"""
        if not self.session:
            logger.warning("Cannot end: No active session")
            return False

        # Stop monitoring
        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)

        # Update session status
        status = "completed" if success else "failed"
        self.session.update_status(status)

        # Send final status update
        if self.status_callback:
            try:
                progress = self.session.get_current_progress()
                self.status_callback("session_ended", progress)
            except Exception as e:
                logger.error(f"Error sending final status update: {e}")

        # Clear callback reference to prevent memory leaks
        self.callback = None
        self.status_callback = None

        logger.info(f"Session ended with status: {status}")
        return True

    def check_activity_timeout(self, timeout_minutes: Optional[int] = None) -> bool:
        """Check if the session has been inactive for too long"""
        if not self.session or not self.session.steps:
            return False

        # Use configuration value if not specified
        if timeout_minutes is None:
            timeout_minutes = Config.SESSION_TIMEOUT_MINUTES

        # Get the timestamp of the last step
        last_step = self.session.steps[-1]
        last_step_time = datetime.fromisoformat(last_step.get('timestamp', self.session.start_time))

        # Calculate time difference
        now = datetime.now()
        time_diff = now - last_step_time

        # Check if the difference exceeds the timeout
        if time_diff.total_seconds() > (timeout_minutes * 60):
            logger.warning(f"Session inactive for {timeout_minutes} minutes, considering timeout")
            return True

        return False

    def get_session_progress(self) -> Dict[str, Any]:
        """Get the current session progress"""
        if not self.session:
            return {
                "active": False,
                "status": None
            }

        return {
            "active": True,
            **self.session.get_current_progress()
        }


# Global instance for use from the UI
_training_wheels = None


def get_instance() -> TrainingWheels:
    """Get the global TrainingWheels instance"""
    global _training_wheels
    if _training_wheels is None:
        _training_wheels = TrainingWheels()
    return _training_wheels


# === API Functions ===

def set_capture_region(region: Tuple[int, int, int, int]) -> bool:
    """Set the region of the screen to capture"""
    return get_instance().set_region(region)


def start_new_session(user_id: str, course_id: str, goal: str, callback: Callable,
                      status_callback: Callable | None = None,
                      include_explanations: bool = Config.DEFAULT_EXPLANATION_MODE) -> str:
    """Start a new training session"""
    return get_instance().start_session(user_id, course_id, goal, callback, status_callback, include_explanations)


def add_knowledge(knowledge: List[Dict[str, Any]]) -> bool:
    """Add knowledge to the current session"""
    return get_instance().add_knowledge(knowledge)


def add_chat_interaction(role: str, content: str) -> bool:
    """Add a chat interaction to the session knowledge"""
    return get_instance().add_chat_interaction(role, content)


def toggle_explanation_mode(enabled: bool) -> bool:
    """Toggle explanation mode"""
    return get_instance().toggle_explanation_mode(enabled)


def toggle_audio_mode(enabled: bool) -> bool:
    """Toggle audio mode"""
    return get_instance().toggle_audio_mode(enabled)


def start_guidance() -> bool:
    """Start the guidance monitoring"""
    return get_instance().start_monitoring()


def request_next_step() -> bool:
    """Request the next step (on user action)"""
    return get_instance().request_next_step()


def pause_guidance() -> bool:
    """Pause the current guidance session"""
    return get_instance().pause_session()


def resume_guidance() -> bool:
    """Resume a paused guidance session"""
    return get_instance().resume_session()


def end_guidance(success: bool = True) -> bool:
    """End the current guidance session"""
    return get_instance().end_session(success)


def test_s3_connection():
    """Test S3 connection and return result"""
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_DEFAULT_REGION
        )

        # Test connection
        response = s3_client.list_buckets()

        # Test bucket
        try:
            s3_client.head_bucket(Bucket=Config.S3_BUCKET_NAME)
            bucket_exists = True
        except Exception:
            bucket_exists = False

        return {
            "connected": True,
            "bucket_exists": bucket_exists,
            "buckets": [bucket['Name'] for bucket in response['Buckets']]
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }


def test_polly_connection():
    """Test AWS Polly connection and return result"""
    try:
        polly_client = boto3.client(
            'polly',
            aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
            region_name=Config.AWS_DEFAULT_REGION
        )

        # Test connection by listing voices
        response = polly_client.describe_voices(LanguageCode='en-US')

        return {
            "connected": True,
            "voices": [voice['Id'] for voice in response['Voices']],
            "default_voice": Config.POLLY_VOICE_ID
        }
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }


def check_session_status() -> Dict[str, Any]:
    """Get the status of the current session"""
    instance = get_instance()
    progress = instance.get_session_progress()

    # Check for timeout
    timed_out = instance.check_activity_timeout()
    progress["timed_out"] = timed_out

    return progress


def get_active_knowledge() -> List[Dict[str, Any]]:
    """Get knowledge from the active session"""
    instance = get_instance()
    if not instance.session:
        return []

    return instance.session.knowledge


# Initialize when module is imported
get_instance()