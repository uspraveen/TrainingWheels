# training_wheels.py - Fixed S3 upload deduplication and performance issues

import os
import json
import uuid
import time
import hashlib
import threading
import requests
import sys
import base64
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Callable
import pyautogui
import boto3
from PIL import Image, ImageDraw, ImageFont
import imagehash
import tempfile
from io import BytesIO
from dotenv import load_dotenv
import dotenv

# Import retriever for preemptive initialization
try:
    import retriever
except ImportError as e:
    print(f"Failed to import retriever: {e}")

# Load environment variables
load_dotenv()
dotenv.load_dotenv()


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
    

    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist"""
        os.makedirs(cls.STORAGE_DIR, exist_ok=True)
        

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
            print("S3 is enabled but AWS credentials are missing. Check your environment variables.")
            return False

        if not cls.OPENAI_API_KEY:
            print("OpenAI API key is missing. Check your environment variables.")
            return False

        return True


class S3Manager:
    """Manages S3 operations with deduplication - S3 only, no fallbacks"""

    def __init__(self):
        self.s3_client = None
        self._upload_cache = {}
        self._file_hash_cache = {}
        self._initialize_s3_client()

    def _get_file_hash(self, file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            return file_hash
        except Exception:
            return str(time.time())

    def _initialize_s3_client(self):
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=Config.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=Config.AWS_SECRET_ACCESS_KEY,
                region_name=Config.AWS_DEFAULT_REGION
            )
            # Test connection
            self.s3_client.list_buckets()
            # Test bucket exists
            self.s3_client.head_bucket(Bucket=Config.S3_BUCKET_NAME)
        except Exception as e:
            print(f"S3 initialization failed: {e}")
            self.s3_client = None

    def upload_file(self, local_path: str, session_id: str, file_type: str, filename: str) -> str:
        if not self.s3_client:
            self._initialize_s3_client()
            if not self.s3_client:
                raise Exception("S3 not available")

        # Check cache
        cache_key = (session_id, file_type, filename)
        if cache_key in self._upload_cache:
            return self._upload_cache[cache_key]

        # Check file hash
        if os.path.exists(local_path):
            file_hash = self._get_file_hash(local_path)
            hash_key = f"{session_id}_{file_type}_{file_hash}"
            if hash_key in self._file_hash_cache:
                cached_hash, cached_url = self._file_hash_cache[hash_key]
                if cached_hash == file_hash:
                    self._upload_cache[cache_key] = cached_url
                    return cached_url

        # Ensure file exists
        if not os.path.exists(local_path):
            if file_type == "screenshots":
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                error_img = Image.new('RGB', (800, 600), color=(255, 255, 255))
                draw = ImageDraw.Draw(error_img)
                draw.text((50, 50), f"Error: Screenshot not found\n{datetime.now()}", fill=(255, 0, 0))
                error_img.save(local_path)

        # Upload to S3
        key = f"{Config.S3_FOLDER}/{session_id}/{file_type}/{filename}"
        self.s3_client.upload_file(local_path, Config.S3_BUCKET_NAME, key)
        url = f"https://{Config.S3_BUCKET_NAME}.s3.amazonaws.com/{key}"
        
        # Cache result
        self._upload_cache[cache_key] = url
        if os.path.exists(local_path):
            file_hash = self._get_file_hash(local_path)
            hash_key = f"{session_id}_{file_type}_{file_hash}"
            self._file_hash_cache[hash_key] = (file_hash, url)
        
        return url


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
            print(f"Invalid region dimensions: {region}")
            return False

        self.region = region
        print(f"Screen capture region set to {region}")

        # Take test screenshot
        success, test_path = self.take_test_screenshot()
        return success

    def take_test_screenshot(self) -> Tuple[bool, Optional[str]]:
        """Take a test screenshot to verify the region works"""
        if not self.region:
            print("Cannot take test screenshot: No region set")
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
                print(f"Test screenshot successful with PyAutoGUI: {test_path}")
            except Exception as e:
                print(f"PyAutoGUI screenshot failed: {e}, trying alternative methods")

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
                        print(f"Test screenshot successful with ImageMagick: {test_path}")

                        # Optimize the saved image
                        try:
                            img = Image.open(test_path)
                            img = self._optimize_image(img)
                            img.save(test_path)
                        except Exception as opt_error:
                            print(f"Failed to optimize ImageMagick screenshot: {opt_error}")

                    except Exception as img_error:
                        print(f"ImageMagick screenshot failed: {img_error}, trying gnome-screenshot")

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
                            print(f"Test screenshot successful with gnome-screenshot: {test_path}")

                            # Optimize the saved image
                            try:
                                img = Image.open(test_path)
                                img = self._optimize_image(img)
                                img.save(test_path)
                            except Exception as opt_error:
                                print(f"Failed to optimize gnome-screenshot: {opt_error}")

                        except Exception as gnome_error:
                            print(f"All screenshot methods failed: {gnome_error}")
                            return False, None
                else:
                    # If not on Linux and PyAutoGUI failed, we fail
                    print(f"Screenshot failed and no alternatives available on this platform")
                    return False, None

            # Upload to S3 only once - no duplication
            self.s3_manager.upload_file(
                test_path,
                "test",
                "screenshots",
                test_filename
            )

            return True, test_path
        except Exception as e:
            print(f"Test screenshot failed: {e}")
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

            # Convert to RGB if needed (in case of RGBA)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            return image
        except Exception as e:
            print(f"Error optimizing image: {e}")
            return image  # Return original image on error

    def capture_screenshot(self, session_id: str, step_index: int, force: bool = False) -> Optional[str]:
        """Capture screenshot and return the path if it represents a change"""
        if not self.region:
            print("Cannot capture screenshot: No region set")
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
                    print(f"Screenshot file missing or empty after capture: {file_path}")
                    return None

            except Exception as e:
                print(f"Failed to capture screenshot: {e}")
                return None

            # Check if it represents a change or if we should force capture
            if force:
                self.no_change_counter = 0
                self.last_image_hash = imagehash.phash(screenshot)
                print(f"Forced screenshot captured: {file_path}")

                # Add to previous screenshots for context
                self.previous_screenshots.append(file_path)
                if len(self.previous_screenshots) > 5:  # Keep only the 5 most recent
                    self.previous_screenshots.pop(0)

                return file_path

            # Check for significant change
            is_change = self._is_significant_change(screenshot)

            if is_change:
                # Significant change detected
                self.no_change_counter = 0
                print(f"Screenshot captured due to significant change: {file_path}")

                # Add to previous screenshots for context
                self.previous_screenshots.append(file_path)
                if len(self.previous_screenshots) > 5:  # Keep only the 5 most recent
                    self.previous_screenshots.pop(0)

                return file_path
            else:
                # No significant change
                self.no_change_counter += 1

                # Check if we've had too many consecutive no-changes
                if self.no_change_counter >= Config.MAX_CONSECUTIVE_NO_CHANGES:
                    self.no_change_counter = 0
                    print(f"Screenshot captured after {Config.MAX_CONSECUTIVE_NO_CHANGES} consecutive no-changes: {file_path}")

                    # Add to previous screenshots for context
                    self.previous_screenshots.append(file_path)
                    if len(self.previous_screenshots) > 5:  # Keep only the 5 most recent
                        self.previous_screenshots.pop(0)

                    return file_path
                else:
                    # No change and not enough consecutive no-changes
                    # Delete the file since we're not using it
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception as e:
                        print(f"Could not delete unused screenshot {file_path}: {e}")
                    
                    return None

        except Exception as e:
            print(f"Screenshot capture failed: {e}")
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

            if difference > Config.CHANGE_THRESHOLD:
                self.last_image_hash = current_hash
                return True

            return False
        except Exception as e:
            print(f"Error checking for significant change: {e}")
            return True  # Default to assuming change on error


# LLM Manager
class LLMManager:
    """Manages interactions with LLM for step guidance"""

    def __init__(self, s3_manager=None):
        self.api_key = Config.OPENAI_API_KEY
        self.model = Config.OPENAI_MODEL
        
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
        """Get next instruction from LLM - S3 only"""
        try:
            # Get S3 URL
            screenshot_url = self.s3_manager.upload_file(
                screenshot_path,
                session_id,
                "screenshots",
                os.path.basename(screenshot_path)
            )

            # Build messages
            system_message = self._build_system_message(step_index, include_explanations)
            user_message_text = self._build_user_message_text(
                goal=goal,
                step_index=step_index,
                knowledge=knowledge,
                previous_steps=previous_steps or [],
                chat_history=chat_history or []
            )

            # Call OpenAI API
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

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload
            )

            if response.status_code != 200:
                return {
                    "instruction": f"Error getting step {step_index} guidance. Please try again.",
                    "format": "text",
                    "error": True
                }

            # Parse response
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content")

            if not content:
                return {
                    "instruction": f"Error generating step {step_index}. Please try again.",
                    "format": "text",
                    "error": True
                }

            # Save response
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

            self.s3_manager.upload_file(response_path, session_id, "llm_responses", response_filename)

            # Parse JSON response
            try:
                guidance = json.loads(content)
                instruction_data = {
                    "instruction": guidance.get("instruction", f"Step {step_index}: No specific instruction provided"),
                    "format": guidance.get("format", "text"),
                    "details": guidance.get("details", {})
                }

                if "explanation" in guidance:
                    instruction_data["explanation"] = guidance.get("explanation", "")

                return instruction_data

            except (json.JSONDecodeError, TypeError):
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

        except Exception:
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
    - IMPORTANT: If no course materials are available, use your general knowledge and best practices to guide the user through their goal. You can still provide valuable step-by-step guidance even without specific course content.
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
        else:
            knowledge_text = "No specific course materials found for this goal.\n"

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
        
        self.s3_manager = s3_manager
        # Track uploaded files to prevent duplicates
        self._uploaded_files = set()

        # Create session directory
        self.session_dir = Config.get_session_dir(session_id)
        self.state_file = os.path.join(self.session_dir, "session_state.json")

        # Initialize state file
        self._save_state()

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
            
            "chat_history_count": len(self.chat_history)
        }

        if hasattr(self, 'end_time'):
            state["end_time"] = self.end_time

        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)

            # Also save to S3 only once per save
            state_upload_key = f"{self.session_id}_state_{os.path.basename(self.state_file)}"
            if state_upload_key not in self._uploaded_files:
                if self.s3_manager:
                    self.s3_manager.upload_file(self.state_file, self.session_id, "state", "session_state.json")
                    self._uploaded_files.add(state_upload_key)

        except Exception as e:
            print(f"Error saving session state: {e}")

    def add_step(self, instruction: Dict[str, Any], screenshot_path: str = None):
        """Add a step to the session with deduplication"""
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

        if screenshot_path:
            # Create unique upload key for this screenshot
            screenshot_upload_key = f"{self.session_id}_screenshot_{os.path.basename(screenshot_path)}"
            
            # Only upload if not already uploaded
            if screenshot_upload_key not in self._uploaded_files:
                if self.s3_manager:
                    url = self.s3_manager.upload_file(
                        screenshot_path, 
                        self.session_id, 
                        "screenshots",
                        os.path.basename(screenshot_path)
                    )
                    self._uploaded_files.add(screenshot_upload_key)
                else:
                    url = screenshot_path
            else:
                # Use cached URL or reconstruct it
                url = f"https://{Config.S3_BUCKET_NAME}.s3.amazonaws.com/{Config.S3_FOLDER}/{self.session_id}/screenshots/{os.path.basename(screenshot_path)}"

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

            # Upload to S3 only once
            knowledge_upload_key = f"{self.session_id}_knowledge_{os.path.basename(knowledge_file)}"
            if knowledge_upload_key not in self._uploaded_files:
                if self.s3_manager:
                    self.s3_manager.upload_file(knowledge_file, self.session_id, "knowledge",
                                                os.path.basename(knowledge_file))
                    self._uploaded_files.add(knowledge_upload_key)

            self._save_state()
        except Exception:
            pass

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

            # Upload to S3 only once
            chat_upload_key = f"{self.session_id}_chat_{os.path.basename(chat_file)}"
            if chat_upload_key not in self._uploaded_files:
                if self.s3_manager:
                    self.s3_manager.upload_file(chat_file, self.session_id, "chat", os.path.basename(chat_file))
                    self._uploaded_files.add(chat_upload_key)

            print(f"Added chat interaction: {role}")
            self._save_state()
        except Exception as e:
            print(f"Error saving chat interaction: {e}")

    def request_next_step(self):
        """Set flag for user-requested next step"""
        self.next_step_requested = True
        print("User requested next step")

    def toggle_explanation_mode(self, enabled: bool):
        """Toggle explanation mode"""
        self.include_explanations = enabled
        print(f"Explanation mode set to: {enabled}")
        self._save_state()

    def update_status(self, status: str):
        """Update the session status"""
        self.status = status

        if status == "completed":
            self.end_time = datetime.now().isoformat()

        self._save_state()
        print(f"Session status updated to: {status}")

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
        self.session = SessionState(session_id, user_id, course_id, goal, include_explanations, self.s3_manager)
        self.callback = callback
        self.status_callback = status_callback

        # Update status
        self.session.update_status("initializing")

        # Take initial screenshot and add first step
        screenshot_path = self.screenshot_manager.capture_screenshot(
            session_id,
            0,
            force=True
        )

        # Add initial step
        if screenshot_path:
            init_step = {
                "instruction": f"Initializing session for goal: {goal}",
                "format": "text",
                "system_step": True
            }
            self.session.add_step(init_step, screenshot_path)

        self._notify_status("session_started", {"session_id": session_id})
        return session_id

    def _notify_status(self, status_type: str, data: Any | None = None):
        """Tell the UI that something noteworthy happened."""
        if self.status_callback:
            try:
                self.status_callback(status_type, data)
            except Exception:
                pass

    def add_knowledge(self, knowledge: List[Dict[str, Any]]):
        """Add knowledge to the session"""
        if not self.session:
            return False

        self.session.add_knowledge(knowledge)
        return True

    def add_chat_interaction(self, role: str, content: str):
        """Add a chat interaction to the session knowledge"""
        if not self.session:
            return False

        self.session.add_chat_interaction(role, content)
        return True

    def toggle_explanation_mode(self, enabled: bool):
        """Toggle explanation mode"""
        if not self.session:
            return False

        self.session.toggle_explanation_mode(enabled)
        return True

    def start_monitoring(self) -> bool:
        """Start the monitoring thread"""
        if not self.session:
            return False

        if not self.screenshot_manager.region:
            return False

        # Update session status
        self.session.update_status("in_progress")

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        return True

    def _monitoring_loop(self):
        """Background thread to monitor screenshots and generate steps"""
        while self.monitoring_active:
            try:
                # Skip if session is not in progress
                if not self.session or self.session.status != "in_progress":
                    break

                # Check if user requested next step
                if self.session.next_step_requested:
                    self.process_next_step(force=True)
                else:
                    # Capture screenshot and process if changed
                    self.process_next_step(force=False)

                # Sleep before next check
                time.sleep(Config.SCREENSHOT_INTERVAL)

            except Exception:
                time.sleep(2.0)  # Longer sleep on error

    def process_next_step(self, force: bool = False):
        """Process the next step based on current screen with deduplication"""
        if not self.session:
            return

        if not self.screenshot_manager.region:
            return

        try:
            # Capture screenshot (this handles deduplication internally)
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
                if self.callback:
                    self.callback({
                        "instruction": "Error capturing screenshot. Please check your application window.",
                        "format": "text",
                        "error": True
                    })
                return

            # Always get LLM guidance (even with no knowledge - let LLM use general knowledge)
            instruction = self.llm_manager.get_next_instruction(
                session_id=self.session.session_id,
                goal=self.session.goal,
                step_index=self.session.step_index,
                screenshot_path=screenshot_path,
                knowledge=self.session.knowledge,  # Pass whatever knowledge we have (could be empty)
                previous_steps=self.session.steps,
                include_explanations=self.session.include_explanations,
                chat_history=self.session.chat_history
            )

            # Add the step (this handles S3 upload deduplication)
            self.session.add_step(instruction, screenshot_path)

            # Call the callback function
            if self.callback:
                self.callback(instruction)

        except Exception:
            # Try to inform the user
            if hasattr(self, 'callback') and self.callback:
                try:
                    self.callback({
                        "instruction": f"Error processing step",
                        "format": "text",
                        "error": True
                    })
                except:
                    pass

    def request_next_step(self) -> bool:
        """Force a new step on user request"""
        if not self.session or self.session.status != "in_progress":
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
            return False

        self.monitoring_active = False

        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=1.0)

        self.session.update_status("paused")
        return True

    def resume_session(self) -> bool:
        """Resume a paused session"""
        if not self.session:
            return False

        if self.session.status != "paused":
            return False

        self.session.update_status("in_progress")
        self.start_monitoring()

        return True

    def end_session(self, success: bool = True) -> bool:
        """End the current session"""
        if not self.session:
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
            except Exception:
                pass

        # Clear callback reference to prevent memory leaks
        self.callback = None
        self.status_callback = None

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


def cleanup_session_data(session_id: str):
    """Clean up all session data when session ends"""
    try:
        session_dir = Config.get_session_dir(session_id)
        if os.path.exists(session_dir):
            import shutil
            shutil.rmtree(session_dir)
    except Exception:
        pass


def end_guidance(success: bool = True) -> bool:
    """End the current guidance session"""
    instance = get_instance()
    session_id = None
    
    # Get session ID before ending
    if instance.session:
        session_id = instance.session.session_id
    
    # End the session
    result = instance.end_session(success)
    
    # Clean up session data after ending
    if session_id:
        cleanup_session_data(session_id)
    
    return result

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