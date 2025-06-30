# learnchain_tutor.py - Complete updated version with enhanced chat dialog

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QStackedWidget, QScrollArea, QFrame, QHBoxLayout,
    QSizePolicy, QGraphicsOpacityEffect, QSpacerItem, QTextEdit,
    QSplitter, QDialog, QDialogButtonBox, QCheckBox, QSlider,
    QComboBox, QFileDialog, QProgressDialog, QMessageBox, QRadioButton
)
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QFontDatabase, QPainter, QPen, QPainterPath
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, pyqtSignal, QEasingCurve, QSize, QTimer, QPoint, QRect,
    QSequentialAnimationGroup, QVariantAnimation, QUrl, QThread, QBuffer
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

import sys
import os
import json
import logging
import base64
import pyautogui
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from PyQt6.QtWidgets import QApplication
import ui_integration
from PyQt6.QtGui import QIcon
from PyQt6.QtGui import QIcon
import winreg
import os
import sys
from pathlib import Path
import ctypes


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enhanced modern dark theme color scheme
PRIMARY_COLOR = "#121212"  # Deep dark gray background
CARD_BACKGROUND = "rgba(30, 30, 30, 0.8)"  # Semi-transparent for glass effect
ACCENT_COLOR = "#FF3B30"  # Vivid orange-red
SECONDARY_COLOR = "#10b981"  # Keep green for success
TEXT_COLOR = "#FFFFFF"  # Pure white for primary text
SECONDARY_TEXT = "#CCCCCC"  # Soft gray for secondary text
LIGHT_GRAY = "#2A2A2A"  # Darker gray for alternate backgrounds
GRAY_TEXT = "#CCCCCC"  # Medium gray for secondary text
BUTTON_ORANGE = "#FF3B30"  # Use accent color
BUTTON_RED = "#dc2626"  # Keep red for stop
BUTTON_BLUE = "#FF3B30"  # Use accent color for consistency
BORDER_COLOR = "#3A3A3A"  # Darker border
DISABLED_COLOR = "#555555"  # Darker disabled state
STEP_LABEL_BG = "#FF3B30"  # Use accent color
PAUSED_BG_COLOR = "#1A1A1A"  # Slightly different dark
GOAL_BG_COLOR = "rgba(30, 30, 30, 0.6)"  # Glass effect
GOAL_BORDER_COLOR = "#FF3B30"  # Accent border
EXPLANATION_BG_COLOR = "rgba(45, 45, 45, 0.8)"  # Dark explanation background
EXPLANATION_BORDER_COLOR = "#FF3B30"  # Accent border

#Icon paths (assuming these files exist in an 'icons' directory)
#ICON_DIR = os.path.join(os.path.dirname(__file__), "icons")


def is_admin():
    """Check if running as administrator"""
    try:
        import ctypes
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def run_as_admin():
    """Restart the application as administrator"""
    try:
        import ctypes
        import sys

        if is_admin():
            return True

        # Get the current executable path
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            exe_path = sys.executable
        else:
            # Running as Python script
            exe_path = sys.executable
            script_path = os.path.abspath(sys.argv[0])
            args = f'"{script_path}"'

        # Attempt elevation
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",  # Verb for "Run as administrator"
            exe_path,
            args if not getattr(sys, 'frozen', False) else "",
            None,
            1  # nShowCmd = SW_SHOWNORMAL
        )

        # If successful (result > 32), exit current instance
        if result > 32:
            print("✅ Successfully elevated to administrator")
            return False  # Signal to exit current instance
        else:
            print(f"❌ Elevation failed with code: {result}")
            return True  # Continue without elevation

    except Exception as e:
        print(f"❌ Elevation error: {e}")
        return True


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            # Running as PyInstaller bundle
            base_path = sys._MEIPASS
            resource_file = os.path.join(base_path, relative_path)
            print(f"PyInstaller mode, base path: {base_path}")
            print(f"Resource path for '{relative_path}': {resource_file}")

            # Also try with training_wheels prefix if direct path doesn't work
            if not os.path.exists(resource_file):
                alt_path = os.path.join(base_path, "training_wheels", relative_path)
                print(f"Trying alternate path: {alt_path}")
                if os.path.exists(alt_path):
                    return alt_path

            return resource_file
        else:
            # Running in development mode
            base_path = os.path.dirname(os.path.realpath(__file__))
            resource_file = os.path.join(base_path, relative_path)
            print(f"Development mode, base path: {base_path}")
            print(f"Resource path for '{relative_path}': {resource_file}")
            return resource_file

    except Exception as e:
        print(f"Error in resource_path: {e}")
        # Fallback to simple path
        return os.path.join(os.path.dirname(__file__), relative_path)


ICON_DIR = Path(resource_path("icons"))
ASSET_DIR = Path(resource_path("assets"))

def set_taskbar_icon(window, icon_path):
    """Force set taskbar icon using Windows API"""
    try:
        import ctypes
        from ctypes import wintypes

        # Load the icon
        icon = QIcon(icon_path)
        window.setWindowIcon(icon)

        # Get window handle
        hwnd = int(window.winId())

        # Windows API constants
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1

        # Load icon using Windows API
        hicon = ctypes.windll.user32.LoadImageW(
            0,  # hInst
            icon_path,  # name
            1,  # type (IMAGE_ICON)
            0,  # cx
            0,  # cy
            0x00000010 | 0x00008000  # LR_LOADFROMFILE | LR_SHARED
        )

        if hicon:
            # Set both small and large icons
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon)
            ctypes.windll.user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon)
            print(f"✅ Windows API icon set: {icon_path}")
            return True
        else:
            print(f"❌ Failed to load icon via Windows API: {icon_path}")
            return False

    except Exception as e:
        print(f"❌ Error setting taskbar icon: {e}")
        return False

def force_icon_refresh():
    """Force Windows to refresh the taskbar icon"""
    try:
        # Clear icon cache
        import subprocess
        subprocess.run([
            'taskkill', '/f', '/im', 'explorer.exe'
        ], capture_output=True, check=False)

        subprocess.run([
            'del', '/a', '/q', os.path.expandvars(r'%localappdata%\IconCache.db')
        ], shell=True, capture_output=True, check=False)

        subprocess.run([
            'start', 'explorer.exe'
        ], shell=True, capture_output=True, check=False)
    except:
        pass

# Loading custom fonts
def load_fonts():
    font_dir = os.path.join(os.path.dirname(__file__), "fonts")

    # Ensure font directory exists
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)

    # Font paths - comment out if not available
    # QFontDatabase.addApplicationFont(os.path.join(font_dir, "Inter-Regular.ttf"))
    # QFontDatabase.addApplicationFont(os.path.join(font_dir, "Inter-Bold.ttf"))
    # QFontDatabase.addApplicationFont(os.path.join(font_dir, "Inter-SemiBold.ttf"))


class StepNumberLabel(QLabel):
    """Custom step number badge with rounded corners"""

    def __init__(self, step_number, parent=None):
        super().__init__(parent)
        self.setText(str(step_number))
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(40, 24)
        self.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self.setStyleSheet(f"""
            color: white;
            background-color: {STEP_LABEL_BG};
            border-radius: 12px;
        """)


class TypewriterLabel(QLabel):
    """Label with typewriter animation effect - FIXED VERSION"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.full_text = ""
        self.current_text = ""
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.update_text)
        self.char_index = 0
        self.char_delay = 5  # Changed from 30 to 5 milliseconds for faster typing effect

    def set_text_with_animation(self, text):
        """Start typewriter animation with the given text"""
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
            
        self.full_text = text
        self.current_text = ""
        self.char_index = 0
        self.setText("")
        
        # Only start animation if we have text
        if self.full_text:
            self.animation_timer.start(self.char_delay)

    def update_text(self):
        """Update the displayed text with the next character"""
        if self.char_index < len(self.full_text):
            self.current_text += self.full_text[self.char_index]
            self.setText(self.current_text)
            self.char_index += 1
        else:
            self.animation_timer.stop()


def render_instruction_blocks(blocks: List[Dict[str, str]], layout: QVBoxLayout):
    for block in blocks:
        btype = block.get("type")
        content = block.get("content", "")
        label = block.get("label", "")

        if btype == "text":
            text_label = QLabel(content)
            text_label.setWordWrap(True)
            text_label.setFont(QFont("Inter", 13))
            text_label.setStyleSheet(f"color: {TEXT_COLOR};")
            layout.addWidget(text_label)

        elif btype == "link":
            frame = QFrame()
            frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {LIGHT_GRAY};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 5px;
                    padding: 6px;
                }}
            """)
            hlayout = QHBoxLayout(frame)
            link_label = QLabel(label or content)
            link_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            link_label.setStyleSheet(f"color: {TEXT_COLOR};")
            hlayout.addWidget(link_label)

            copy_btn = QPushButton("Copy")
            copy_btn.clicked.connect(lambda _, c=content: QApplication.clipboard().setText(c))
            hlayout.addWidget(copy_btn)
            layout.addWidget(frame)

        elif btype == "code":
            step = AnimatedStep(0, {"instruction": content, "format": "code"}, False)
            layout.addWidget(step)


def parse_instruction_json(json_str: str) -> Optional[List[Dict[str, str]]]:
    try:
        parsed = json.loads(json_str)
        return parsed.get("blocks", [])
    except json.JSONDecodeError:
        return None


class AnimatedStep(QFrame):
    """A custom widget for displaying an instruction step with animations"""

    def __init__(self, step_number, content, is_current=False, parent=None):
        super().__init__(parent)

        # Set frame properties
        self.setObjectName("stepItem")
        self.setMinimumHeight(90)  # Minimum height that grows with content

        # Style based on current status
        if is_current:
            self.setStyleSheet(f"""
                #stepItem {{
                    background-color: rgba(45, 45, 45, 0.6);
                    border-left: 4px solid {ACCENT_COLOR};
                    border-radius: 6px;
                    margin: 4px 0px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                #stepItem {{
                    background-color: rgba(30, 30, 30, 0.4);
                    border-left: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                    margin: 4px 0px;
                }}
            """)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(10)

        # Step number badge and label row
        step_header = QHBoxLayout()
        step_header.setContentsMargins(0, 0, 0, 5)

        self.step_badge = StepNumberLabel(step_number)
        step_header.addWidget(self.step_badge)

        # ✅ Remove "instructions enabled"
        step_label = QLabel(f"Step {step_number}")
        step_label.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        step_label.setStyleSheet(f"color: {ACCENT_COLOR};")
        step_header.addWidget(step_label)

        step_header.addStretch()
        layout.addLayout(step_header)

        # Defensive: handle empty content
        if not content:
            print("⚠️ No step content found — skipping.")
            return

        # ROBUST CONTENT HANDLING - Convert everything to safe format
        try:
            # Main logic: check for new JSON "blocks" format
            if isinstance(content, dict):
                blocks = []
                explanation = content.get('explanation', '')
                
                try:
                    instruction_data = content.get('instruction', '')
                    
                    # Handle different instruction formats
                    if isinstance(instruction_data, list):
                        # Instruction is already a list of blocks
                        blocks = instruction_data
                    elif isinstance(instruction_data, str):
                        # Try to parse as JSON
                        try:
                            parsed = json.loads(instruction_data)
                            blocks = parsed.get('blocks', []) if isinstance(parsed, dict) else []
                            if not explanation and isinstance(parsed, dict) and 'explanation' in parsed:
                                explanation = parsed.get('explanation', '')
                        except (json.JSONDecodeError, TypeError):
                            # If not JSON, treat as plain text
                            blocks = [{"type": "text", "content": str(instruction_data)}]
                    elif isinstance(instruction_data, dict):
                        blocks = instruction_data.get('blocks', [])
                    else:
                        # Fallback: convert whatever it is to text
                        blocks = [{"type": "text", "content": str(instruction_data)}]
                        
                except Exception as e:
                    print(f"Error processing instruction data: {e}")
                    # Emergency fallback
                    blocks = [{"type": "text", "content": "Error processing instruction"}]

                if blocks:
                    for block in blocks:
                        try:
                            btype = block.get("type", "text")
                            value = str(block.get("content", ""))  # Always convert to string
                            label = str(block.get("label", ""))   # Always convert to string

                            if btype == "text":
                                # ENHANCED: Auto-detect URLs in text and make them copyable
                                if self._contains_url(value):
                                    self._create_text_with_links(value, layout)
                                else:
                                    text_label = QLabel(value)
                                    text_label.setFont(QFont("Inter", 13))
                                    text_label.setWordWrap(True)
                                    text_label.setStyleSheet(f"color: {TEXT_COLOR};")
                                    layout.addWidget(text_label)

                            elif btype == "link":
                                link_frame = QFrame()
                                link_frame.setStyleSheet(f"""
                                    QFrame {{
                                        background-color: {LIGHT_GRAY};
                                        border: 1px solid {BORDER_COLOR};
                                        border-radius: 5px;
                                        padding: 6px;
                                    }}
                                """)
                                hbox = QHBoxLayout(link_frame)
                                link_label = QLabel(label or value)
                                link_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                                link_label.setStyleSheet(f"color: {TEXT_COLOR};")
                                hbox.addWidget(link_label)
                                copy_btn = QPushButton("Copy")
                                copy_btn.clicked.connect(lambda _, v=value: QApplication.clipboard().setText(v))
                                hbox.addWidget(copy_btn)
                                layout.addWidget(link_frame)

                            elif btype == "code":
                                self.create_code_block(value, layout)
                        except Exception as block_error:
                            print(f"Error processing block: {block_error}")
                            # Add error block as fallback
                            error_label = QLabel(f"Error displaying block: {str(block_error)}")
                            error_label.setStyleSheet(f"color: {BUTTON_RED};")
                            layout.addWidget(error_label)
                else:
                    # Fallback to legacy format
                    instruction = content.get('instruction', '')
                    # Always ensure instruction is a string
                    if isinstance(instruction, list):
                        instruction = str(instruction)
                    elif not isinstance(instruction, str):
                        instruction = str(instruction)
                        
                    format_type = content.get('format', 'text')
                    if format_type == 'code':
                        self.create_code_block(instruction, layout)
                    else:
                        # ENHANCED: Handle URLs in legacy text format too
                        if self._contains_url(instruction):
                            self._create_text_with_links(instruction, layout)
                        else:
                            self.create_text_block(instruction, layout, is_current)

                if explanation:
                    self.add_explanation(str(explanation), layout)  # Ensure explanation is string
                    
            else:
                # Handle non-dict content - ENHANCED URL handling
                content_str = str(content) if content is not None else "No content"
                if self._contains_url(content_str):
                    self._create_text_with_links(content_str, layout)
                else:
                    self.create_text_block(content_str, layout, is_current)

        except Exception as e:
            print(f"Critical error in AnimatedStep: {e}")
            # Emergency fallback - create a simple error message
            error_label = QLabel(f"Error displaying step: {str(e)}")
            error_label.setStyleSheet(f"color: {BUTTON_RED};")
            layout.addWidget(error_label)

        if is_current:
            self.animate_entrance()

    def create_text_block(self, text, layout, animate=False):
        """Create a regular text display with optional typewriter effect"""
        # CRITICAL: Ensure text is always a string - ROBUST CONVERSION
        try:
            if text is None:
                text = "No content provided"
            elif isinstance(text, list):
                # If it's a list, try to extract meaningful content
                if len(text) == 0:
                    text = "No content provided"
                elif len(text) == 1:
                    # Single item - convert to string
                    text = str(text[0])
                else:
                    # Multiple items - join them or take first meaningful one
                    text_items = []
                    for item in text:
                        if isinstance(item, dict) and 'content' in item:
                            text_items.append(str(item['content']))
                        else:
                            text_items.append(str(item))
                    text = '\n'.join(text_items) if text_items else "Multiple items provided"
            elif isinstance(text, dict):
                # If it's a dict, try to extract content
                if 'content' in text:
                    text = str(text['content'])
                elif 'instruction' in text:
                    text = str(text['instruction'])
                else:
                    text = str(text)
            elif not isinstance(text, str):
                # Any other type - convert to string
                text = str(text)
                
            # Final safety check
            if not text or not text.strip():
                text = "No content provided"
                
        except Exception as e:
            print(f"Error converting text in create_text_block: {e}")
            text = f"Error processing content: {str(e)}"
            
        if animate:
            # Animated text with typewriter effect
            text_label = TypewriterLabel()
            text_label.setFont(QFont("Inter", 13))
            text_label.setWordWrap(True)
            text_label.setStyleSheet(f"color: {TEXT_COLOR};")
            layout.addWidget(text_label)
            self.content_widget = text_label

            # Start animation with a short delay
            QTimer.singleShot(300, lambda: text_label.set_text_with_animation(text))
        else:
            # Static text (for previous steps)
            text_label = QLabel(text)
            text_label.setFont(QFont("Inter", 13))
            text_label.setWordWrap(True)
            text_label.setStyleSheet(f"color: {TEXT_COLOR};")
            layout.addWidget(text_label)
            self.content_widget = text_label

    def create_code_block(self, code, layout):
        """Create a code block display with syntax highlighting and language label"""
        # CRITICAL: Ensure code is always a string - ROBUST CONVERSION
        try:
            if code is None:
                code = "# No code provided"
            elif isinstance(code, list):
                # If it's a list, join the items or extract meaningful content
                if len(code) == 0:
                    code = "# No code provided"
                else:
                    code_items = []
                    for item in code:
                        if isinstance(item, dict) and 'content' in item:
                            code_items.append(str(item['content']))
                        else:
                            code_items.append(str(item))
                    code = '\n'.join(code_items) if code_items else "# No code provided"
            elif isinstance(code, dict):
                # If it's a dict, try to extract content
                if 'content' in code:
                    code = str(code['content'])
                else:
                    code = str(code)
            elif not isinstance(code, str):
                # Any other type - convert to string
                code = str(code)
                
            # Final safety check
            if not code or not code.strip():
                code = "# No code provided"
                
        except Exception as e:
            print(f"Error converting code in create_code_block: {e}")
            code = f"# Error processing code: {str(e)}"

        # Determine code language
        language = self.detect_code_language(code)

        # Code frame with monospace font and syntax highlighting
        code_frame = QFrame()
        code_frame.setStyleSheet(f"""
            QFrame {{
                background-color: #1e293b;
                border-radius: 6px;
                padding: 0px;
            }}
        """)

        code_layout = QVBoxLayout(code_frame)
        code_layout.setContentsMargins(0, 0, 0, 0)
        code_layout.setSpacing(0)

        # Language header
        language_bar = QFrame()
        language_bar.setFixedHeight(36)
        language_bar.setStyleSheet(f"""
            QFrame {{
                background-color: #0f172a;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                border-bottom: 1px solid #334155;
            }}
        """)

        language_layout = QHBoxLayout(language_bar)
        language_layout.setContentsMargins(12, 0, 12, 0)

        # Language indicator
        lang_label = QLabel(language)
        lang_label.setFont(QFont("Inter", 10, QFont.Weight.Medium))
        lang_label.setStyleSheet("color: #94a3b8;")
        language_layout.addWidget(lang_label)

        # Spacer to push copy button to the right
        language_layout.addStretch()

        # Copy button in the header
        copy_btn = QPushButton("Copy")
        copy_btn.setFont(QFont("Inter", 10))
        copy_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        copy_btn.setFixedSize(70, 24)
        copy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: #334155;
                color: #e2e8f0;
                border: none;
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QPushButton:hover {{
                background-color: #475569;
            }}
        """)
        copy_btn.clicked.connect(lambda: self.copy_to_clipboard(code))
        language_layout.addWidget(copy_btn)

        code_layout.addWidget(language_bar)

        # Code content
        code_edit = QTextEdit()
        code_edit.setPlainText(code)
        code_edit.setReadOnly(True)
        code_edit.setFont(QFont("Courier New", 12))
        code_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1e293b;
                color: #e2e8f0;
                border: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
                padding: 12px;
                selection-background-color: #3b82f6;
            }}
        """)
        code_layout.addWidget(code_edit)

        layout.addWidget(code_frame)
        self.content_widget = code_edit

    def add_explanation(self, explanation_text, layout):
        """Add an explanation block to the step"""
        # CRITICAL: Ensure explanation_text is always a string - ROBUST CONVERSION
        try:
            if explanation_text is None:
                explanation_text = "No explanation provided"
            elif isinstance(explanation_text, list):
                # If it's a list, join the items or extract meaningful content
                if len(explanation_text) == 0:
                    explanation_text = "No explanation provided"
                else:
                    exp_items = []
                    for item in explanation_text:
                        if isinstance(item, dict) and 'content' in item:
                            exp_items.append(str(item['content']))
                        else:
                            exp_items.append(str(item))
                    explanation_text = '\n'.join(exp_items) if exp_items else "No explanation provided"
            elif isinstance(explanation_text, dict):
                # If it's a dict, try to extract content
                if 'content' in explanation_text:
                    explanation_text = str(explanation_text['content'])
                else:
                    explanation_text = str(explanation_text)
            elif not isinstance(explanation_text, str):
                # Any other type - convert to string
                explanation_text = str(explanation_text)
                
            # Final safety check
            if not explanation_text or not explanation_text.strip():
                explanation_text = "No explanation provided"
                
        except Exception as e:
            print(f"Error converting explanation in add_explanation: {e}")
            explanation_text = f"Error processing explanation: {str(e)}"
            
        explanation_frame = QFrame()
        explanation_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {EXPLANATION_BG_COLOR};
                border-left: 3px solid {EXPLANATION_BORDER_COLOR};
                border-radius: 6px;
                margin-top: 5px;
            }}
        """)

        explanation_layout = QVBoxLayout(explanation_frame)
        explanation_layout.setContentsMargins(12, 10, 12, 10)

        # Header
        header = QLabel("Why this step matters:")
        header.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {ACCENT_COLOR};")
        explanation_layout.addWidget(header)

        # Explanation text
        explanation = QLabel(explanation_text)
        explanation.setFont(QFont("Inter", 11))
        explanation.setWordWrap(True)
        explanation.setStyleSheet(f"color: {TEXT_COLOR};")
        explanation_layout.addWidget(explanation)

        layout.addWidget(explanation_frame)

    def detect_code_language(self, code):
        """Detect the programming language of the code"""
        code_lower = code.lower()

        # Simple detection based on keywords and syntax
        if "import " in code_lower and ("def " in code_lower or "class " in code_lower):
            return "python"
        elif "function " in code_lower or "const " in code_lower or "let " in code_lower or "var " in code_lower:
            return "javascript"
        elif "package " in code_lower and "func " in code_lower:
            return "go"
        elif "#include " in code_lower or "int main" in code_lower:
            return "c++"
        elif "public class " in code_lower or "public static void main" in code_lower:
            return "java"
        elif "<?php" in code_lower:
            return "php"
        elif "<html" in code_lower or "<div" in code_lower:
            return "html"
        elif "@media" in code_lower or "{" in code_lower and ":" in code_lower:
            return "css"
        elif "apt " in code_lower or "yum " in code_lower or "sudo " in code_lower or code_lower.startswith("#!"):
            return "bash"
        elif "SELECT " in code.upper() or "INSERT INTO" in code.upper() or "UPDATE " in code.upper():
            return "sql"
        else:
            return "code"  # Generic fallback

    def copy_to_clipboard(self, text):
        """Copy text to clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

        # Visual feedback for copy (temporarily change button text)
        sender = self.sender()
        original_text = sender.text()
        sender.setText("Copied!")
        QTimer.singleShot(1500, lambda: sender.setText(original_text))

    def _contains_url(self, text: str) -> bool:
        """Check if text contains URLs"""
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        return bool(re.search(url_pattern, text))
    
    def _create_text_with_links(self, text: str, layout):
        """Create text with auto-detected URLs as copyable links"""
        import re
        
        # Pattern to detect URLs
        url_pattern = r'(https?://[^\s<>"{}|\\^`\[\]]+)'
        
        # Split text by URLs
        parts = re.split(url_pattern, text)
        
        for part in parts:
            if re.match(url_pattern, part):
                # This is a URL - make it copyable
                link_frame = QFrame()
                link_frame.setStyleSheet(f"""
                    QFrame {{
                        background-color: {LIGHT_GRAY};
                        border: 1px solid {BORDER_COLOR};
                        border-radius: 5px;
                        padding: 6px;
                        margin: 2px 0px;
                    }}
                """)
                hbox = QHBoxLayout(link_frame)
                hbox.setContentsMargins(8, 4, 8, 4)
                
                # URL label (shortened for display)
                display_url = part if len(part) <= 50 else part[:47] + "..."
                link_label = QLabel(display_url)
                link_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                link_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-family: monospace;")
                link_label.setFont(QFont("Courier New", 11))
                hbox.addWidget(link_label)
                
                hbox.addStretch()  # Push copy button to the right
                
                copy_btn = QPushButton("Copy")
                copy_btn.setFixedSize(60, 24)
                copy_btn.setFont(QFont("Inter", 9))
                copy_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ACCENT_COLOR};
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 2px 6px;
                    }}
                    QPushButton:hover {{
                        background-color: #FF5722;
                    }}
                """)
                copy_btn.clicked.connect(lambda _, url=part: QApplication.clipboard().setText(url))
                hbox.addWidget(copy_btn)
                
                layout.addWidget(link_frame)
            elif part.strip():  # Non-URL text (ignore empty parts)
                text_label = QLabel(part)
                text_label.setFont(QFont("Inter", 13))
                text_label.setWordWrap(True)
                text_label.setStyleSheet(f"color: {TEXT_COLOR};")
                layout.addWidget(text_label)

    def animate_entrance(self):
        """Animate the step entrance with a fade-in effect"""
        # Create opacity effect
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)

        # Create animation
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.start()


# === UPDATED CHATDIALOG WITH FIXED STYLING ===
class ChatDialog(QDialog):
    """Chat dialog with consistent styling and LLM-enhanced formatting"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle("Chat with Assistant")
        self.setMinimumSize(600, 700)
        
        # Use consistent styling with the rest of the app
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {PRIMARY_COLOR};
            }}
        """)

        # Enhanced defaults
        self.use_query_enhancement = True
        self.use_enhanced_retrieval = True
        self.use_multi_query = True
        self.capture_screen = False
        self.is_processing = False
        self.conversation_count = 0

        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Chat history display
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                background-color: {PRIMARY_COLOR};
                padding: 15px;
                font-family: Inter, Arial;
                font-size: 14px;
                color: {TEXT_COLOR};
                line-height: 1.5;
            }}
            QScrollBar:vertical {{
                background: {LIGHT_GRAY};
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {ACCENT_COLOR};
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: transparent;
            }}
        """)
        layout.addWidget(self.chat_history)

        # Settings section
        settings_container = QHBoxLayout()
        settings_container.setSpacing(15)

        # Screen capture toggle
        self.capture_toggle = QRadioButton("📸 Screen Capture")
        self.capture_toggle.setFont(QFont("Inter", 11))
        self.capture_toggle.setStyleSheet(f"""
            QRadioButton {{
                color: {TEXT_COLOR};
                background: transparent;
                spacing: 8px;
                padding: 4px;
            }}
            QRadioButton::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid {BORDER_COLOR};
                background: rgba(255, 255, 255, 0.05);
            }}
            QRadioButton::indicator:checked {{
                background: {ACCENT_COLOR};
                border: 2px solid {ACCENT_COLOR};
            }}
            QRadioButton::indicator:hover {{
                border: 2px solid {ACCENT_COLOR};
            }}
        """)
        self.capture_toggle.toggled.connect(self.toggle_screen_capture)
        settings_container.addWidget(self.capture_toggle)
        settings_container.addStretch()
        
        layout.addLayout(settings_container)

        # Message input container
        input_container = QFrame()
        input_container.setStyleSheet("background: transparent; border: none;")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)

        # Progress indicator
        self.progress_label = QLabel("Processing your question...")
        self.progress_label.setFont(QFont("Inter", 11))
        self.progress_label.setStyleSheet(f"""
            color: {ACCENT_COLOR}; 
            background: rgba(255, 59, 48, 0.1);
            border: 1px solid {ACCENT_COLOR};
            border-radius: 6px;
            padding: 6px 12px;
            font-style: italic;
        """)
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setVisible(False)
        input_layout.addWidget(self.progress_label)

        # Text input field
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Ask me anything about your training...")
        self.message_input.setFixedHeight(45)
        self.message_input.setFont(QFont("Inter", 13))
        self.message_input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 12px 15px;
                background-color: rgba(45, 45, 45, 0.6);
                color: {TEXT_COLOR};
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
                background-color: rgba(45, 45, 45, 0.8);
            }}
            QLineEdit::placeholder {{
                color: {SECONDARY_TEXT};
            }}
        """)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)

        layout.addWidget(input_container)

        # Button row
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch()

        # Send button
        self.send_btn = QPushButton("Send")
        self.send_btn.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedHeight(40)
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #FF5722;
            }}
            QPushButton:disabled {{
                background-color: {DISABLED_COLOR};
                color: rgba(255, 255, 255, 0.5);
            }}
        """)
        self.send_btn.clicked.connect(self.send_message)
        button_row.addWidget(self.send_btn)

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.setFont(QFont("Inter", 12))
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setFixedHeight(40)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(45, 45, 45, 0.8);
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 0 15px;
            }}
            QPushButton:hover {{
                background-color: {LIGHT_GRAY};
            }}
        """)
        self.close_btn.clicked.connect(self.close_chat)
        button_row.addWidget(self.close_btn)

        layout.addLayout(button_row)

        # Add initial welcome message
        self.add_assistant_message("Hello! I'm your AI learning assistant. I can help with questions about your training materials and provide guidance based on what you're learning.")

    def toggle_screen_capture(self, checked):
        """Toggle screen capture mode"""
        self.capture_screen = checked

    def capture_current_screen(self):
        """Capture current screen and return base64 encoded image"""
        try:
            if hasattr(self.parent_widget, 'screenshot_manager') and self.parent_widget.screenshot_manager.region:
                region = self.parent_widget.screenshot_manager.region
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
            
            from io import BytesIO
            buffer = BytesIO()
            screenshot.save(buffer, format='PNG')
            base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return f"data:image/png;base64,{base64_data}"
            
        except Exception as e:
            print(f"Error capturing screen: {e}")
            return None

    def add_conversation_header(self, number):
        """Add a simple conversation separator"""
        self.chat_history.append(
            f'<div style="margin: 25px 0 20px 0; text-align: center;">'
            f'<div style="display: inline-block; background-color: {ACCENT_COLOR}; '
            f'color: white; border-radius: 12px; padding: 6px 15px; font-weight: bold; font-size: 11px;">'
            f'Conversation {number}</div>'
            f'</div>'
        )
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def add_user_message(self, message):
        """Add user message with consistent styling"""
        capture_indicator = " 📸" if self.capture_screen else ""
        
        self.chat_history.append(
            f'<div style="background-color: {CARD_BACKGROUND}; padding: 15px; '
            f'border-radius: 8px; border-left: 4px solid {ACCENT_COLOR}; '
            f'margin: 8px 0; border: 1px solid {BORDER_COLOR};">'
            f'<div style="font-weight: bold; color: {ACCENT_COLOR}; margin-bottom: 8px; font-size: 13px;">You{capture_indicator}:</div>'
            f'<div style="color: {TEXT_COLOR}; font-size: 14px; line-height: 1.4;">{message}</div>'
            f'</div>'
        )
        
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

        if hasattr(self.parent_widget, 'add_chat_to_session'):
            self.parent_widget.add_chat_to_session('user', message)

    def add_assistant_message(self, message):
        """Add assistant message with consistent styling"""
        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.remove_thinking_indicator()

        # Enhanced message formatting
        formatted_message = self._format_message_with_llm(message)
        
        self.chat_history.append(
            f'<div style="background-color: {CARD_BACKGROUND}; padding: 15px; '
            f'border-radius: 8px; border-left: 4px solid {SECONDARY_COLOR}; '
            f'margin: 8px 0; border: 1px solid {BORDER_COLOR};">'
            f'<div style="font-weight: bold; color: {SECONDARY_COLOR}; margin-bottom: 8px; font-size: 13px;">AI Assistant:</div>'
            f'<div style="color: {TEXT_COLOR}; font-size: 14px; line-height: 1.6;">{formatted_message}</div>'
            f'</div>'
        )
        
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

        if hasattr(self.parent_widget, 'add_chat_to_session'):
            self.parent_widget.add_chat_to_session('assistant', message)

    def _format_message_with_llm(self, message):
        """Format message using LLM for better presentation"""
        try:
            import requests
            import os
            
            system_prompt = """Format this text for HTML display in a chat interface. 
            
            Rules:
            - Convert **bold** to <strong> tags with orange color
            - Convert `code` to <code> tags with monospace font and dark background
            - Convert ```code blocks``` to <pre> with proper styling
            - Convert lists to proper <ul>/<ol> tags
            - Keep line breaks as <br> tags
            - Make URLs clickable links
            - Use clean, readable formatting
            - Reduce excessive spacing
            
            Return only the formatted HTML, no explanation."""

            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                "max_tokens": 1000,
                "temperature": 0
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                },
                json=payload,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                formatted = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if formatted:
                    return formatted
            
            # Fallback to basic formatting
            return self._basic_format(message)
            
        except Exception as e:
            print(f"Error formatting with LLM: {e}")
            return self._basic_format(message)
    
    def _basic_format(self, message):
        """Basic formatting fallback"""
        import re
        
        # Basic markdown to HTML
        message = re.sub(r'\*\*(.*?)\*\*', rf'<strong style="color: {ACCENT_COLOR};">\1</strong>', message)
        message = re.sub(r'`([^`]+)`', rf'<code style="background: {LIGHT_GRAY}; padding: 2px 4px; border-radius: 3px; font-family: monospace;">\1</code>', message)
        message = re.sub(r'(https?://[^\s]+)', r'<a href="\1" style="color: #00D9FF;">\1</a>', message)
        message = message.replace('\n', '<br>')
        
        return message

    def add_sources_message(self, sources):
        """Add sources as a separate message"""
        if sources and len(sources) > 0:
            sources_content = f'<div style="color: {SECONDARY_COLOR}; font-weight: bold; margin-bottom: 6px; font-size: 12px;">📚 Sources:</div>'

            for source in sources:
                sources_content += f'<div style="margin: 2px 0; color: {SECONDARY_TEXT}; font-size: 12px;">📄 {source}</div>'
            
            self.chat_history.append(
                f'<div style="background-color: {CARD_BACKGROUND}; padding: 12px; '
                f'border-radius: 8px; border-left: 4px solid {SECONDARY_COLOR}; '
                f'margin: 4px 0; border: 1px solid {BORDER_COLOR};">'
                f'{sources_content}'
                f'</div>'
            )
            
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def add_thinking_indicator(self):
        """Add thinking indicator"""
        self.thinking_html = (
            f'<div id="thinking_indicator" style="background-color: {CARD_BACKGROUND}; padding: 15px; '
            f'border-radius: 8px; border-left: 4px solid {SECONDARY_COLOR}; '
            f'margin: 8px 0; border: 1px solid {BORDER_COLOR};">'
            f'<div style="font-weight: bold; color: {SECONDARY_COLOR}; margin-bottom: 8px; font-size: 13px;">AI Assistant:</div>'
            f'<div style="color: {SECONDARY_COLOR}; font-style: italic;">Thinking...</div>'
            f'<div id="dot-animation" style="color: {SECONDARY_COLOR}; margin-top: 4px;">●●●</div>'
            f'</div>'
        )

        self.chat_history.append(self.thinking_html)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())
        QApplication.processEvents()

        self.thinking_dots = 0
        self.thinking_timer = QTimer()
        self.thinking_timer.timeout.connect(self.animate_thinking)
        self.thinking_timer.start(500)

    def animate_thinking(self):
        """Simple thinking animation"""
        dot_patterns = ["●○○", "○●○", "○○●"]
        self.thinking_dots = (self.thinking_dots + 1) % len(dot_patterns)

        current_html = self.chat_history.toHtml()
        current_html = re.sub(
            r'<div id="dot-animation"[^>]*>[^<]*</div>',
            f'<div id="dot-animation" style="color: {SECONDARY_COLOR}; margin-top: 4px;">{dot_patterns[self.thinking_dots]}</div>',
            current_html
        )

        self.chat_history.setHtml(current_html)
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def remove_thinking_indicator(self):
        """Remove thinking indicator"""
        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.thinking_timer.stop()

        if hasattr(self, 'thinking_html'):
            current_html = self.chat_history.toHtml()
            current_html = re.sub(r'<div id="thinking_indicator".*?</div>\s*</div>', '', current_html, flags=re.DOTALL)
            self.chat_history.setHtml(current_html)

    def send_message(self):
        """Enhanced send_message with screen capture support"""
        if self.is_processing:
            return

        message = self.message_input.text().strip()
        if not message:
            return

        self.is_processing = True
        stored_message = message
        self.message_input.clear()

        # Increment conversation counter
        self.conversation_count += 1
        self.add_conversation_header(self.conversation_count)

        # Add user message
        self.add_user_message(stored_message)

        # Update UI
        self.message_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Processing...")
        self.progress_label.setVisible(True)
        self.add_thinking_indicator()

        try:
            # Capture screen if enabled
            screenshot_data = None
            if self.capture_screen:
                screenshot_data = self.capture_current_screen()

            # Get course_id
            course_id = self.parent_widget.course_id if hasattr(self.parent_widget, 'course_id') else "001"

            # Enhanced retrieval with screen capture support
            if screenshot_data:
                answer, sources = self._get_answer_with_image(stored_message, screenshot_data, course_id)
            else:
                answer, sources = self._get_standard_answer(stored_message, course_id)

            # Add assistant response
            self.add_assistant_message(answer)

            # Add sources if available
            if sources:
                self.add_sources_message(sources)

        except Exception as e:
            self.add_assistant_message(f"I encountered an error: {str(e)}. Please try again.")

        finally:
            # Reset processing state
            self.is_processing = False
            self.message_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send")
            self.progress_label.setVisible(False)

    def _get_answer_with_image(self, question, screenshot_data, course_id):
        """Get answer using LLM with image analysis"""
        try:
            from retriever import get_retriever_answer
            
            retrieval_result = get_retriever_answer(
                question=question,
                course_id=course_id,
                use_query_enhancement=True,
                use_enhanced_retrieval=True,
                use_multi_query=True,
                num_queries=3
            )
            
            context = retrieval_result.get("answer", "")
            sources = retrieval_result.get("source_documents", [])
            
            # Get the goal from parent widget
            goal = getattr(self.parent_widget, 'goal', 'No specific goal set')
            
            system_prompt = f"""You are an expert learning assistant with access to course materials and visual analysis capabilities.

The user's ultimate learning goal is: {goal}

Analyze the screenshot and provide helpful, specific guidance based on what you see, the course context, and how it relates to their goal.

Format your response in clear, helpful markdown with:
- Specific actionable steps
- Code examples if relevant (use ```language blocks)
- **Bold text** for important points
- Numbered lists for step-by-step instructions

If no course materials are available for the question, use your general knowledge to provide helpful guidance."""

            user_message = f"""
Question: {question}

Course Context: {context if context else "No specific course materials found for this question."}

Please analyze the screenshot and provide helpful guidance based on what you see and the question asked.
"""

            import requests
            import os
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_message},
                            {"type": "image_url", "image_url": {"url": screenshot_data}}
                        ]
                    }
                ],
                "max_tokens": 1500
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                },
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return answer, sources
            else:
                return self._get_standard_answer(question, course_id)

        except Exception as e:
            print(f"Error in image analysis: {e}")
            return self._get_standard_answer(question, course_id)

    def _get_standard_answer(self, question, course_id):
        """Get standard answer using enhanced retrieval"""
        try:
            from retriever import get_retriever_answer
            
            result = get_retriever_answer(
                question=question,
                course_id=course_id,
                use_query_enhancement=True,
                use_enhanced_retrieval=True,
                use_multi_query=True,
                num_queries=5
            )

            answer = result.get("answer", "")
            sources = result.get("source_documents", [])
            
            # If no good answer from materials, use general knowledge
            if not answer or "I don't have enough information" in answer:
                enhanced_answer = self._get_general_knowledge_answer(question)
                if enhanced_answer:
                    answer = enhanced_answer
                    sources = ["General Knowledge"]

            return answer, sources

        except Exception as e:
            print(f"Error in standard retrieval: {e}")
            return "I encountered an error while searching for information. Please try rephrasing your question.", []

    def _get_general_knowledge_answer(self, question):
        """Get answer using general knowledge when no course materials are found"""
        try:
            import requests
            import os

            # Get the goal from parent widget
            goal = getattr(self.parent_widget, 'goal', 'No specific goal set')

            system_prompt = f"""You are a helpful learning assistant. The user has asked a question but no specific course materials were found.

The user's ultimate learning goal is: {goal}

Provide a helpful, educational answer using your general knowledge that relates to their goal. Format your response in clear markdown with:
- **Bold text** for key concepts
- Numbered lists for step-by-step processes
- Code blocks with ```language when showing code
- Bullet points for features or options

Be comprehensive but concise, and focus on practical, actionable guidance that helps them progress toward their goal."""

            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}"}
                ],
                "max_tokens": 1000
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
                },
                json=payload,
                timeout=20
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return None

        except Exception as e:
            print(f"Error getting general knowledge answer: {e}")
            return None

    def closeEvent(self, event):
        """Override close event"""
        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.thinking_timer.stop()
        self.close_chat()
        event.accept()

    def close_chat(self):
        """Enhanced close handler"""
        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.thinking_timer.stop()
        self.hide()
        

class AnimatedLineEdit(QLineEdit):
    """QLineEdit with animated border effect"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation = QPropertyAnimation(self, b"")
        self.border_intensity = 0.3
        self.setup_animation()
        
    def setup_animation(self):
        """Setup the border animation"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_border)
        self.timer.start(50)  # Update every 50ms
        self.animation_progress = 0
        
    def update_border(self):
        """Update border animation"""
        self.animation_progress += 0.05
        if self.animation_progress > 1:
            self.animation_progress = 0
            
        # Create animated border intensity
        intensity = 0.3 + 0.4 * abs(0.5 - self.animation_progress)
        
        if self.hasFocus():
            border_color = f"rgba(255, 59, 48, {intensity})"
        else:
            border_color = f"rgba(255, 255, 255, {intensity * 0.3})"
            
        self.setStyleSheet(f"""
            QLineEdit {{
                background: rgba(45, 45, 45, 0.6);
                border: 2px solid {border_color};
                border-radius: 12px;
                padding: 15px 20px;
                color: {TEXT_COLOR};
                font-size: 14px;
            }}
            QLineEdit::placeholder {{
                color: {SECONDARY_TEXT};
            }}
        """)


class AnimatedTextEdit(QTextEdit):
    """QTextEdit with animated border effect"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_animation()
        
    def setup_animation(self):
        """Setup the border animation"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_border)
        self.timer.start(50)
        self.animation_progress = 0
        
    def update_border(self):
        """Update border animation"""
        self.animation_progress += 0.05
        if self.animation_progress > 1:
            self.animation_progress = 0
            
        intensity = 0.3 + 0.4 * abs(0.5 - self.animation_progress)
        
        if self.hasFocus():
            border_color = f"rgba(255, 59, 48, {intensity})"
        else:
            border_color = f"rgba(255, 255, 255, {intensity * 0.3})"
            
        self.setStyleSheet(f"""
            QTextEdit {{
                background: rgba(45, 45, 45, 0.6);
                border: 2px solid {border_color};
                border-radius: 12px;
                padding: 15px 20px;
                color: {TEXT_COLOR};
                font-size: 14px;
            }}
        """)


class LoginScreen(QWidget):
    """The login screen for the LearnChain Tutor app with modern dark theme"""

    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.init_ui()

    def init_ui(self):
        # Main container with dark background gradient
        self.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 {PRIMARY_COLOR},
                                          stop: 1 #0F0F0F);
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 80, 50, 80)
        layout.setSpacing(30)

        # Center everything in a glass card
        card = QFrame()
        card.setObjectName("glassCard")
        card.setStyleSheet(f"""
            #glassCard {{
                background: {CARD_BACKGROUND};
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 40px;
            }}
        """)
        
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(25)

        # App logo with accent color - FIXED FORMATTING
        logo_label = QLabel("LearnChain")
        logo_label.setFont(QFont("Inter", 36, QFont.Weight.Bold))  # Bigger
        logo_label.setStyleSheet(f"""
            color: {ACCENT_COLOR}; 
            margin: 0px; 
            padding: 0px;
            background: transparent;
        """)
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(logo_label)

        # Title with white text - FIXED FORMATTING
        title = QLabel("Training Wheels")
        title.setFont(QFont("Inter", 20, QFont.Weight.Normal))  # Bigger
        title.setStyleSheet(f"""
            color: {TEXT_COLOR}; 
            margin: 0px; 
            padding: 0px;
            background: transparent;
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(title)

        # Instruction text - FIXED FORMATTING
        instruction = QLabel("Please enter your credentials to continue")
        instruction.setFont(QFont("Inter", 15))  # Bigger
        instruction.setStyleSheet(f"""
            color: {SECONDARY_TEXT}; 
            margin: 0px; 
            padding: 0px;
            background: transparent;
        """)
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        card_layout.addWidget(instruction)

        # User input field with animated glass styling
        self.user_input = AnimatedLineEdit()
        self.user_input.setPlaceholderText("Enter your User ID")
        self.user_input.setFixedHeight(65)  # Slightly taller
        self.user_input.setFont(QFont("Inter", 15))  # Bigger font
        card_layout.addWidget(self.user_input)

        # Course ID input field with animated styling
        self.course_input = AnimatedLineEdit()
        self.course_input.setPlaceholderText("Enter your Course ID")
        self.course_input.setFixedHeight(65)  # Slightly taller
        self.course_input.setFont(QFont("Inter", 15))  # Bigger font
        card_layout.addWidget(self.course_input)

        # Login button with accent color and glass effect
        login_btn = QPushButton("Login")
        login_btn.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        login_btn.setFixedHeight(55)
        login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        login_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 {ACCENT_COLOR},
                                          stop: 1 #FF6B5A);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #FF5722,
                                          stop: 1 #FF8A65);
            }}
            QPushButton:pressed {{
                background: #E53E3E;
            }}
        """)
        login_btn.clicked.connect(self.handle_login)
        card_layout.addWidget(login_btn)

        # Add the card to main layout
        layout.addWidget(card)
        
        # Add stretching space
        layout.addStretch()
        
        self.setLayout(layout)

    def handle_login(self):
        """Handle login button click"""
        user_id = self.user_input.text().strip()
        course_id = self.course_input.text().strip()

        if user_id and course_id:
            # Pass user ID and course ID to main screen and switch
            self.stacked_widget.main_screen.set_user(user_id)
            self.stacked_widget.main_screen.set_course(course_id)
            self.stacked_widget.setCurrentIndex(1)
        elif user_id:
            # If only user ID is provided, still allow login but show a message
            self.stacked_widget.main_screen.set_user(user_id)
            self.stacked_widget.setCurrentIndex(1)

            
class GoalDisplay(QWidget):
    """Goal display with modern dark theme"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Remove any inherited styling that might interfere
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("""
            background-color: transparent;
            border: none;
        """)

        # Main layout with zero margins to maximize space
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Header section
        self.header_label = QLabel("Goal")
        self.header_label.setFixedHeight(40)
        self.header_label.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.header_label.setIndent(20)
        self.header_label.setStyleSheet(f"""
            background-color: {ACCENT_COLOR};
            color: white;
            padding-left: 10px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        """)
        main_layout.addWidget(self.header_label)

        # Content frame with glass effect
        self.content_frame = QFrame()
        self.content_frame.setMinimumHeight(60)
        self.content_frame.setStyleSheet(f"""
            background-color: rgba(45, 45, 45, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-top: none;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
        """)

        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(20, 15, 20, 15)

        # Goal text
        self.content = QLabel()
        self.content.setWordWrap(True)
        self.content.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.content.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        self.content.setStyleSheet(f"""
            color: {TEXT_COLOR}; 
            padding-left: 5px;
            padding-top: 5px;
            padding-bottom: 5px;
        """)

        content_layout.addWidget(self.content)
        main_layout.addWidget(self.content_frame)

    def set_goal(self, text):
        """Set goal text"""
        self.content.setText(text)
        self.content.adjustSize()


class ToggleSwitch(QWidget):
    """Custom toggle switch widget with modern styling"""

    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Create switch track with updated colors
        self.track = QFrame(self)
        self.track.setFixedSize(50, 26)
        self.track.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(85, 85, 85, 0.8);
                border-radius: 13px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
        """)

        # Create handle with updated styling
        self.handle = QFrame(self.track)
        self.handle.setFixedSize(22, 22)
        self.handle.move(2, 2)
        self.handle.setStyleSheet(f"""
            QFrame {{
                background-color: white;
                border-radius: 11px;
                border: none;
            }}
        """)

        layout.addWidget(self.track)

        # State
        self.is_checked = False

        # Make clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(26)

        # Animation for handle
        self.animation = QPropertyAnimation(self.handle, b"pos")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def mousePressEvent(self, event):
        """Handle mouse click to toggle switch"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()

    def toggle(self):
        """Toggle the switch state with updated styling"""
        self.is_checked = not self.is_checked

        # Animate handle position
        if self.is_checked:
            self.animation.setStartValue(self.handle.pos())
            self.animation.setEndValue(QPoint(self.track.width() - self.handle.width() - 2, 2))
            self.track.setStyleSheet(f"""
                QFrame {{
                    background-color: {ACCENT_COLOR};
                    border-radius: 13px;
                    border: 1px solid {ACCENT_COLOR};
                }}
            """)
        else:
            self.animation.setStartValue(self.handle.pos())
            self.animation.setEndValue(QPoint(2, 2))
            self.track.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba(85, 85, 85, 0.8);
                    border-radius: 13px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
            """)

        self.animation.start()

        # Emit signal
        self.toggled.emit(self.is_checked)

    def setChecked(self, checked):
        """Set the state programmatically"""
        if self.is_checked != checked:
            self.toggle()
        elif checked:  # Update styling if already in the correct state
            self.track.setStyleSheet(f"""
                QFrame {{
                    background-color: {ACCENT_COLOR};
                    border-radius: 13px;
                    border: 1px solid {ACCENT_COLOR};
                }}
            """)
            self.handle.move(self.track.width() - self.handle.width() - 2, 2)
        else:
            self.track.setStyleSheet(f"""
                QFrame {{
                    background-color: rgba(85, 85, 85, 0.8);
                    border-radius: 13px;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                }}
            """)
            self.handle.move(2, 2)

    def isChecked(self):
        """Get the current state"""
        return self.is_checked


class MainScreen(QWidget):
    """The main screen showing the active tutor session"""
    guidance_received = pyqtSignal(dict)

    def __init__(self, width=300, height=600):
        super().__init__()
        self.user_id = None
        self.course_id = None
        self.goal = None
        self.step_index = 1
        self.steps = []
        self.is_session_active = False
        self.is_session_paused = False
        self.explanation_mode = False
        self.is_online = False  # Initialize as offline

        # Create enhanced chat dialog
        self.chat_dialog = ChatDialog(self)

        self.setMinimumSize(width, height)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {PRIMARY_COLOR};
            }}
        """)
        
        icon_path = str(resource_path("assets/icon.ico"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.init_ui()
        
        # Check S3 connection to set online status
        self.check_s3_connection()
        
        # Update the online status display
        QTimer.singleShot(100, self.update_online_status)  # Small delay to ensure UI is ready
        
        try:
            import retriever
            logger.info("🚀 Starting preemptive initialization on app startup...")
            ready = retriever.wait_for_full_initialization(timeout=2.0)
            if ready:
                logger.info("✅ Preemptive initialization complete!")
            else:
                logger.info("⏳ Preemptive initialization in progress...")
        except Exception as e:
            logger.error(f"❌ Preemptive initialization error: {e}")

    def check_s3_connection(self):
        """Check S3 connection and update online status"""
        try:
            import training_wheels as tw
            result = tw.test_s3_connection()
            self.is_online = result.get("connected", False)
        except Exception as e:
            logger.error(f"Error checking S3 connection: {e}")
            self.is_online = False

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # ===== Header section =====
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(30, 30, 30, 0.8);
                border-bottom: 1px solid {BORDER_COLOR};
            }}
        """)
        header.setFixedHeight(60)

        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)

        # App logo/name
        app_name = QLabel("LearnChain Tutor")
        app_name.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        app_name.setStyleSheet(f"color: {ACCENT_COLOR};")
        header_layout.addWidget(app_name)

        # User welcome on the right
        self.user_welcome = QLabel()
        self.user_welcome.setFont(QFont("Inter", 12))
        self.user_welcome.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.user_welcome.setStyleSheet(f"color: {TEXT_COLOR};")
        header_layout.addWidget(self.user_welcome)

        self.layout.addWidget(header)

        # ===== Goal input section =====
        self.goal_input_section = QFrame()
        self.goal_input_section.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                                          stop: 0 {PRIMARY_COLOR},
                                          stop: 1 #0F0F0F);
            }}
        """)

        goal_layout = QVBoxLayout(self.goal_input_section)
        goal_layout.setContentsMargins(50, 50, 50, 50)
        goal_layout.setSpacing(30)

        # Glass card container
        goal_card = QFrame()
        goal_card.setObjectName("goalCard")
        goal_card.setStyleSheet(f"""
            #goalCard {{
                background: {CARD_BACKGROUND};
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
                padding: 40px;
            }}
        """)

        card_layout = QVBoxLayout(goal_card)
        card_layout.setSpacing(25)

        # Hero text with accent highlight - FIXED FORMATTING AND BACKGROUND
        hero_container = QVBoxLayout()
        hero_container.setSpacing(8)  # Proper spacing between lines
        hero_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hero_line1 = QLabel("What do you want")
        hero_line1.setFont(QFont("Inter", 32, QFont.Weight.Bold))  # Bigger font
        hero_line1.setStyleSheet(f"""
            color: {TEXT_COLOR}; 
            margin: 0px; 
            padding: 0px;
            background: transparent;  /* Remove any background */
        """)
        hero_line1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hero_line2 = QLabel()
        hero_line2.setText(f'to <span style="color: {ACCENT_COLOR};">master</span> today?')
        hero_line2.setFont(QFont("Inter", 32, QFont.Weight.Bold))  # Bigger font
        hero_line2.setStyleSheet(f"""
            color: {TEXT_COLOR}; 
            margin: 0px; 
            padding: 0px;
            background: transparent;  /* Remove any background */
        """)
        hero_line2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        hero_container.addWidget(hero_line1)
        hero_container.addWidget(hero_line2)
        card_layout.addLayout(hero_container)

        # Goal input - make it scrollable with animated border
        self.goal_input = AnimatedTextEdit()
        self.goal_input.setPlaceholderText("e.g., Deploy a Django site on Lightsail")
        self.goal_input.setMaximumHeight(120)
        self.goal_input.setMinimumHeight(90)   # Slightly taller minimum
        self.goal_input.setFont(QFont("Inter", 15))  # Bigger font
        card_layout.addWidget(self.goal_input)

        # Settings toggles in glass container - FIXED VISIBILITY AND ALIGNMENT
        toggles_frame = QFrame()
        toggles_frame.setStyleSheet(f"""
            QFrame {{
                background: rgba(45, 45, 45, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 12px;
                padding: 20px;
            }}
        """)
        toggles_frame.setFixedHeight(140)  # Increased height for better spacing

        toggles_layout = QVBoxLayout(toggles_frame)  
        toggles_layout.setContentsMargins(20, 20, 20, 20)
        toggles_layout.setSpacing(20)

        # First row of toggles with improved alignment
        first_row = QHBoxLayout()
        first_row.setSpacing(60)  # More space between toggles

        # Include Explanations toggle - FIXED: Properly centered
        explanation_container = QVBoxLayout()
        explanation_container.setSpacing(12)
        explanation_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        explanation_label = QLabel("Include Explanations")
        explanation_label.setFont(QFont("Inter", 14, QFont.Weight.Medium))  # Bigger font
        explanation_label.setStyleSheet(f"color: {TEXT_COLOR}; padding: 0px; margin: 0px;")
        explanation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the text
        explanation_container.addWidget(explanation_label)

        self.explanation_toggle = ToggleSwitch()
        self.explanation_toggle.toggled.connect(self.toggle_explanation_mode)
        explanation_container.addWidget(self.explanation_toggle, 0, Qt.AlignmentFlag.AlignCenter)

        # Web Search toggle - FIXED: Properly centered  
        search_container = QVBoxLayout()
        search_container.setSpacing(12)
        search_container.setAlignment(Qt.AlignmentFlag.AlignCenter)

        search_label = QLabel("Web Search")
        search_label.setFont(QFont("Inter", 14, QFont.Weight.Medium))  # Bigger font
        search_label.setStyleSheet(f"color: {SECONDARY_TEXT}; padding: 0px; margin: 0px;")
        search_label.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the text
        search_container.addWidget(search_label)

        search_toggle = ToggleSwitch()
        search_toggle.setEnabled(False)
        search_container.addWidget(search_toggle, 0, Qt.AlignmentFlag.AlignCenter)

        first_row.addLayout(explanation_container)
        first_row.addLayout(search_container)
        first_row.addStretch()
        toggles_layout.addLayout(first_row)
        card_layout.addWidget(toggles_frame)

        # Fullscreen notification
        fullscreen_notification = QFrame()
        fullscreen_notification.setStyleSheet(f"""
            QFrame {{
                background: rgba(255, 59, 48, 0.15);
                border: 1px solid {ACCENT_COLOR};
                border-radius: 8px;
                padding: 12px;
            }}
        """)

        notification_layout = QHBoxLayout(fullscreen_notification)
        notification_layout.setSpacing(10)

        # Bell icon
        bell_icon = QLabel("🔔")
        bell_icon.setFont(QFont("Inter", 16))
        notification_layout.addWidget(bell_icon)

        notification_text = QLabel("Fullscreen mode enabled")
        notification_text.setFont(QFont("Inter", 13, QFont.Weight.Medium))
        notification_text.setStyleSheet(f"color: {TEXT_COLOR};")
        notification_layout.addWidget(notification_text)
        notification_layout.addStretch()

        card_layout.addWidget(fullscreen_notification)

        # Start Session button
        start_btn = QPushButton("Start Session")
        start_btn.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        start_btn.setFixedHeight(55)
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 {ACCENT_COLOR},
                                          stop: 1 #FF6B5A);
                color: white;
                border: none;
                border-radius: 12px;
                font-weight: bold;
                margin-top: 15px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                                          stop: 0 #FF5722,
                                          stop: 1 #FF8A65);
            }}
            QPushButton:pressed {{
                background: #E53E3E;
            }}
        """)
        start_btn.clicked.connect(self.start_session)
        card_layout.addWidget(start_btn)

        # FIXED: Centered online status with S3 connection logic
        status_container = QHBoxLayout()
        status_container.addStretch()  # Center the status
        
        self.status_dot = QLabel("●")
        self.status_dot.setFont(QFont("Inter", 16))
        
        self.status_text = QLabel()
        self.status_text.setFont(QFont("Inter", 13))
        
        # Set initial status based on S3 connection
        self.update_online_status()
        
        status_container.addWidget(self.status_dot)
        status_container.addWidget(self.status_text)
        status_container.addStretch()  # Center the status
        
        card_layout.addLayout(status_container)

        goal_layout.addWidget(goal_card)
        goal_layout.addStretch()

        self.layout.addWidget(self.goal_input_section)

        # ===== Enhanced Goal display (shown after session starts) =====
        self.goal_display_container = QFrame()
        self.goal_display_container.setStyleSheet(f"""
            background-color: {PRIMARY_COLOR};
            padding: 0px;
        """)
        self.goal_display_container.setVisible(False)

        goal_container_layout = QVBoxLayout(self.goal_display_container)
        goal_container_layout.setContentsMargins(20, 15, 20, 15)
        goal_container_layout.setSpacing(0)

        # Custom goal display
        self.goal_display = GoalDisplay()
        goal_container_layout.addWidget(self.goal_display)

        self.layout.addWidget(self.goal_display_container)

        # ===== Instruction area (steps) =====
        self.instruction_container = QFrame()
        self.instruction_container.setObjectName("instructionContainer")
        self.instruction_container.setStyleSheet(f"""
            #instructionContainer {{
                background-color: {PRIMARY_COLOR};
                border: none;
            }}
        """)

        instruction_layout = QVBoxLayout(self.instruction_container)
        instruction_layout.setContentsMargins(0, 0, 0, 0)
        instruction_layout.setSpacing(0)

        self.instruction_area = QScrollArea()
        self.instruction_area.setWidgetResizable(True)
        self.instruction_area.setFrameShape(QFrame.Shape.NoFrame)
        self.instruction_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.instruction_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: {PRIMARY_COLOR};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {LIGHT_GRAY};
                width: 8px;
                margin: 0px;
            }}
            QScrollBar::handle:vertical {{
                background: {ACCENT_COLOR};
                min-height: 20px;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)

        # Container for steps
        self.step_container = QWidget()
        self.step_layout = QVBoxLayout(self.step_container)
        self.step_layout.setContentsMargins(15, 15, 15, 15)
        self.step_layout.setSpacing(10)
        self.step_layout.addStretch()

        self.instruction_area.setWidget(self.step_container)
        instruction_layout.addWidget(self.instruction_area)

        self.layout.addWidget(self.instruction_container, 1)

        # ===== Button controls =====
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(30, 30, 30, 0.8);
                border-top: 1px solid {BORDER_COLOR};
            }}
        """)
        self.button_panel.setFixedHeight(80)
        self.button_panel.setVisible(False)

        button_layout = QHBoxLayout(self.button_panel)
        button_layout.setContentsMargins(15, 15, 15, 15)
        button_layout.setSpacing(10)

        # Chat button - only enabled when paused
        self.chat_btn = QPushButton()
        self.chat_btn.setFixedSize(50, 50)
        self.chat_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.chat_btn.setIcon(QIcon(str(resource_path("icons/chat.png"))))
        self.chat_btn.setIconSize(QSize(24, 24))
        self.chat_btn.setToolTip("Ask a question (only available when session is paused)")
        self.chat_btn.setEnabled(False)
        self.chat_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(45, 45, 45, 0.6);
                border: 1px solid {BORDER_COLOR};
                border-radius: 25px;
            }}
            QPushButton:hover {{
                background-color: {LIGHT_GRAY};
                border: 1px solid {ACCENT_COLOR};
            }}
            QPushButton:disabled {{
                background-color: {DISABLED_COLOR};
                border: 1px solid {BORDER_COLOR};
                opacity: 0.5;
            }}
        """)
        self.chat_btn.clicked.connect(self.open_chat)
        button_layout.addWidget(self.chat_btn)

        # Add spacer to push buttons to the right
        button_layout.addStretch()

        # Pause button
        self.pause_btn = QPushButton()
        self.pause_btn.setFixedSize(50, 50)
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.setIcon(QIcon(str(resource_path("icons/pause.png"))))
        self.pause_btn.setIconSize(QSize(24, 24))
        self.pause_btn.setToolTip("Pause session")
        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BUTTON_ORANGE};
                color: white;
                border: none;
                border-radius: 25px;
            }}
            QPushButton:hover {{
                background-color: #FF5722;
            }}
        """)
        self.pause_btn.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_btn)

        # Stop button
        self.stop_btn = QPushButton()
        self.stop_btn.setFixedSize(50, 50)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setIcon(QIcon(str(resource_path("icons/stop.png"))))
        self.stop_btn.setIconSize(QSize(24, 24))
        self.stop_btn.setToolTip("Stop session")
        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BUTTON_RED};
                color: white;
                border: none;
                border-radius: 25px;
            }}
            QPushButton:hover {{
                background-color: #dc2626;
            }}
        """)
        self.stop_btn.clicked.connect(self.stop_session)
        button_layout.addWidget(self.stop_btn)

        # Next button
        self.next_btn = QPushButton()
        self.next_btn.setFixedHeight(50)
        self.next_btn.setMinimumWidth(140)
        self.next_btn.setText("Next Step")
        self.next_btn.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        self.next_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.next_btn.setIcon(QIcon(str(resource_path("icons/next.png"))))
        self.next_btn.setIconSize(QSize(20, 20))
        self.next_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BUTTON_BLUE};
                color: white;
                border: none;
                border-radius: 25px;
                padding: 0 20px 0 15px;
                text-align: right;
            }}
            QPushButton:hover {{
                background-color: #FF5722;
            }}
            QPushButton:disabled {{
                background-color: {DISABLED_COLOR};
            }}
        """)
        self.next_btn.clicked.connect(self.next_step)
        button_layout.addWidget(self.next_btn)

        self.layout.addWidget(self.button_panel)

    # Keep all other existing methods unchanged...
    def update_online_status(self):
        """Update the online/offline status display"""
        if self.is_online:
            self.status_dot.setStyleSheet(f"color: {SECONDARY_COLOR};")
            self.status_text.setText("Online")
            self.status_text.setStyleSheet(f"color: {SECONDARY_TEXT};")
        else:
            self.status_dot.setStyleSheet(f"color: {BUTTON_RED};")
            self.status_text.setText("Offline")
            self.status_text.setStyleSheet(f"color: {SECONDARY_TEXT};")

    def set_user(self, user_id):
        """Set the user ID and update welcome message"""
        self.user_id = user_id
        self.user_welcome.setText(f"Welcome, {user_id}")

    def set_course(self, course_id):
        """Set the course ID"""
        self.course_id = course_id

    def toggle_explanation_mode(self, enabled):
        """Toggle explanation mode on/off"""
        self.explanation_mode = enabled

        # Update UI
        if hasattr(self, 'explanation_toggle'):
            self.explanation_toggle.setChecked(enabled)

        # Update training wheels if active
        if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
            import training_wheels as tw
            tw.toggle_explanation_mode(enabled)

    def start_session(self):
        """Start a new training session"""
        goal = self.goal_input.toPlainText().strip()  # Use toPlainText() for QTextEdit
        if goal:
            # Save goal and update UI state
            self.goal = goal
            self.step_index = 1
            self.is_session_active = True
            self.is_session_paused = False

            # Get explanation mode state
            self.explanation_mode = self.explanation_toggle.isChecked()

            # Update UI visibility
            self.goal_input_section.setVisible(False)
            self.goal_display_container.setVisible(True)
            self.button_panel.setVisible(True)

            # Set goal text
            self.goal_display.set_goal(goal)

            # Reset instruction area background
            self.instruction_container.setStyleSheet(f"""
                #instructionContainer {{
                    background-color: {PRIMARY_COLOR};
                }}
            """)

            # Clear any previous steps
            self.clear_steps()

    def clear_steps(self):
        """Clear all steps from the instruction area"""
        # Remove all widgets except the stretch at the end
        while self.step_layout.count() > 1:
            item = self.step_layout.itemAt(0)
            if item is not None:
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
                    widget.deleteLater()

        self.steps = []

    def add_instruction(self, content):
        """Add a new instruction step with proper step numbering"""
        # Skip system steps in step numbering display
        is_system_step = isinstance(content, dict) and content.get('system_step', False)
        if is_system_step:
            display_step_index = None
        else:
            display_step_index = self.step_index

        # Reset style on previous current step (if any)
        for step in self.steps:
            step.setStyleSheet(f"""
                #stepItem {{
                    background-color: rgba(30, 30, 30, 0.4);
                    border-left: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                    margin: 4px 0px;
                }}
            """)

        # Create new step (current)
        new_step = AnimatedStep(
            display_step_index if display_step_index is not None else "i",
            content,
            True
        )

        # Add to layout before the stretch
        self.step_layout.insertWidget(self.step_layout.count() - 1, new_step)

        # Save reference and increment counter (only for non-system steps)
        self.steps.append(new_step)
        if not is_system_step:
            self.step_index += 1

        # Scroll to make the new step visible
        QTimer.singleShot(100, lambda: self.instruction_area.ensureWidgetVisible(new_step))
        # Force scroll to bottom after adding new step
        QTimer.singleShot(100, lambda: self.instruction_area.verticalScrollBar().setValue(
            self.instruction_area.verticalScrollBar().maximum()
        ))

    def add_chat_to_session(self, role, content):
        """Add chat interaction to training wheels session"""
        if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
            import training_wheels as tw
            tw.add_chat_interaction(role, content)

    def next_step(self):
        """Process the next instruction step"""
        if not self.is_session_active or self.is_session_paused:
            return

        # In a real implementation, this would get the next step from your LLM
        # For demonstration, we'll alternate between text and code examples

        if self.step_index % 2 == 0:
            # Text instruction example
            next_instruction = {
                'instruction': "Simulate LLM instruction: do the next logical step",
                'format': 'text'
            }
        else:
            # Code instruction example
            next_instruction = {
                'instruction': '# Example code for next step\ndef process_data(input_file):\n    with open(input_file, "r") as f:\n        data = f.read()\n    \n    # Process the data\n    result = data.upper()\n    \n    return result',
                'format': 'code'
            }

        self.add_instruction(next_instruction)

    def toggle_pause(self):
        """Toggle between pause and resume states"""
        if not self.is_session_active:
            return

        self.is_session_paused = not self.is_session_paused

        if self.is_session_paused:
            # Change to play/resume icon
            self.pause_btn.setIcon(QIcon(str(resource_path("icons/play.png"))))
            self.pause_btn.setToolTip("Resume session")
            self.pause_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {SECONDARY_COLOR};
                    color: white;
                    border: none;
                    border-radius: 25px;
                }}
                QPushButton:hover {{
                    background-color: #059669;
                }}
            """)

            # Disable next button
            self.next_btn.setEnabled(False)
            self.next_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DISABLED_COLOR};
                    color: white;
                    border: none;
                    border-radius: 25px;
                    padding: 0 20px 0 15px;
                    text-align: right;
                }}
            """)

            # Enable chat button
            self.chat_btn.setEnabled(True)
            self.chat_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: rgba(45, 45, 45, 0.6);
                    border: 1px solid {ACCENT_COLOR};
                    border-radius: 25px;
                }}
                QPushButton:hover {{
                    background-color: {LIGHT_GRAY};
                    border: 1px solid {ACCENT_COLOR};
                }}
            """)

            # Apply paused background
            self.step_container.setStyleSheet(f"""
                background-color: {PAUSED_BG_COLOR};
            """)

            self.instruction_area.setStyleSheet(f"""
                QScrollArea {{
                    background-color: {PAUSED_BG_COLOR};
                    border: none;
                }}
                QScrollBar:vertical {{
                    background: {LIGHT_GRAY};
                    width: 8px;
                    margin: 0px;
                }}
                QScrollBar::handle:vertical {{
                    background: {ACCENT_COLOR};
                    min-height: 20px;
                    border-radius: 4px;
                }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                    height: 0px;
                }}
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                    background: none;
                }}
            """)

            self.instruction_container.setStyleSheet(f"""
                #instructionContainer {{
                    background-color: {PAUSED_BG_COLOR};
                }}
            """)
        else:
            # Change back to pause icon
            self.pause_btn.setIcon(QIcon(str(resource_path("icons/pause.png"))))
            self.pause_btn.setToolTip("Pause session")
            self.pause_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BUTTON_ORANGE};
                    color: white;
                    border: none;
                    border-radius: 25px;
                }}
                QPushButton:hover {{
                    background-color: #FF5722;
                }}
            """)

            # Enable next button
            self.next_btn.setEnabled(True)
            self.next_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {BUTTON_BLUE};
                    color: white;
                    border: none;
                    border-radius: 25px;
                    padding: 0 20px 0 15px;
                    text-align: right;
                }}
                QPushButton:hover {{
                    background-color: #FF5722;
                }}
            """)

            # Disable chat button
            self.chat_btn.setEnabled(False)
            self.chat_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {DISABLED_COLOR};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 25px;
                    opacity: 0.5;
                }}
            """)

            # Reset backgrounds
            self.step_container.setStyleSheet("")

            self.instruction_area.setStyleSheet(f"""
                QScrollArea {{
                    background-color: {PRIMARY_COLOR};
                    border: none;
                }}
                QScrollBar:vertical {{
                    background: {LIGHT_GRAY};
                    width: 8px;
                    margin: 0px;
                }}
                QScrollBar::handle:vertical {{
                    background: {ACCENT_COLOR};
                    min-height: 20px;
                    border-radius: 4px;
                }}
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                    height: 0px;
                }}
                QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                    background: none;
                }}
            """)

            self.instruction_container.setStyleSheet(f"""
                #instructionContainer {{
                    background-color: {PRIMARY_COLOR};
                }}
            """)

    def stop_session(self):
        """Stop the current session and return to goal input"""
        # Update state
        self.is_session_active = False
        self.is_session_paused = False

        # Update UI
        self.goal_input_section.setVisible(True)
        self.goal_display_container.setVisible(False)
        self.button_panel.setVisible(False)

        # Clear input and steps
        self.goal_input.clear()
        self.clear_steps()

        # Reset instruction area background
        self.instruction_container.setStyleSheet(f"""
            #instructionContainer {{
                background-color: {PRIMARY_COLOR};
            }}
        """)

    def open_chat(self):
        """Open the enhanced chat dialog for asking questions with the current course ID"""
        if self.is_session_paused:
            # Ensure the dialog has access to the course_id
            if not hasattr(self.chat_dialog, 'course_id') and hasattr(self, 'course_id'):
                self.chat_dialog.course_id = self.course_id

            # Display a message if course_id is not set
            if not hasattr(self, 'course_id') or not self.course_id:
                self.chat_dialog.add_system_message("Warning: No course ID is set. Using default course.")
                self.chat_dialog.course_id = "001"

            self.chat_dialog.show()


# Apply UI integration
MainScreen = ui_integration.integrate_training_wheels(MainScreen)


class LearnChainTutor(QStackedWidget):
    """Main application container"""

    def __init__(self):
        super().__init__()

        # Set window icon
        icon_path = str(resource_path("assets/icon.ico"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            print(f"✅ LearnChainTutor icon set: {icon_path}")
        else:
            print(f"❌ LearnChainTutor icon not found: {icon_path}")

        # Initialize screens
        self.login_screen = LoginScreen(self)
        self.main_screen = MainScreen()

        # Add to stacked widget
        self.addWidget(self.login_screen)
        self.addWidget(self.main_screen)

        # Start with login screen
        self.setCurrentIndex(0)

        # Set initial size
        self.resize(650, 850)

        # Set window title
        self.setWindowTitle("LearnChain Tutor")

        # Apply global dark theme styles
        self.setStyleSheet(f"""
            QWidget {{
                font-family: Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: {PRIMARY_COLOR};
                color: {TEXT_COLOR};
            }}

            QPushButton {{
                border-radius: 4px;
                padding: 8px 16px;
            }}

            QScrollBar:vertical {{
                background: {LIGHT_GRAY};
                width: 10px;
                margin: 0px;
            }}

            QScrollBar::handle:vertical {{
                background: {ACCENT_COLOR};
                min-height: 20px;
                border-radius: 5px;
            }}

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}

            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: none;
            }}
        """)

    def showEvent(self, event):
        """Override showEvent to set icon after window is shown"""
        super().showEvent(event)
        icon_path = str(resource_path("assets/icon.ico"))
        if os.path.exists(icon_path):
            # Force set taskbar icon after window is visible
            QTimer.singleShot(100, lambda: set_taskbar_icon(self, icon_path))


def register_custom_protocol():
    protocol = "learnchn"
    exe_path = os.path.abspath(sys.argv[0])

    try:
        key = winreg.CreateKey(winreg.HKEY_CLASSES_ROOT, protocol)
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, "URL:LearnChain Protocol")
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")

        command_key = winreg.CreateKey(key, r"shell\open\command")
        winreg.SetValueEx(command_key, None, 0, winreg.REG_SZ, f'"{exe_path}" "%1"')

        print(f"✅ Protocol '{protocol}://' registered.")
    except Exception as e:
        print(f"❌ Protocol registration failed: {e}")


def test_icon_file():
    """Test if icon file is valid"""
    # Test both possible paths
    paths_to_try = [
        str(resource_path("assets/icon.ico")),
        str(resource_path("training_wheels/assets/icon.ico")),
        "training_wheels/assets/icon.ico",
        "assets/icon.ico"
    ]

    for icon_path in paths_to_try:
        print(f"🔍 Testing icon path: {icon_path}")
        if os.path.exists(icon_path):
            file_size = os.path.getsize(icon_path)
            print(f"✅ Icon found! Size: {file_size} bytes")
            return icon_path

    print("❌ No icon file found in any location")
    return None


if __name__ == "__main__":
    # Test icon first
    icon_path = test_icon_file()

    # Check for admin rights for protocol registration
    if not is_admin():
        print("⚠️ Protocol registration requires administrator privileges")
        if not run_as_admin():
            print("✅ Restarting as administrator...")
            sys.exit(0)  # Exit current instance, admin instance will start
        else:
            print("⚠️ Continuing without admin rights (protocol registration will fail)")
    else:
        print("✅ Running as administrator")

    # Register protocol (will succeed if admin, fail gracefully if not)
    register_custom_protocol()

    # Set application ID BEFORE creating QApplication
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('LearnChainTutor.Application.1.0')
        print("✅ App ID set")
    except Exception as e:
        print(f"❌ Failed to set App ID: {e}")

    # Enable high DPI scaling for PyQt6
    try:
        if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    except:
        pass

    app = QApplication([])

    # Set application properties
    app.setApplicationName("LearnChain Tutor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("LearnChain")

    # Set application icon MULTIPLE ways
    if icon_path and os.path.exists(icon_path):
        icon = QIcon(icon_path)
        app.setWindowIcon(icon)
        QApplication.setWindowIcon(icon)
        print(f"✅ App icon set from: {icon_path}")
    else:
        print(f"❌ Icon not found: {icon_path}")

    QApplication.setStyle("Fusion")

    # Apply dark theme to the entire application
    app.setStyleSheet(f"""
        QWidget {{
            background-color: {PRIMARY_COLOR};
            color: {TEXT_COLOR};
            font-family: Inter, sans-serif;
            font-size: 13px;
        }}
        QLineEdit {{
            background-color: rgba(45, 45, 45, 0.6);
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
            padding: 8px;
            color: {TEXT_COLOR};
        }}
        QLineEdit:focus {{
            border: 1px solid {ACCENT_COLOR};
        }}
        QTextEdit {{
            background-color: rgba(45, 45, 45, 0.6);
            border: 1px solid {BORDER_COLOR};
            border-radius: 6px;
            padding: 8px;
            color: {TEXT_COLOR};
        }}
        QTextEdit:focus {{
            border: 1px solid {ACCENT_COLOR};
        }}
    """)

    # Use environment variable to check for integrated mode
    integrated_mode = os.environ.get('INTEGRATED_MODE', 'false').lower() == 'true'

    if integrated_mode:
        window = MainScreen()
        window.show()
        # Force set icon after showing
        if icon_path and os.path.exists(icon_path):
            QTimer.singleShot(200, lambda: set_taskbar_icon(window, icon_path))
    else:
        window = LearnChainTutor()
        window.show()
        # Force set icon after showing
        if icon_path and os.path.exists(icon_path):
            QTimer.singleShot(200, lambda: set_taskbar_icon(window, icon_path))

    app.exec()

    # Trigger preemptive initialization IMMEDIATELY
    try:
        import retriever
        print("🚀 ULTRA-FAST MODE: Starting preemptive initialization...")
        # Start but don't wait - let it run in background
    except Exception as e:
        print(f"❌ Preemptive initialization failed: {e}")