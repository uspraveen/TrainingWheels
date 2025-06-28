# learnchain_tutor.py

from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QStackedWidget, QScrollArea, QFrame, QHBoxLayout,
    QSizePolicy, QGraphicsOpacityEffect, QSpacerItem, QTextEdit,
    QSplitter, QDialog, QDialogButtonBox, QCheckBox, QSlider,
    QComboBox, QFileDialog, QProgressDialog, QMessageBox
)
from PyQt6.QtGui import QFont, QPixmap, QColor, QIcon, QFontDatabase, QPainter, QPen, QPainterPath
from PyQt6.QtCore import (
    Qt, QPropertyAnimation, pyqtSignal, QEasingCurve, QSize, QTimer, QPoint, QRect,
    QSequentialAnimationGroup, QVariantAnimation, QUrl, QThread
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

import sys
import os
import json
import logging
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

# Modern, professional color scheme
PRIMARY_COLOR = "#FFFFFF"  # White background
ACCENT_COLOR = "#dc2626"  # Blue accent color
SECONDARY_COLOR = "#10b981"  # Green for success
TEXT_COLOR = "#1e293b"  # Dark slate for text
LIGHT_GRAY = "#f8fafc"  # Light gray for alternate backgrounds
GRAY_TEXT = "#64748b"  # Medium gray for secondary text
BUTTON_ORANGE = "#1e293b"  # Orange for pause button
BUTTON_RED = "#dc2626"  # Red for stop button
BUTTON_BLUE = "#dc2626"  # Blue for next button
BORDER_COLOR = "#e2e8f0"  # Light gray for borders
DISABLED_COLOR = "#cbd5e1"  # Disabled button color
STEP_LABEL_BG = "#ef4444"  # Light blue for step number badge
PAUSED_BG_COLOR = "#f1f5f9"  # Lighter gray for paused state
GOAL_BG_COLOR = "#dc2626"  # Very light blue for goal background
GOAL_BORDER_COLOR = "#ef4444"  # Light blue for goal border
EXPLANATION_BG_COLOR = "#fffbeb"  # Light yellow for explanation background
EXPLANATION_BORDER_COLOR = "#fcd34d"  # Yellow for explanation border

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
            color: {ACCENT_COLOR};
            background-color: {STEP_LABEL_BG};
            border-radius: 12px;
        """)





class TypewriterLabel(QLabel):
    """Label with typewriter animation effect"""

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
        self.full_text = text
        self.current_text = ""
        self.char_index = 0
        self.setText("")
        self.animation_timer.start(self.char_delay)

    def update_text(self):
        """Update the displayed text with the next character"""
        if self.char_index < len(self.full_text):
            self.current_text += self.full_text[self.char_index]
            self.setText(self.current_text)
            self.char_index += 1
        else:
            self.animation_timer.stop()


class AudioPlayer(QWidget):
    """Audio player widget for step instructions"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.current_audio_path = None

        # Set up audio player components
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        # Connect signals
        self.player.playbackStateChanged.connect(self.update_play_button)
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)

        # Create UI
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Play/Pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon(os.path.join(ICON_DIR, "play_audio.png")))
        self.play_button.setFixedSize(32, 32)
        self.play_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                border-radius: 16px;
            }}
            QPushButton:hover {{
                background-color: #1e40af;
            }}
        """)
        self.play_button.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_button)

        # Progress slider
        self.progress_slider = QSlider(Qt.Orientation.Horizontal)
        self.progress_slider.setRange(0, 100)
        self.progress_slider.setValue(0)
        self.progress_slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {BORDER_COLOR};
                height: 6px;
                background: {LIGHT_GRAY};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {ACCENT_COLOR};
                border: none;
                width: 14px;
                height: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }}
            QSlider::sub-page:horizontal {{
                background: {ACCENT_COLOR};
                border-radius: 3px;
            }}
        """)
        self.progress_slider.sliderMoved.connect(self.set_position)
        layout.addWidget(self.progress_slider)

        # Volume button
        self.volume_button = QPushButton()
        self.volume_button.setIcon(QIcon(os.path.join(ICON_DIR, "volume.png")))
        self.volume_button.setFixedSize(32, 32)
        self.volume_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {LIGHT_GRAY};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 16px;
            }}
            QPushButton:hover {{
                background-color: #e2e8f0;
            }}
        """)
        self.volume_button.clicked.connect(self.toggle_mute)
        layout.addWidget(self.volume_button)

        # Set initial volume
        self.audio_output.setVolume(0.7)

        # Hide player initially
        self.hide()

    def set_audio(self, audio_path):
        """Set the audio file to play"""
        if not audio_path or not os.path.exists(audio_path):
            self.hide()
            return

        self.current_audio_path = audio_path
        self.player.setSource(QUrl.fromLocalFile(audio_path))
        self.show()

    def toggle_play(self):
        """Toggle between play and pause"""
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def toggle_mute(self):
        """Toggle mute/unmute"""
        self.audio_output.setMuted(not self.audio_output.isMuted())

        # Update button icon
        if self.audio_output.isMuted():
            self.volume_button.setIcon(QIcon(os.path.join(ICON_DIR, "volume_mute.png")))
        else:
            self.volume_button.setIcon(QIcon(os.path.join(ICON_DIR, "volume.png")))

    def update_play_button(self, state):
        """Update play button icon based on playback state"""
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_button.setIcon(QIcon(os.path.join(ICON_DIR, "pause_audio.png")))
        else:
            self.play_button.setIcon(QIcon(os.path.join(ICON_DIR, "play_audio.png")))

    def update_position(self, position):
        """Update slider position during playback"""
        if self.player.duration() > 0:
            self.progress_slider.setValue(int(position / self.player.duration() * 100))

    def update_duration(self, duration):
        """Update slider when audio duration changes"""
        self.progress_slider.setValue(0)

    def set_position(self, position):
        """Set playback position when slider is moved"""
        if self.player.duration() > 0:
            new_position = int(position / 100.0 * self.player.duration())
            self.player.setPosition(new_position)

def render_instruction_blocks(blocks: List[Dict[str, str]], layout: QVBoxLayout):
    for block in blocks:
        btype = block.get("type")
        content = block.get("content", "")
        label = block.get("label", "")

        if btype == "text":
            text_label = QLabel(content)
            text_label.setWordWrap(True)
            text_label.setFont(QFont("Inter", 13))
            layout.addWidget(text_label)

        elif btype == "link":
            frame = QFrame()
            frame.setStyleSheet("""
                QFrame {
                    background-color: #f1f5f9;
                    border: 1px solid #e2e8f0;
                    border-radius: 5px;
                    padding: 6px;
                }
            """)
            hlayout = QHBoxLayout(frame)
            link_label = QLabel(label or content)
            link_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
                    background-color: transparent;
                    border-left: 4px solid {ACCENT_COLOR};
                    border-radius: 6px;
                    margin: 4px 0px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                #stepItem {{
                    background-color: transparent;
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

        step_label = QLabel(f"Step {step_number}")
        step_label.setFont(QFont("Inter", 12, QFont.Weight.Bold))
        step_label.setStyleSheet(f"color: {ACCENT_COLOR};")
        step_header.addWidget(step_label)

        step_header.addStretch()
        layout.addLayout(step_header)

        # Main logic: check for new JSON "blocks" format
        if isinstance(content, dict):
            blocks = []
            explanation = content.get('explanation', '')
            audio_path = content.get('audio_path', None)

            try:
                instruction_data = content.get('instruction', '')
                if isinstance(instruction_data, str):
                    parsed = json.loads(instruction_data)
                    blocks = parsed.get('blocks', [])
                    if not explanation and 'explanation' in parsed:
                        explanation = parsed.get('explanation', '')
                elif isinstance(instruction_data, dict):
                    blocks = instruction_data.get('blocks', [])
            except Exception:
                pass  # fallback to legacy

            if blocks:
                for block in blocks:
                    btype = block.get("type")
                    value = block.get("content", "")
                    label = block.get("label", "")

                    if btype == "text":
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
                        hbox.addWidget(link_label)
                        copy_btn = QPushButton("Copy")
                        copy_btn.clicked.connect(lambda _, v=value: QApplication.clipboard().setText(v))
                        hbox.addWidget(copy_btn)
                        layout.addWidget(link_frame)

                    elif btype == "code":
                        self.create_code_block(value, layout)
            else:
                # Fallback to legacy format
                instruction = content.get('instruction', '')
                format_type = content.get('format', 'text')
                if format_type == 'code':
                    self.create_code_block(instruction, layout)
                else:
                    self.create_text_block(instruction, layout, is_current)

            if explanation:
                self.add_explanation(explanation, layout)

            if audio_path and os.path.exists(audio_path):
                self.audio_player = AudioPlayer()
                self.audio_player.set_audio(audio_path)
                layout.addWidget(self.audio_player)
        else:
            self.create_text_block(content, layout, is_current)

        if is_current:
            self.animate_entrance()

    def create_text_block(self, text, layout, animate=False):
        """Create a regular text display with optional typewriter effect"""
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
        header.setStyleSheet("color: #92400e;")
        explanation_layout.addWidget(header)

        # Explanation text
        explanation = QLabel(explanation_text)
        explanation.setFont(QFont("Inter", 11))
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #78350f;")
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


# === MODIFICATIONS TO CHATDIALOG IN LEARNCHAIN_TUTOR.PY ===

class ChatDialog(QDialog):
    """Dialog for user to ask questions and receive help during a session, integrated with retriever backend"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle("Chat with Assistant")
        self.setMinimumSize(500, 600)  # Slightly larger dialog for better readability
        self.setStyleSheet(f"background-color: {PRIMARY_COLOR};")

        # Knowledge retriever settings - with reasonable defaults
        self.use_query_enhancement = True
        self.use_enhanced_retrieval = True
        self.use_multi_query = True
        self.is_processing = False  # Track if a query is being processed
        self.conversation_count = 0  # Track number of Q&A pairs

        # Ensure the dialog is modal - prevents interaction with parent window
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Chat history display with enhanced scrollbar styling
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet(f"""
            QTextEdit {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: {LIGHT_GRAY};
                padding: 10px;
                font-family: Inter, Arial;
                font-size: 12px;
                color: {TEXT_COLOR};
            }}
            QScrollBar:vertical {{
                background: #f1f5f9;
                width: 12px;
                margin: 0px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: #3b82f6;
                min-height: 30px;
                border-radius: 6px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
                background: #e2e8f0;
                border-radius: 6px;
            }}
        """)
        layout.addWidget(self.chat_history)

        # Settings section with enhanced styling
        settings_group = QFrame()
        settings_group.setStyleSheet(f"""
            QFrame {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: {PRIMARY_COLOR};
                padding: 10px;
            }}
        """)
        settings_layout = QVBoxLayout(settings_group)

        # Settings label
        settings_label = QLabel("Retrieval Settings")
        settings_label.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        settings_label.setStyleSheet(f"color: {TEXT_COLOR};")
        settings_layout.addWidget(settings_label)

        # Settings toggles with enhanced checkbox styling
        toggle_layout = QHBoxLayout()

        # Custom stylesheet for checkboxes
        checkbox_style = f"""
            QCheckBox {{
                spacing: 8px;
                font-size: 13px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 1px solid {BORDER_COLOR};
            }}
            QCheckBox::indicator:unchecked {{
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: {ACCENT_COLOR};
                border: 1px solid {ACCENT_COLOR};
                image: url({os.path.join(ICON_DIR, "check.png")});
            }}
            QCheckBox::indicator:hover {{
                border: 1px solid {ACCENT_COLOR};
            }}
        """

        # Query enhancement toggle
        self.enhance_toggle = QCheckBox("Query Enhancement")
        self.enhance_toggle.setChecked(True)
        self.enhance_toggle.setFont(QFont("Inter", 10))
        self.enhance_toggle.setStyleSheet(checkbox_style)
        self.enhance_toggle.stateChanged.connect(self.update_settings)
        toggle_layout.addWidget(self.enhance_toggle)

        # Enhanced retrieval toggle
        self.retrieval_toggle = QCheckBox("Enhanced Retrieval")
        self.retrieval_toggle.setChecked(True)
        self.retrieval_toggle.setFont(QFont("Inter", 10))
        self.retrieval_toggle.setStyleSheet(checkbox_style)
        self.retrieval_toggle.stateChanged.connect(self.update_settings)
        toggle_layout.addWidget(self.retrieval_toggle)

        settings_layout.addLayout(toggle_layout)
        layout.addWidget(settings_group)

        # After the existing toggles, add multi-query toggle
        # Multi-query toggle
        self.multi_query_toggle = QCheckBox("Multi-Query Retrieval")
        self.multi_query_toggle.setChecked(True)  # Default to enabled
        self.multi_query_toggle.setFont(QFont("Inter", 10))
        self.multi_query_toggle.setStyleSheet(checkbox_style)
        self.multi_query_toggle.setToolTip("Generate multiple diverse queries for better results")
        self.multi_query_toggle.stateChanged.connect(self.update_settings)
        toggle_layout.addWidget(self.multi_query_toggle)

        # Message input with container for progress indicator
        input_container = QFrame()
        input_container.setStyleSheet("background: transparent; border: none;")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(5)

        # Progress indicator label (hidden by default)
        self.progress_label = QLabel("Processing your question...")
        self.progress_label.setFont(QFont("Inter", 10, QFont.Weight.Normal))
        self.progress_label.setStyleSheet(f"color: {ACCENT_COLOR}; font-style: italic;")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setVisible(False)  # Initially hidden
        input_layout.addWidget(self.progress_label)

        # Text input field
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your question here...")
        self.message_input.setFixedHeight(45)
        self.message_input.setFont(QFont("Inter", 11))
        self.message_input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 10px 15px;
                background-color: {PRIMARY_COLOR};

            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)

        layout.addWidget(input_container)

        # Send button
        button_row = QHBoxLayout()

        # Spacer to push buttons to the right
        button_row.addStretch()

        # Send button
        self.send_btn = QPushButton("Send")
        self.send_btn.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: #808080;
            }}
            QPushButton:disabled {{
                background-color: {DISABLED_COLOR};
                color: white;
            }}
        """)
        self.send_btn.clicked.connect(self.send_message)
        button_row.addWidget(self.send_btn)

        # Close button
        self.close_btn = QPushButton("Close")
        self.close_btn.setFont(QFont("Inter", 11))
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {LIGHT_GRAY};
                color: {TEXT_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                padding: 10px 20px;
            }}
            QPushButton:hover {{
                background-color: #808080;
            }}
        """)
        self.close_btn.clicked.connect(self.close_chat)
        button_row.addWidget(self.close_btn)

        layout.addLayout(button_row)

        # Try to load a check icon for the checkboxes
        try:
            # Create icons directory if it doesn't exist
            if not os.path.exists(ICON_DIR):
                os.makedirs(ICON_DIR)

            # Create a simple check icon if it doesn't exist
            check_icon_path = os.path.join(ICON_DIR, "check.png")
            if not os.path.exists(check_icon_path):
                # Create a simple check mark using QPainter
                pixmap = QPixmap(18, 18)
                pixmap.fill(Qt.GlobalColor.transparent)
                painter = QPainter(pixmap)
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
                painter.drawLine(4, 9, 8, 13)
                painter.drawLine(8, 13, 14, 5)
                painter.end()
                pixmap.save(check_icon_path)
                print(f"Created check icon at: {check_icon_path}")
        except Exception as e:
            print(f"Error creating check icon: {e}")

        # Add initial welcome message
            # Add initial welcome message
            self.add_assistant_message("I'm here to help with your training session. What can I assist you with?")

            # Display a system message explaining the functionality
            self.add_system_message(
                "This assistant is connected to your course materials. Ask questions about what you're learning!")

    def update_settings(self):
        """Update retriever settings based on toggle states"""
        self.use_query_enhancement = self.enhance_toggle.isChecked()
        self.use_enhanced_retrieval = self.retrieval_toggle.isChecked()
        self.use_multi_query = self.multi_query_toggle.isChecked()  # Add this

        # Show a system message to inform about settings changes
        self.add_system_message(
            f"Settings updated: Query enhancement {'enabled' if self.use_query_enhancement else 'disabled'}, "
            f"Enhanced retrieval {'enabled' if self.use_enhanced_retrieval else 'disabled'},"
            f"Multi-query {'enabled' if self.use_multi_query else 'disabled'}"
        )

    def add_conversation_header(self, number):
        """Add a very clear header to separate conversations"""
        self.chat_history.append(
            f'<div style="margin: 30px 0 20px 0; text-align: center;">'
            f'<div style="display: inline-block; background-color: {ACCENT_COLOR}; color: white; '
            f'border-radius: 15px; padding: 5px 15px; font-weight: bold; font-size: 11px;">'
            f'Conversation {number}</div>'
            f'<div style="height: 2px; background-color: #94a3b8; margin-top: 10px;"></div>'
            f'</div>'
        )

        # Auto-scroll to the bottom
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def add_user_message(self, message):
        """Add a user message to the chat history with improved styling"""
        # Add user message with enhanced styling and more distinct background
        self.chat_history.append(
            f'<div style="background-color: #eff6ff; padding: 12px 15px; border-radius: 8px; '
            f'border-left: 4px solid {ACCENT_COLOR}; margin: 10px 0;">'
            f'<div style="font-weight: bold; color: {TEXT_COLOR}; margin-bottom: 8px; font-size: 13px;">You:</div>'
            f'<div style="margin: 5px 0; font-size: 13px;">{message}</div>'
            f'</div>'
        )

        # Auto-scroll to the bottom
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

        # Also add to training wheels chat history
        if hasattr(self.parent_widget, 'add_chat_to_session'):
            self.parent_widget.add_chat_to_session('user', message)

    def add_thinking_indicator(self):
        """Add a thinking indicator while waiting for response"""
        # Create the HTML for the thinking indicator
        self.thinking_html = (
            f'<div id="thinking_indicator" style="background-color: #f8fafc; padding: 12px 15px; '
            f'border-radius: 8px; border-left: 4px solid #9ca3af; margin: 10px 0;">'
            f'<div style="font-weight: bold; color: {ACCENT_COLOR}; margin-bottom: 8px; font-size: 13px;">Assistant:</div>'
            f'<div style="margin: 5px 0; color: {ACCENT_COLOR}; font-style: italic;">Assistant is thinking...</div>'
            f'<div id="dot-animation" style="font-size: 24px; letter-spacing: 3px; color: {ACCENT_COLOR}; text-align: center;">•••</div>'
            f'</div>'
        )

        # Add the thinking indicator to the chat history
        self.chat_history.append(self.thinking_html)

        # Auto-scroll to the bottom
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

        # Force the UI to update immediately
        QApplication.processEvents()

        # Start animation
        self.thinking_dots = 0
        self.thinking_timer = QTimer()
        self.thinking_timer.timeout.connect(self.animate_thinking)
        self.thinking_timer.start(400)  # Update every 400ms

    def animate_thinking(self):
        """Animate the thinking indicator dots"""
        dot_patterns = ["•  ", "•• ", "•••", " ••", "  •", "   "]
        self.thinking_dots = (self.thinking_dots + 1) % len(dot_patterns)

        # Update the dots in the HTML
        dot_html = f'<div id="dot-animation" style="font-size: 24px; letter-spacing: 3px; color: {ACCENT_COLOR};">{dot_patterns[self.thinking_dots]}</div>'

        current_html = self.chat_history.toHtml()
        updated_html = current_html.replace(
            '<div id="dot-animation" style="font-size: 24px; letter-spacing: 3px; color: #2563eb;">•••</div>',
            dot_html)
        updated_html = updated_html.replace(
            f'<div id="dot-animation" style="font-size: 24px; letter-spacing: 3px; color: {ACCENT_COLOR};">{dot_patterns[(self.thinking_dots - 1) % len(dot_patterns)]}</div>',
            dot_html)

        # Set the updated HTML
        self.chat_history.setHtml(updated_html)

        # Keep scroll at the bottom
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def remove_thinking_indicator(self):
        """Remove the thinking indicator"""
        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.thinking_timer.stop()

        # Only try to remove if thinking_html exists
        if hasattr(self, 'thinking_html'):
            # Remove the thinking indicator from the HTML
            current_html = self.chat_history.toHtml()
            updated_html = current_html.replace(self.thinking_html, '')

            # Also try to remove it if the dots have been animated
            for dots in ["•  ", "•• ", "•••", " ••", "  •", "   "]:
                indicator_with_dots = self.thinking_html.replace(
                    '<div id="dot-animation" style="font-size: 24px; letter-spacing: 3px; color: #2563eb;">•••</div>',
                    f'<div id="dot-animation" style="font-size: 24px; letter-spacing: 3px; color: {ACCENT_COLOR};">{dots}</div>')
                updated_html = updated_html.replace(indicator_with_dots, '')

            # Set the updated HTML
            self.chat_history.setHtml(updated_html)

    def add_assistant_message(self, message):
        """Add an assistant message to the chat history with improved styling and list formatting"""
        # First check if we need to remove thinking indicator
        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.remove_thinking_indicator()

        # Process message to improve formatting for lists
        formatted_message = self._format_message_content(message)

        # Add assistant message with enhanced styling
        self.chat_history.append(
            f'<div style="background-color: #f0f9ff; padding: 12px 15px; border-radius: 8px; '
            f'border-left: 4px solid #10b981; margin: 10px 0;">'
            f'<div style="font-weight: bold; color: {ACCENT_COLOR}; margin-bottom: 8px; font-size: 13px;">Assistant:</div>'
            f'<div style="margin: 5px 0; font-size: 13px;">{formatted_message}</div>'
            f'</div>'
        )

        # Auto-scroll to the bottom
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

        # Also add to training wheels chat history
        if hasattr(self.parent_widget, 'add_chat_to_session'):
            self.parent_widget.add_chat_to_session('assistant', message)

    def _format_message_content(self, message):
        """Format message content for better readability, especially for lists"""
        # Handle numbered lists (e.g., "1. Item")
        message = self._format_numbered_list(message)

        # Handle bullet lists (e.g., "• Item" or "- Item")
        message = self._format_bullet_list(message)

        # Improve spacing for paragraphs
        message = message.replace("\n\n", "</p><p>")

        # Enhance important terms with bold
        message = self._highlight_important_terms(message)

        return message

    def _format_numbered_list(self, text):
        """Format numbered lists for better display"""
        import re

        # Check if text potentially contains a numbered list
        if re.search(r'\d+\.\s', text):
            # Start with an empty result
            result = []
            in_list = False
            lines = text.split('\n')

            for line in lines:
                # Check if this line is a list item
                if re.match(r'^\s*\d+\.\s', line):
                    if not in_list:
                        # Start a new list
                        in_list = True
                        result.append('<ol style="margin-left: 20px; margin-top: 10px; margin-bottom: 10px;">')

                    # Add the list item with improved styling
                    list_content = re.sub(r'^\s*\d+\.\s', '', line)
                    result.append(f'<li style="margin-bottom: 8px;"><strong>{list_content}</strong></li>')
                else:
                    if in_list:
                        # End the current list
                        in_list = False
                        result.append('</ol>')

                    # Add regular line
                    result.append(line)

            # Close any open list
            if in_list:
                result.append('</ol>')

            return '\n'.join(result)

        return text

    def _format_bullet_list(self, text):
        """Format bullet lists for better display"""
        import re

        # Check if text potentially contains a bullet list
        if re.search(r'[•\-\*]\s', text):
            # Start with an empty result
            result = []
            in_list = False
            lines = text.split('\n')

            for line in lines:
                # Check if this line is a list item
                if re.match(r'^\s*[•\-\*]\s', line):
                    if not in_list:
                        # Start a new list
                        in_list = True
                        result.append('<ul style="margin-left: 20px; margin-top: 10px; margin-bottom: 10px;">')

                    # Add the list item with improved styling
                    list_content = re.sub(r'^\s*[•\-\*]\s', '', line)
                    result.append(f'<li style="margin-bottom: 8px;">{list_content}</li>')
                else:
                    if in_list:
                        # End the current list
                        in_list = False
                        result.append('</ul>')

                    # Add regular line
                    result.append(line)

            # Close any open list
            if in_list:
                result.append('</ul>')

            return '\n'.join(result)

        return text

    def _highlight_important_terms(self, text):
        """Highlight important terms in the text"""
        # List of terms to highlight
        important_terms = [
            "Seven B's", "7 B's", "Be Available", "Be Efficient", "Be Knowledgeable",
            "Be Proactive", "Be Respectful", "Be Responsive", "Be Trustworthy"
        ]

        # Highlight each term
        for term in important_terms:
            text = text.replace(term, f"<strong>{term}</strong>")

        return text

    def add_system_message(self, message):
        """Add a system message to the chat history with improved styling"""
        self.chat_history.append(
            f'<div style="background-color: #f1f5f9; padding: 5px 10px; border-radius: 4px; '
            f'margin: 5px 0; border-left: 2px solid #64748b;">'
            f'<span style="font-style: italic; color: #64748b;">System: {message}</span>'
            f'</div>'
        )

        # Auto-scroll to the bottom
        self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def add_sources_message(self, sources):
        """Add sources information to the chat history with improved styling"""
        if sources and len(sources) > 0:
            sources_html = (
                f'<div style="background-color: #fffbeb; padding: 8px 12px; border-radius: 6px; '
                f'border-left: 3px solid #eab308; margin: 5px 0;">'
                f'<div style="font-weight: bold; color: #854d0e;">Sources:</div>'
                f'<ul style="margin: 5px 0 5px 20px; padding: 0;">'
            )

            for source in sources:
                sources_html += f'<li style="margin-bottom: 3px;">{source}</li>'

            sources_html += '</ul></div>'

            self.chat_history.append(sources_html)

            # Auto-scroll to the bottom
            self.chat_history.verticalScrollBar().setValue(self.chat_history.verticalScrollBar().maximum())

    def send_message(self):
        """Send the user's message and get a response using the retriever backend"""
        # Prevent multiple sends while processing
        if self.is_processing:
            return

        message = self.message_input.text().strip()
        if not message:
            return

        # Set processing state FIRST - before any retrieval happens
        self.is_processing = True

        # Store the message and immediately clear the input field
        stored_message = message
        self.message_input.clear()

        # Increment conversation counter and add header for clear separation
        self.conversation_count += 1
        self.add_conversation_header(self.conversation_count)

        # Add user message to chat
        self.add_user_message(stored_message)

        # Update UI to show processing state BEFORE retrieval
        self.message_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Processing...")
        self.progress_label.setVisible(True)

        # Add thinking indicator BEFORE retrieval
        self.add_thinking_indicator()

        # Force UI update to ensure changes are visible immediately
        QApplication.processEvents()

        # Use a timer to update the progress indicators
        self.dots = 0
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_indicator)
        self.progress_timer.start(300)  # Update every 300ms

        try:
            # Get course_id from parent window
            course_id = self.parent_widget.course_id if hasattr(self.parent_widget, 'course_id') else "001"

            # Import only the working functions
            from retriever import query_course_knowledge, generate_diverse_queries

            # Use multi-query if enabled
            if self.use_multi_query:
                # Generate diverse queries
                try:
                    queries = generate_diverse_queries(stored_message, num_queries=5)
                    self.add_system_message(f"Generated {len(queries)} diverse queries to find the best answer")

                    # Collect all answers
                    all_answers = []
                    all_sources = set()

                    for query in queries:
                        result = query_course_knowledge(
                            course_id=course_id,
                            question=query,
                            use_query_enhancement=False,  # Don't enhance since we already have diverse queries
                            use_enhanced_retrieval=self.use_enhanced_retrieval,
                            k=3
                        )

                        if result.get("answer") and "I don't have enough information" not in result.get("answer", ""):
                            all_answers.append(result["answer"])
                            all_sources.update(result.get("source_documents", []))

                    # Combine answers or use the best one
                    if all_answers:
                        # For now, just use the first good answer
                        # You could enhance this to combine answers intelligently
                        answer = all_answers[0]
                        sources = list(all_sources)
                    else:
                        answer = "I don't have enough information in the course materials to answer that question."
                        sources = []

                except Exception as e:
                    # If diverse query generation fails, fall back to single query
                    logger.error(f"Error generating diverse queries: {e}")
                    self.use_multi_query = False

            # Single query mode (fallback or if multi-query is disabled)
            if not self.use_multi_query:
                result = query_course_knowledge(
                    course_id=course_id,
                    question=stored_message,
                    use_query_enhancement=self.use_query_enhancement,
                    use_enhanced_retrieval=self.use_enhanced_retrieval,
                    k=10,
                    use_parallel_processing=True
                )

                answer = result.get("answer", "I'm unable to find an answer to that question.")
                sources = result.get("source_documents", [])

                # If query was enhanced, show the enhancement
                query_info = result.get("query_info", {})
                if query_info and "original_query" in query_info and "enhanced_query" in query_info:
                    self.add_system_message(
                        f"Query enhanced from: '{query_info['original_query']}' "
                        f"to: '{query_info['enhanced_query']}'"
                    )

            # Add assistant response
            self.add_assistant_message(answer)

            # Add sources if available
            if sources:
                self.add_sources_message(sources)

        except Exception as e:
            # Handle any errors gracefully
            self.add_system_message(f"Error: {str(e)}")
            self.add_assistant_message(
                "I encountered an error while processing your question. Please try again later.")

        finally:
            # Reset processing state
            self.is_processing = False
            self.message_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send")
            self.progress_label.setVisible(False)

            # Stop the progress timer
            if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
                self.progress_timer.stop()

    def update_progress_indicator(self):
        """Update the progress indicator with animated dots"""
        self.dots = (self.dots + 1) % 4
        dots_text = "." * self.dots
        self.progress_label.setText(f"Processing your question{dots_text.ljust(3)}")

    def closeEvent(self, event):
        """Override close event to ensure proper handling"""
        # Stop any active timers
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()

        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.thinking_timer.stop()

        self.close_chat()
        event.accept()

    def close_chat(self):
        """Custom close handler to ensure we properly handle state"""
        # Stop any active timers
        if hasattr(self, 'progress_timer') and self.progress_timer.isActive():
            self.progress_timer.stop()

        if hasattr(self, 'thinking_timer') and self.thinking_timer.isActive():
            self.thinking_timer.stop()

        self.hide()  # Hide instead of closing to maintain instance

class LoginScreen(QWidget):
    """The login screen for the LearnChain Tutor app with fixed logo display"""


    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setStyleSheet(f"background-color: {PRIMARY_COLOR};")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(50, 80, 50, 80)
        layout.setSpacing(20)

        # App logo - modified for more reliable loading
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Explicitly create a default logo text as fallback
        logo_label.setText("LearnChain")
        logo_label.setFont(QFont("Inter", 24, QFont.Weight.Bold))
        logo_label.setStyleSheet(f"color: {ACCENT_COLOR};")

        # Try to load the logo image if it exists
        try:
            logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
            if os.path.exists(logo_path) and os.path.isfile(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    # Successfully loaded the image
                    logo_label.setText("")  # Clear the text
                    logo_label.setPixmap(pixmap.scaledToWidth(150, Qt.TransformationMode.SmoothTransformation))
                    print(f"Logo loaded from: {logo_path}")
                else:
                    print(f"Failed to load logo: {logo_path} (image is null)")
            else:
                print(f"Logo file not found at: {logo_path}")
        except Exception as e:
            print(f"Error loading logo: {str(e)}")

        layout.addWidget(logo_label)

        # Title with app name
        title = QLabel("Training Wheels")
        title.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {TEXT_COLOR};")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Login instruction text
        instruction = QLabel("Please enter your credentials to continue")
        instruction.setFont(QFont("Inter", 12))
        instruction.setStyleSheet(f"color: {GRAY_TEXT};")
        instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instruction)

        # Add spacer
        layout.addItem(QSpacerItem(20, 20))

        # User input field
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter your User ID")
        self.user_input.setFixedHeight(50)
        self.user_input.setFont(QFont("Inter", 12))
        self.user_input.setStyleSheet(f"""
                QLineEdit {{
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 8px;
                    padding: 12px 15px;
                    background-color: {LIGHT_GRAY};
                }}
                QLineEdit:focus {{
                    border: 1px solid {ACCENT_COLOR};
                }}
            """)
        layout.addWidget(self.user_input)

        # Small spacing
        layout.addItem(QSpacerItem(10, 10))

        # Course ID input field
        self.course_input = QLineEdit()
        self.course_input.setPlaceholderText("Enter your Course ID")
        self.course_input.setFixedHeight(50)
        self.course_input.setFont(QFont("Inter", 12))
        self.course_input.setStyleSheet(f"""
                QLineEdit {{
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 8px;
                    padding: 12px 15px;
                    background-color: {LIGHT_GRAY};
                }}
                QLineEdit:focus {{
                    border: 1px solid {ACCENT_COLOR};
                }}
            """)
        layout.addWidget(self.course_input)

        # Small spacing
        layout.addItem(QSpacerItem(20, 20))

        # Login button
        login_btn = QPushButton("Login")
        login_btn.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        login_btn.setFixedHeight(50)
        login_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        login_btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {ACCENT_COLOR};
                    color: white;
                    border: none;
                    border-radius: 8px;
                }}
                QPushButton:hover {{
                    background-color: #808080;
                }}
                QPushButton:pressed {{
                    background-color: #1e3a8a;
                }}
            """)
        login_btn.clicked.connect(self.handle_login)
        layout.addWidget(login_btn)

        # Add stretching space at the bottom
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

    # COMPLETE CLASSES THAT NEED MODIFICATION

class GoalDisplay(QWidget):
    """Completely redesigned Goal display with guaranteed visible header text"""


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

        # Header section - explicit foreground and background colors
        self.header_label = QLabel("Goal")
        self.header_label.setFixedHeight(40)
        self.header_label.setFont(QFont("Inter", 11, QFont.Weight.Bold))
        # CRITICAL: Setting text alignment, padding, and colors explicitly
        self.header_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.header_label.setIndent(20)  # Left padding for text
        self.header_label.setStyleSheet(f"""
                background-color: {ACCENT_COLOR};
                color: white;
                padding-left: 10px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            """)
        main_layout.addWidget(self.header_label)

        # Content frame with light background and border
        self.content_frame = QFrame()
        self.content_frame.setMinimumHeight(60)
        self.content_frame.setStyleSheet(f"""
                background-color: #f0f9ff;
                border: 1px solid #e0e7ff;
                border-top: none;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            """)

        content_layout = QVBoxLayout(self.content_frame)
        content_layout.setContentsMargins(20, 15, 20, 15)

        # Goal text with explicit font and alignment
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
        # Force layout update
        self.content.adjustSize()

class ToggleSwitch(QWidget):
    """Custom toggle switch widget for explanation mode"""

    toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)

        # Set up layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Create switch track and handle
        self.track = QFrame(self)
        self.track.setFixedSize(44, 22)
        self.track.setStyleSheet(f"""
                QFrame {{
                    background-color: {DISABLED_COLOR};
                    border-radius: 11px;
                }}
            """)

        # Create handle (the circle that moves)
        self.handle = QFrame(self.track)
        self.handle.setFixedSize(18, 18)
        self.handle.move(2, 2)  # Position at left side initially
        self.handle.setStyleSheet(f"""
                QFrame {{
                    background-color: white;
                    border-radius: 9px;
                }}
            """)

        # Add label if provided


        layout.addWidget(self.track)

        # State
        self.is_checked = False

        # Make clickable
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(24)

        # Animation for handle
        self.animation = QPropertyAnimation(self.handle, b"pos")
        self.animation.setDuration(150)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def mousePressEvent(self, event):
        """Handle mouse click to toggle switch"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.toggle()

    def toggle(self):
        """Toggle the switch state"""
        self.is_checked = not self.is_checked

        # Animate handle position
        if self.is_checked:
            self.animation.setStartValue(self.handle.pos())
            self.animation.setEndValue(QPoint(self.track.width() - self.handle.width() - 2, 2))
            self.track.setStyleSheet(f"""
                    QFrame {{
                        background-color: {ACCENT_COLOR};
                        border-radius: 11px;
                    }}
                """)
        else:
            self.animation.setStartValue(self.handle.pos())
            self.animation.setEndValue(QPoint(2, 2))
            self.track.setStyleSheet(f"""
                    QFrame {{
                        background-color: {DISABLED_COLOR};
                        border-radius: 11px;
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
                        border-radius: 11px;
                    }}
                """)
            self.handle.move(self.track.width() - self.handle.width() - 2, 2)
        else:
            self.track.setStyleSheet(f"""
                    QFrame {{
                        background-color: {DISABLED_COLOR};
                        border-radius: 11px;
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
        self.course_id = None  # Added to store course ID
        self.goal = None
        self.step_index = 1
        self.steps = []
        self.is_session_active = False
        self.is_session_paused = False
        self.explanation_mode = False  # Track if explanation mode is enabled
        self.audio_mode = False  # Track if audio mode is enabled

        # Create chat dialog
        self.chat_dialog = ChatDialog(self)

        self.setMinimumSize(width, height)
        self.setStyleSheet(f"""
             QWidget {{
                 background-color: {PRIMARY_COLOR};
             }}
         """)
        #icon_path = str(resource_path("training_wheels/assets/icon.ico"))
        icon_path = str(resource_path("assets/icon.ico"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # ===== Header section =====
        header = QFrame()
        header.setStyleSheet(f"""
            QFrame {{
                background-color: {LIGHT_GRAY};
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
                background-color: {LIGHT_GRAY};
                border-bottom: 1px solid {BORDER_COLOR};
            }}
        """)

        goal_layout = QVBoxLayout(self.goal_input_section)
        goal_layout.setContentsMargins(30, 30, 30, 30)
        goal_layout.setSpacing(20)

        goal_title = QLabel("What would you like to learn today?")
        goal_title.setFont(QFont("Inter", 16, QFont.Weight.Bold))
        goal_title.setStyleSheet(f"color: {TEXT_COLOR};")
        goal_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        goal_layout.addWidget(goal_title)

        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText("e.g., Deploy a Django site on Lightsail")
        self.goal_input.setFixedHeight(50)
        self.goal_input.setFont(QFont("Inter", 12))
        self.goal_input.setStyleSheet(f"""
            QLineEdit {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 8px;
                padding: 12px 15px;
                background-color: {PRIMARY_COLOR};
            }}
            QLineEdit:focus {{
                border: 1px solid {ACCENT_COLOR};
            }}
        """)
        goal_layout.addWidget(self.goal_input)

        # Add toggle switches for explanation mode and audio
        # Add toggle switches for explanation mode and audio
        toggles_frame = QFrame()
        toggles_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {PRIMARY_COLOR};
                border-radius: 8px;
                border: 1px solid {BORDER_COLOR};
            }}
        """)
        toggles_frame.setFixedHeight(60)  # Add fixed height

        toggles_layout = QHBoxLayout(toggles_frame)
        toggles_layout.setContentsMargins(20, 15, 20, 15)  # Better padding
        toggles_layout.setSpacing(40)  # More spacing between toggles

        # Explanation mode toggle
        explanation_container = QHBoxLayout()
        explanation_label = QLabel("Include Explanations")
        explanation_label.setFont(QFont("Inter", 11))
        explanation_label.setStyleSheet(f"color: {TEXT_COLOR};")
        explanation_container.addWidget(explanation_label)
        explanation_container.addSpacing(10)
        self.explanation_toggle = ToggleSwitch()
        self.explanation_toggle.toggled.connect(self.toggle_explanation_mode)
        explanation_container.addWidget(self.explanation_toggle)
        toggles_layout.addLayout(explanation_container)

        # Audio mode toggle
        audio_container = QHBoxLayout()
        audio_label = QLabel("Enable Audio")
        audio_label.setFont(QFont("Inter", 11))
        audio_label.setStyleSheet(f"color: {TEXT_COLOR};")
        audio_container.addWidget(audio_label)
        audio_container.addSpacing(10)
        self.audio_toggle = ToggleSwitch()
        self.audio_toggle.toggled.connect(self.toggle_audio_mode)
        audio_container.addWidget(self.audio_toggle)
        toggles_layout.addLayout(audio_container)

        toggles_layout.addStretch()  # Push toggles to left

        goal_layout.addWidget(toggles_frame)

        start_btn = QPushButton("Start Session")
        start_btn.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        start_btn.setFixedHeight(50)
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.setIcon(QIcon(str(resource_path("icons/play.png"))))
        start_btn.setIconSize(QSize(20, 20))
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                border: none;
                border-radius: 8px;
                padding-left: 15px;
            }}
            QPushButton:hover {{
                background-color: #808080;
            }}
            QPushButton:pressed {{
                background-color: #1e3a8a;
            }}
        """)
        start_btn.clicked.connect(self.start_session)
        goal_layout.addWidget(start_btn)

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

        # ===== Instruction area (steps) with ability to gray out when paused =====
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
        self.step_layout.addStretch()  # Push content to the top

        self.instruction_area.setWidget(self.step_container)
        instruction_layout.addWidget(self.instruction_area)

        self.layout.addWidget(self.instruction_container, 1)  # 1 = stretch factor

        # ===== Button controls =====
        self.button_panel = QFrame()
        self.button_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {LIGHT_GRAY};
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
        self.chat_btn.setEnabled(False)  # Initially disabled until paused
        self.chat_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {PRIMARY_COLOR};
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

        # Toggle explanation mode button
        """self.explanation_btn = QPushButton()
        self.explanation_btn.setFixedSize(50, 50)
        self.explanation_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.explanation_btn.setIcon(QIcon(os.path.join(ICON_DIR, "explanation.png")))
        self.explanation_btn.setIconSize(QSize(24, 24))
        self.explanation_btn.setToolTip("Toggle explanation mode")
        self.explanation_btn.setStyleSheet(f"
            QPushButton {{
                background-color: {PRIMARY_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 25px;
            }}
            QPushButton:hover {{
                background-color: {LIGHT_GRAY};
                border: 1px solid {ACCENT_COLOR};
            }}
        ")
        self.explanation_btn.clicked.connect(lambda: self.toggle_explanation_mode(not self.explanation_mode))
        button_layout.addWidget(self.explanation_btn)

        # Audio toggle button
        self.audio_btn = QPushButton()
        self.audio_btn.setFixedSize(50, 50)
        self.audio_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.audio_btn.setIcon(QIcon(os.path.join(ICON_DIR, "audio.png")))
        self.audio_btn.setIconSize(QSize(24, 24))
        self.audio_btn.setToolTip("Toggle audio mode")
        self.audio_btn.setStyleSheet(f"
            QPushButton {{
                background-color: {PRIMARY_COLOR};
                border: 1px solid {BORDER_COLOR};
                border-radius: 25px;
            }}
            QPushButton:hover {{
                background-color: {LIGHT_GRAY};
                border: 1px solid {ACCENT_COLOR};
            }}
        ")
        self.audio_btn.clicked.connect(lambda: self.toggle_audio_mode(not self.audio_mode))
        button_layout.addWidget(self.audio_btn)
"""
        # Add spacer to push buttons to the right
        button_layout.addStretch()

        # Pause button (icon button)
        self.pause_btn = QPushButton()
        self.pause_btn.setFixedSize(50, 50)
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.setIcon(QIcon(str(resource_path("icons/pause.png"))))

        self.pause_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BUTTON_ORANGE};
                color: white;
                border: none;
                border-radius: 25px;
                padding: 8px 18px;
            }}
            QPushButton:hover {{
                background-color: #808080;
            }}
        """)

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
                background-color: #ea580c;
            }}
        """)
        self.pause_btn.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_btn)

        # Stop button (icon button)
        self.stop_btn = QPushButton()
        self.stop_btn.setFixedSize(50, 50)
        self.stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stop_btn.setIcon(QIcon(os.path.join(ICON_DIR, "stop.png")))

        self.stop_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {BUTTON_RED};
                color: white;
                border: none;
                border-radius: 25px;
                padding: 8px 18px;
            }}
            QPushButton:hover {{
                background-color: #b91c1c;
            }}
        """)

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

        # Next button (text + icon)
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
                background-color: #2563eb;
            }}
            QPushButton:disabled {{
                background-color: {DISABLED_COLOR};
            }}
        """)
        self.next_btn.clicked.connect(self.next_step)
        button_layout.addWidget(self.next_btn)

        self.layout.addWidget(self.button_panel)

    def set_user(self, user_id):
        """Set the user ID and update welcome message"""
        self.user_id = user_id
        self.user_welcome.setText(f"Welcome, {user_id}")

    def set_course(self, course_id):
        """Set the course ID"""
        self.course_id = course_id
        # You could update UI to show the course ID if desired

    def toggle_explanation_mode(self, enabled):
        """Toggle explanation mode on/off"""
        self.explanation_mode = enabled

        # Update UI
        if hasattr(self, 'explanation_toggle'):
            self.explanation_toggle.setChecked(enabled)

        # Update button appearance
        if hasattr(self, 'explanation_btn'):
            if enabled:
                self.explanation_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {EXPLANATION_BG_COLOR};
                        border: 2px solid {EXPLANATION_BORDER_COLOR};
                        border-radius: 25px;
                    }}
                    QPushButton:hover {{
                        background-color: #fff7ed;
                    }}
                """)
            else:
                self.explanation_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {PRIMARY_COLOR};
                        border: 1px solid {BORDER_COLOR};
                        border-radius: 25px;
                    }}
                    QPushButton:hover {{
                        background-color: {LIGHT_GRAY};
                        border: 1px solid {ACCENT_COLOR};
                    }}
                """)

        # Add notification
        notification = {
            'instruction': f"✓ Explanation mode {'enabled' if enabled else 'disabled'}",
            'format': 'text',
            'system_step': True
        }
        self.add_instruction(notification)


        # Update training wheels if active
        if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
            import training_wheels as tw
            tw.toggle_explanation_mode(enabled)

            # Add step to inform user
            action = "enabled" if enabled else "disabled"
            self.add_instruction({
                'instruction': f"Explanation mode {action}. {('You will now receive explanations about why each step matters.' if enabled else 'Explanations will no longer be shown.')}",
                'format': 'text',
                'system_step': True
            })

    def toggle_audio_mode(self, enabled):
        """Toggle audio mode on/off"""
        self.audio_mode = enabled

        # Update UI
        if hasattr(self, 'audio_toggle'):
            self.audio_toggle.setChecked(enabled)

        # Update button appearance
        if hasattr(self, 'audio_btn'):
            if enabled:
                self.audio_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {ACCENT_COLOR};
                        border: 2px solid #1e40af;
                        border-radius: 25px;
                    }}
                    QPushButton:hover {{
                        background-color: #2563eb;
                    }}
                """)
            else:
                self.audio_btn.setStyleSheet(f"""
                    QPushButton {{
                        background-color: {PRIMARY_COLOR};
                        border: 1px solid {BORDER_COLOR};
                        border-radius: 25px;
                    }}
                    QPushButton:hover {{
                        background-color: {LIGHT_GRAY};
                        border: 1px solid {ACCENT_COLOR};
                    }}
                """)

        # Add notification
        notification = {
            'instruction': f"🔊 Audio mode {'enabled' if enabled else 'disabled'}",
            'format': 'text',
            'system_step': True
        }
        self.add_instruction(notification)

        # Update training wheels if active
        if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
            import training_wheels as tw
            tw.toggle_audio_mode(enabled)

            # Add step to inform user
            action = "enabled" if enabled else "disabled"
            self.add_instruction({
                'instruction': f"Audio mode {action}. {('Steps will now be read aloud.' if enabled else 'Steps will no longer be read aloud.')}",
                'format': 'text',
                'system_step': True
            })

    def start_session(self):
        """Start a new training session with synchronized step indexes"""
        goal = self.goal_input.text().strip()
        if goal:
            # Save goal and update UI state
            self.goal = goal
            # Start with step index 1 (NOT 0 - which is for system initialization only)
            self.step_index = 1
            self.is_session_active = True
            self.is_session_paused = False

            # Get explanation mode state
            self.explanation_mode = self.explanation_toggle.isChecked()

            # Get audio mode state
            self.audio_mode = self.audio_toggle.isChecked()

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

        self.steps = []  # Clear step references

    def add_instruction(self, content):
        """Add a new instruction step with proper step numbering

        Args:
            content: Can be a string or a dict with 'instruction' and 'format' keys
        """
        # Skip system steps in step numbering display
        is_system_step = isinstance(content, dict) and content.get('system_step', False)
        if is_system_step:
            display_step_index = None  # Don't show step number for system steps
        else:
            display_step_index = self.step_index

        # Reset style on previous current step (if any)
        for step in self.steps:
            step.setStyleSheet(f"""
                #stepItem {{
                    background-color: {PRIMARY_COLOR};
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
                    background-color: {PRIMARY_COLOR};
                    border: 1px solid {ACCENT_COLOR};
                    border-radius: 25px;
                }}
                QPushButton:hover {{
                    background-color: {LIGHT_GRAY};
                    border: 1px solid {ACCENT_COLOR};
                }}
            """)

            # Directly change the background color of the step container widget
            self.step_container.setStyleSheet(f"""
                background-color: {PAUSED_BG_COLOR};
            """)

            # Apply colored background to the instruction area
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

            # Also update the container background
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
                    background-color: #ea580c;
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
                    background-color: #808080;
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

            # Reset step container background
            self.step_container.setStyleSheet("")

            # Reset scroll area background
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

            # Restore normal background to instruction area
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
        """Open the chat dialog for asking questions with the current course ID"""
        if self.is_session_paused:
            # Ensure the dialog has access to the course_id
            if not hasattr(self.chat_dialog, 'course_id') and hasattr(self, 'course_id'):
                self.chat_dialog.course_id = self.course_id

            # Display a message if course_id is not set
            if not hasattr(self, 'course_id') or not self.course_id:
                self.chat_dialog.add_system_message("Warning: No course ID is set. Using default course.")
                self.chat_dialog.course_id = "001"  # Default course ID

            self.chat_dialog.show()



MainScreen = ui_integration.integrate_training_wheels(MainScreen)


class LearnChainTutor(QStackedWidget):
    """Main application container"""

    def __init__(self):
        super().__init__()

        # Set window icon FIRST
        #icon_path = str(resource_path("training_wheels/assets/icon.ico"))
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

        # Apply global style improvements
        self.setStyleSheet(f"""
            QWidget {{
                font-family: Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
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
                background: #b9b9b9;
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
        #icon_path = str(resource_path("training_wheels/assets/icon.ico"))
        icon_path = str(resource_path("assets/icon.ico"))
        if os.path.exists(icon_path):
            # Force set taskbar icon after window is visible
            QTimer.singleShot(100, lambda: set_taskbar_icon(self, icon_path))


class AnimatedStep(QFrame):
    """A single instruction step with smooth reveal animation and proper styling"""

    def __init__(self, step_number, content, is_current=False):
        super().__init__()
        self.setObjectName("stepItem")
        self.content = content

        if is_current:
            self.setStyleSheet(f"""
                #stepItem {{
                    background-color: #f1f5f9;
                    border-left: 4px solid {ACCENT_COLOR};
                    border-radius: 6px;
                    margin: 4px 0px;
                }}
            """)
        else:
            self.setStyleSheet(f"""
                #stepItem {{
                    background-color: {PRIMARY_COLOR};
                    border-left: 1px solid {BORDER_COLOR};
                    border-radius: 6px;
                    margin: 4px 0px;
                }}
            """)

        self.init_ui(step_number)
        self.start_animation()

    def init_ui(self, step_number):
        """Set up the UI for the step item"""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        step_indicator = QLabel(str(step_number))
        step_indicator.setObjectName("stepNumber")
        step_indicator.setFixedSize(30, 30)
        step_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_indicator.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        step_indicator.setStyleSheet(f"""
            #stepNumber {{
                background-color: {ACCENT_COLOR};
                color: white;
                border-radius: 15px;
                font-size: 13px;
            }}
        """)
        main_layout.addWidget(step_indicator)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(5)

        instruction_blocks = self.content.get("instruction", [])
        explanation_blocks = self.content.get("explanation", [])
        is_system_step = self.content.get("system_step", False)

        # Support fallback if instruction is a string (legacy)
        if isinstance(instruction_blocks, str):
            instruction_blocks = [{"type": "text", "content": instruction_blocks}]
        has_instruction = isinstance(instruction_blocks, list) and len(instruction_blocks) > 0

        # Show fallback label if no instruction
        if not has_instruction and not is_system_step:
            label = QLabel(f"Step {step_number}: No specific instruction provided")
            label.setStyleSheet("font-size: 13px; font-style: italic; color: #64748b;")
            content_layout.addWidget(label)

        # Render instruction blocks
        for block in instruction_blocks:
            btype = block.get("type")
            bcontent = block.get("content", "")
            blabel = block.get("label", "")

            if btype == "text":
                label = QLabel(bcontent)
                label.setWordWrap(True)
                label.setStyleSheet("font-size: 14px;")
                content_layout.addWidget(label)

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
                hbox = QHBoxLayout(frame)
                text = QLabel(f"<b>{blabel}</b><br><small style='color:#64748b'>{bcontent}</small>")
                text.setTextFormat(Qt.TextFormat.RichText)
                text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                text.setWordWrap(True)
                hbox.addWidget(text)

                btn = QPushButton("Copy Link")
                btn.setToolTip(f"Copy link: {bcontent}")
                btn.clicked.connect(lambda _, v=bcontent: QApplication.clipboard().setText(v))

                hbox.addWidget(btn)
                content_layout.addWidget(frame)

            elif btype == "code":
                code_edit = QPlainTextEdit()
                code_edit.setReadOnly(True)
                code_edit.setFont(QFont("Consolas, 'Courier New', monospace", 12))
                code_edit.setStyleSheet(f"""
                    QPlainTextEdit {{
                        background-color: #1e293b;
                        color: #e2e8f0;
                        border-radius: 4px;
                        padding: 8px;
                        border: none;
                    }}
                """)
                code_edit.setPlainText(bcontent)
                lines = bcontent.count('\n') + 1
                line_height = QFontMetrics(code_edit.font()).height()
                code_edit.setFixedHeight(min(lines, 15) * line_height + 30)
                content_layout.addWidget(code_edit)

        # Render explanation blocks
        if explanation_blocks:
            frame = QFrame()
            frame.setStyleSheet(f"""
                background-color: {EXPLANATION_BG_COLOR};
                border-left: 3px solid {EXPLANATION_BORDER_COLOR};
                border-radius: 4px;
                margin-top: 8px;
            """)
            exp_layout = QVBoxLayout(frame)
            exp_layout.setContentsMargins(12, 10, 12, 10)

            for block in explanation_blocks:
                if block.get("type") == "text":
                    elabel = QLabel(block.get("content", ""))
                    elabel.setWordWrap(True)
                    elabel.setStyleSheet("font-size: 13px; color: #78350f;")
                    exp_layout.addWidget(elabel)

            content_layout.addWidget(frame)

        main_layout.addLayout(content_layout, 1)



    def start_animation(self):
        """Start the reveal animation"""
        # Create opacity effect
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.opacity_effect.setOpacity(0)
        self.setGraphicsEffect(self.opacity_effect)

        # Create and configure animation
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(500)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        # Start the animation
        self.animation.start()


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

    app.setStyleSheet("""
        QWidget {
            background-color: #ffffff;
            color: #1e293b;
            font-family: Inter, sans-serif;
            font-size: 13px;
        }
        QLineEdit {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            padding: 8px;
        }
        QLineEdit:focus {
            border: 1px solid #dc2626;
        }
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