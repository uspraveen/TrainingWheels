# ui_integration.py

import os
import sys
import json
import logging
import traceback
from typing import Tuple, Optional, Dict, List, Any, Callable
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QDialog, QGraphicsOpacityEffect, QRubberBand, QMessageBox,
    QProgressBar, QSplashScreen
)
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QPixmap, QIcon
from PyQt6.QtWidgets import QTextBrowser
from PyQt6.QtCore import Qt, QRect, QSize, QPoint, QPropertyAnimation, QTimer, QMetaObject, QThread, pyqtSignal
# ---------- choose the right enum for queued calls (Qt5 vs Qt6) ----------
if hasattr(Qt, "ConnectionType"):                     # PyQt6 / PySide6
    QUEUED = Qt.ConnectionType.QueuedConnection
else:                                                 # PyQt5 / PySide2
    QUEUED = Qt.QueuedConnection
# -------------------------------------------------------------------------


# Set up logging
logger = logging.getLogger(__name__)

# Import training wheels
import training_wheels as tw


class BackgroundWorker(QThread):
    """Background worker thread for handling knowledge retrieval"""
    finished = pyqtSignal(object)  # Signal with knowledge results
    status_update = pyqtSignal(str)  # Signal for status updates
    error = pyqtSignal(str)  # Signal for errors

    def __init__(self, goal: str, course_id: str):
        super().__init__()
        self.goal = goal
        self.course_id = course_id

    def run(self):
        """Run the background task"""
        try:
            self.status_update.emit("Fetching knowledge...")
            knowledge = fetch_knowledge_from_retriever(self.goal, self.course_id)
            self.status_update.emit(f"Found {len(knowledge)} knowledge items")
            self.finished.emit(knowledge)
        except Exception as e:
            error_info = f"Error fetching knowledge: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_info)
            self.error.emit(str(e))
            self.finished.emit([])  # Return empty knowledge to keep things running


class RegionSelectionDialog(QDialog):
    """Dialog for selecting a screen region to monitor"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.setWindowTitle("Select Screen Region")

        # Make the dialog full screen with semi-transparency
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 100);")

        # Set up the rubber band for selection
        self.origin = QPoint()
        self.rubberBand = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.rubberBand.setStyleSheet("background-color: rgba(0, 120, 215, 50); border: 2px solid #0078d7;")

        # Create instruction label
        self.instruction = QLabel("Click and drag to select the screen area to monitor", self)
        self.instruction.setStyleSheet("background-color: #0078d7; color: white; padding: 10px; border-radius: 5px;")
        self.instruction.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.instruction.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.instruction.setFixedWidth(500)
        self.instruction.setGeometry(
            (self.width() - 500) // 2,
            50,
            500,
            50
        )

        # Cancel button
        self.cancel_btn = QPushButton("Cancel", self)
        self.cancel_btn.setStyleSheet("""
            background-color: #ef4444; 
            color: white; 
            padding: 10px; 
            border-radius: 5px;
            font-weight: bold;
        """)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setFixedSize(100, 40)
        self.cancel_btn.setGeometry(50, 50, 100, 40)

        # Preview label
        self.preview_label = QLabel("Screenshot Preview", self)
        self.preview_label.setStyleSheet("""
            background-color: #0078d7; 
            color: white; 
            padding: 5px; 
            border-radius: 5px;
            font-weight: bold;
        """)
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setFixedSize(200, 30)
        self.preview_label.move(50, 100)
        self.preview_label.hide()  # Initially hidden

        # Preview image
        self.preview_image = QLabel(self)
        self.preview_image.setStyleSheet("background-color: white; border: 1px solid #0078d7;")
        self.preview_image.setFixedSize(300, 200)
        self.preview_image.move(50, 140)
        self.preview_image.hide()  # Initially hidden

        # Confirm selection button
        self.confirm_btn = QPushButton("Confirm Selection", self)
        self.confirm_btn.setStyleSheet("""
            background-color: #10b981; 
            color: white; 
            padding: 10px; 
            border-radius: 5px;
            font-weight: bold;
        """)
        self.confirm_btn.clicked.connect(self.accept_selection)
        self.confirm_btn.setFixedSize(150, 40)
        self.confirm_btn.move(50, 350)
        self.confirm_btn.hide()  # Initially hidden

        # Selection result
        self.selected_region: Optional[Tuple[int, int, int, int]] = None

    def resizeEvent(self, event):
        """Handle resize events to position instruction label"""
        self.instruction.setGeometry(
            (self.width() - 500) // 2,
            50,
            500,
            50
        )
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press to start selection"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.position().toPoint()
            self.rubberBand.setGeometry(QRect(self.origin, QSize()))
            self.rubberBand.show()

            # Hide preview if showing a new selection
            self.preview_label.hide()
            self.preview_image.hide()
            self.confirm_btn.hide()

    def mouseMoveEvent(self, event):
        """Handle mouse move to resize selection"""
        if self.rubberBand.isVisible():
            # Calculate new rubber band geometry
            current_pos = event.position().toPoint()
            selection = QRect(self.origin, current_pos).normalized()
            self.rubberBand.setGeometry(selection)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete selection"""
        if event.button() == Qt.MouseButton.LeftButton and self.rubberBand.isVisible():
            # Get final selection rect
            selection_rect = self.rubberBand.geometry()

            # Ensure minimum size
            if selection_rect.width() < 50 or selection_rect.height() < 50:
                QMessageBox.warning(
                    self,
                    "Selection Too Small",
                    "Please select a larger area (at least 50x50 pixels)."
                )
                return

            # Store selected region as (x, y, width, height)
            self.selected_region = (
                selection_rect.x(),
                selection_rect.y(),
                selection_rect.width(),
                selection_rect.height()
            )

            # Take a test screenshot and show preview
            self.show_preview()

    def show_preview(self):
        """Show a preview of the selected region"""
        if not self.selected_region:
            return

        try:
            # Set the region in training wheels
            success = tw.set_capture_region(self.selected_region)

            if success:
                # Get the test screenshot path
                test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "training_data", "test_screenshots")

                # Find the most recent test screenshot
                if os.path.exists(test_dir):
                    files = [f for f in os.listdir(test_dir) if f.startswith("test_screenshot_")]
                    if files:
                        # Sort by name (contains timestamp)
                        files.sort(reverse=True)
                        latest = os.path.join(test_dir, files[0])

                        # Show the preview
                        pixmap = QPixmap(latest)
                        self.preview_image.setPixmap(pixmap.scaled(
                            300, 200,
                            Qt.AspectRatioMode.KeepAspectRatio
                        ))
                        self.preview_label.show()
                        self.preview_image.show()
                        self.confirm_btn.show()

                        logger.info(f"Showing preview from: {latest}")
            else:
                QMessageBox.warning(
                    self,
                    "Screenshot Failed",
                    "Failed to capture a test screenshot of the selected region."
                )

        except Exception as e:
            logger.error(f"Error showing preview: {e}")
            QMessageBox.warning(
                self,
                "Preview Error",
                f"Error capturing preview: {str(e)}"
            )

    def accept_selection(self):
        """Accept the current selection"""
        if self.selected_region:
            self.accept()


class LoadingOverlay(QWidget):
    """Overlay widget showing a loading indicator with status updates"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setParent(parent)

        # Make the overlay semi-transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Set up the overlay layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Add container for better styling
        container = QFrame()
        container.setObjectName("loadingContainer")
        container.setStyleSheet("""
            #loadingContainer {
                background-color: rgba(255, 255, 255, 0.9);
                border-radius: 10px;
                border: 1px solid #e2e8f0;
            }
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(30, 30, 30, 30)
        container_layout.setSpacing(15)

        # Add loading label
        self.loading_label = QLabel("Loading...")
        self.loading_label.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        self.loading_label.setStyleSheet("color: #ef4444;")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.loading_label)

        # Add status label
        self.status_label = QLabel("Please wait")
        self.status_label.setFont(QFont("Inter", 11))
        self.status_label.setStyleSheet("color: #64748b;")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.status_label)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #D3D3D3;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background-color: #ef4444;
                border-radius: 4px;
            }
        """)
        container_layout.addWidget(self.progress_bar)

        layout.addWidget(container)

        # Set fixed size for the container
        container.setFixedSize(300, 150)

        # Hide by default
        self.hide()

    def showEvent(self, event):
        """Position overlay when shown"""
        if self.parent():
            self.resize(self.parent().size())
        super().showEvent(event)

    def set_status(self, status_text):
        """Update the status text"""
        self.status_label.setText(status_text)

    def set_title(self, title_text):
        """Update the title/loading text"""
        self.loading_label.setText(title_text)


class GuidanceDisplay(QFrame):
    """Widget to display formatted guidance with code blocks"""

    def __init__(self, content, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)

        # Create text browser for HTML content
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                background-color: transparent;
                border: none;
                color: #1e293b;
                font-size: 14px;
            }
        """)

        # Format and set the content
        formatted_html = self.format_content(content.get('instruction', ''))
        self.text_browser.setHtml(formatted_html)

        # Adjust height based on content
        self.text_browser.document().setDocumentMargin(0)
        height = self.text_browser.document().size().height()
        self.text_browser.setMaximumHeight(int(height + 20))

        layout.addWidget(self.text_browser)

    def format_content(self, text):
        """Format text with code blocks and links"""
        import re
        import html

        # Store code blocks
        code_blocks = []
        code_counter = 0

        def store_code_block(match):
            nonlocal code_counter
            language = match.group(1) or ''
            code = match.group(2).strip()
            placeholder = f"__CODE_BLOCK_{code_counter}__"
            code_blocks.append((placeholder, language, code))
            code_counter += 1
            return placeholder

        # Replace code blocks with placeholders
        text = re.sub(r'```(\w*)\n?(.*?)```', store_code_block, text, flags=re.DOTALL)

        # Handle commands starting with $ or #
        lines = text.split('\n')
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('$') or (stripped.startswith('#') and not stripped.startswith('##')):
                lines[i] = f'```bash\n{line}\n```'
        text = '\n'.join(lines)
        text = re.sub(r'```(\w*)\n?(.*?)```', store_code_block, text, flags=re.DOTALL)

        # Escape HTML
        text = html.escape(text)

        # Format inline code
        text = re.sub(r'`([^`]+)`',
                      r'<code style="background-color: #e2e8f0; padding: 2px 6px; border-radius: 3px; font-family: monospace; color: #0969da;">\1</code>',
                      text)

        # Create HTML for code blocks
        for placeholder, language, code in code_blocks:
            code_html = self.create_code_block_html(code, language)
            text = text.replace(placeholder, code_html)

        # Make URLs clickable
        url_pattern = r'(https?://[^\s<>"{}|\\^`\[\]]+)'
        text = re.sub(url_pattern, r'<a href="\1" style="color: #0969da; text-decoration: underline;">\1</a>', text)

        # Convert newlines to <br>
        text = text.replace('\n', '<br>')

        return f"""
        <html>
        <body style="margin: 0; padding: 0;">
            {text}
        </body>
        </html>
        """

    def create_code_block_html(self, code, language):
        """Create HTML for a code block"""
        import html
        code_escaped = html.escape(code)

        if not language and (code.strip().startswith('$') or 'npm' in code or 'pip' in code):
            language = 'bash'

        return f'''
        <div style="margin: 10px 0; font-family: monospace;">
            <div style="background-color: #f6f8fa; padding: 4px 8px; border: 1px solid #d1d9e0; border-bottom: none; border-radius: 6px 6px 0 0; font-size: 12px; color: #57606a;">
                {language or 'code'}
            </div>
            <pre style="margin: 0; background-color: #f6f8fa; padding: 10px; border: 1px solid #d1d9e0; border-radius: 0 0 6px 6px; overflow-x: auto; font-size: 13px; line-height: 1.45;">
{code_escaped}</pre>
        </div>
        '''


class RegionSelectionScreen(QWidget):
    """Initial screen for region selection"""

    def __init__(self, main_screen):
        super().__init__()
        self.main_screen = main_screen
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(50, 80, 50, 80)
        layout.setSpacing(20)

        # Title
        title = QLabel("Set Up Training Wheels")
        title.setFont(QFont("Inter", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: #dc2626;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Before starting, please select the area of your screen you want to monitor. "
            "This area should include the application or browser window you'll be using for this training."
        )
        instructions.setFont(QFont("Inter", 12))
        instructions.setStyleSheet("color: #1e293b;")
        instructions.setWordWrap(True)
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Screenshot example
        example_frame = QFrame()
        example_frame.setStyleSheet("background-color: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0;")
        example_frame.setMinimumHeight(200)

        example_layout = QVBoxLayout(example_frame)

        example_label = QLabel("Example:")
        example_label.setFont(QFont("Inter", 10, QFont.Weight.Bold))
        example_label.setStyleSheet("color: #64748b;")
        example_layout.addWidget(example_label)

        # Show placeholder text
        placeholder = QLabel("You will see a preview of your selected region before confirming.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #94a3b8; font-style: italic;")
        example_layout.addWidget(placeholder)

        layout.addWidget(example_frame)

        # Spacer
        layout.addSpacing(20)

        # Select region button
        select_btn = QPushButton("Select Screen Region")
        select_btn.setFont(QFont("Inter", 13, QFont.Weight.Bold))
        select_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        select_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc2626;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
            }
            QPushButton:hover {
                background-color: #808080;
            }
        """)
        select_btn.clicked.connect(self.select_region)
        layout.addWidget(select_btn)

        # Skip button (for testing or development)
        skip_btn = QPushButton("Skip (Use Full Screen)")
        skip_btn.setFont(QFont("Inter", 11))
        skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        skip_btn.setStyleSheet("""
            QPushButton {
                background-color: #f8fafc;
                color: #64748b;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #808080;
                color: #334155;
            }
        """)
        skip_btn.clicked.connect(self.skip_selection)
        layout.addWidget(skip_btn)

        # S3 test button
        test_s3_btn = QPushButton("Test S3 Connection")
        test_s3_btn.setFont(QFont("Inter", 11))
        test_s3_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        test_s3_btn.setStyleSheet("""
            QPushButton {
                background-color: #f8fafc;
                color: #64748b;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #808080;
                color: #334155;
            }
        """)
        test_s3_btn.clicked.connect(self.test_s3)
        layout.addWidget(test_s3_btn)

        # Add stretching space at the bottom
        layout.addStretch()

    def select_region(self):
        """Open the region selection dialog"""
        # Minimize our window first to see the screen
        if self.window().isMaximized():
            self.window().showNormal()
        self.window().showMinimized()

        # Short delay to ensure window is minimized
        QTimer.singleShot(500, self._open_selection_dialog)

    def _open_selection_dialog(self):
        """Actually open the selection dialog after minimizing"""
        dialog = RegionSelectionDialog(self)
        result = dialog.exec()

        # Restore our window
        self.window().showNormal()

        if result == QDialog.DialogCode.Accepted and dialog.selected_region:
            # Set the region in the training wheels module
            tw.set_capture_region(dialog.selected_region)
            # Move to the next screen (goal input)
            self.main_screen.show_goal_input()
        else:
            # User cancelled, do nothing
            pass

    def skip_selection(self):
        """Skip region selection, use full screen instead"""
        try:
            # Get screen dimensions using more reliable methods
            if sys.platform.startswith('linux'):
                # On Linux, try to get accurate screen dimensions
                try:
                    import subprocess
                    # Try xrandr for X11
                    result = subprocess.run(['xrandr', '--current'], stdout=subprocess.PIPE, text=True)
                    output = result.stdout

                    # Parse the primary display dimensions
                    import re
                    match = re.search(r'(\d+)x(\d+)\+0\+0', output)
                    if match:
                        width, height = int(match.group(1)), int(match.group(2))
                        full_screen_region = (0, 0, width, height)
                        logger.info(f"Using xrandr detected screen dimensions: {width}x{height}")
                    else:
                        # Fallback to PyQt
                        screen = QApplication.primaryScreen()
                        geometry = screen.geometry()
                        # Apply scaling factor
                        scaling = screen.devicePixelRatio()
                        width = int(geometry.width() * scaling)
                        height = int(geometry.height() * scaling)
                        full_screen_region = (0, 0, width, height)
                        logger.info(f"Using PyQt screen dimensions with scaling: {width}x{height}")
                except Exception as e:
                    logger.warning(f"Failed to get screen dimensions with xrandr: {e}")
                    # Fallback to PyQt
                    screen = QApplication.primaryScreen()
                    geometry = screen.geometry()
                    full_screen_region = (0, 0, geometry.width(), geometry.height())
            else:
                # For other platforms, use PyQt
                screen = QApplication.primaryScreen()
                geometry = screen.geometry()
                full_screen_region = (0, 0, geometry.width(), geometry.height())

            logger.info(f"Setting full screen region: {full_screen_region}")
            tw.set_capture_region(full_screen_region)

            # Take a test screenshot to verify
            success, test_path = tw.get_instance().screenshot_manager.take_test_screenshot()
            if not success:
                QMessageBox.warning(
                    self,
                    "Screenshot Failed",
                    "Failed to capture a full screen screenshot. Please try manual selection instead."
                )
                return False

            # Move to the next screen
            self.main_screen.show_goal_input()
            return True
        except Exception as e:
            logger.error(f"Error setting full screen region: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to set full screen region: {str(e)}"
            )
            return False

    def test_s3(self):
        """Test S3 connection and show result"""
        result = tw.test_s3_connection()

        if result.get("connected", False):
            buckets = ", ".join(result.get("buckets", []))
            bucket_exists = result.get("bucket_exists", False)

            if bucket_exists:
                message = f"S3 connection successful! Found buckets: {buckets}\nSelected bucket exists."
                QMessageBox.information(self, "S3 Connection Test", message)
            else:
                message = f"S3 connection successful! Found buckets: {buckets}\nWARNING: Selected bucket does not exist."
                QMessageBox.warning(self, "S3 Connection Test", message)
        else:
            error = result.get("error", "Unknown error")
            QMessageBox.critical(self, "S3 Connection Test", f"S3 connection failed: {error}")


# BEST FIX: In ui_integration.py - Use the existing get_retriever_answer function

def fetch_knowledge_from_retriever(goal: str, course_id: str) -> List[Dict[str, Any]]:
    """Fetch knowledge using the proper get_retriever_answer function"""
    try:
        # Import the function that ALREADY handles course ID formatting correctly!
        from retriever import get_retriever_answer
        
        logger.info(f"🔧 BEST FIX: Using get_retriever_answer for goal: '{goal}' (course: {course_id})")
        
        # This function already:
        # 1. Adds "Course_" prefix correctly  
        # 2. Uses multi-query parallel processing
        # 3. Has all the optimizations
        result = get_retriever_answer(
            question=goal,
            course_id=course_id,  # Just pass "001" - function will format it correctly
            use_query_enhancement=True,
            use_enhanced_retrieval=True,
            use_multi_query=True,  # Enable parallel multi-query processing
            num_queries=10  # Use 10 diverse queries
        )
        
        # Extract knowledge from results (same format as multi_query_retrieval_with_individual_answers)
        all_knowledge = []
        
        # Add the main consolidated answer
        if result.get("answer") and "I don't have enough information" not in result.get("answer", ""):
            primary_knowledge = {
                "query": goal,
                "content": result["answer"],
                "type": "consolidated_answer", 
                "course_id": course_id,
                "sources": result.get("source_documents", []),
                "priority": 1
            }
            all_knowledge.append(primary_knowledge)
        
        # Add individual query results if available
        individual_results = result.get("individual_results", [])
        for individual in individual_results:
            if individual.get("has_info", False) and individual.get("answer"):
                knowledge_item = {
                    "query": individual["query"],
                    "content": individual["answer"], 
                    "type": "individual_answer",
                    "course_id": course_id,
                    "sources": individual.get("sources", []),
                    "priority": 2
                }
                all_knowledge.append(knowledge_item)
        
        # Log performance metrics
        metrics = result.get("performance_metrics", {})
        if metrics:
            total_time = metrics.get("total_time", 0)
            num_queries = metrics.get("num_queries", 0) 
            valid_results = metrics.get("num_valid_results", 0)
            course_id_used = metrics.get("course_id_used", "unknown")
            
            logger.info(f"🚀 PARALLEL retrieval via get_retriever_answer completed in {total_time:.2f}s!")
            logger.info(f"   → Processed {num_queries} queries simultaneously")
            logger.info(f"   → Got {valid_results} valid results")
            logger.info(f"   → Course ID used: {course_id_used}")
        
        logger.info(f"Retrieved {len(all_knowledge)} knowledge items using get_retriever_answer")
        return all_knowledge
        
    except ImportError as e:
        logger.error(f"Could not import get_retriever_answer: {e}")
        # Fallback to manual course ID formatting
        return fetch_knowledge_with_manual_formatting(goal, course_id)
    except Exception as e:
        logger.error(f"Error in get_retriever_answer: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def fetch_knowledge_with_manual_formatting(goal: str, course_id: str) -> List[Dict[str, Any]]:
    """Fallback with manual course ID formatting"""
    try:
        from retriever import multi_query_retrieval_with_individual_answers
        
        # Manual formatting as fallback
        if course_id and not course_id.startswith("Course_"):
            formatted_course_id = "Course_" + course_id
        else:
            formatted_course_id = course_id
            
        logger.info(f"🔧 FALLBACK: Manual formatting {course_id} → {formatted_course_id}")
        
        result = multi_query_retrieval_with_individual_answers(
            course_id=formatted_course_id,
            original_question=goal,
            num_queries=10,
            results_per_query=3,
            use_enhanced_retrieval=True,
            debug_mode=False
        )
        
        # Process results same as above...
        all_knowledge = []
        
        if result.get("answer") and "I don't have enough information" not in result.get("answer", ""):
            primary_knowledge = {
                "query": goal,
                "content": result["answer"],
                "type": "consolidated_answer", 
                "course_id": course_id,
                "sources": result.get("source_documents", []),
                "priority": 1
            }
            all_knowledge.append(primary_knowledge)
        
        individual_results = result.get("individual_results", [])
        for individual in individual_results:
            if individual.get("has_info", False) and individual.get("answer"):
                knowledge_item = {
                    "query": individual["query"],
                    "content": individual["answer"], 
                    "type": "individual_answer",
                    "course_id": course_id,
                    "sources": individual.get("sources", []),
                    "priority": 2
                }
                all_knowledge.append(knowledge_item)
        
        return all_knowledge
        
    except Exception as e:
        logger.error(f"Error in fallback formatting: {e}")
        return []


def create_main_thread_timer(parent, interval, callback):
    """Create a timer that runs in the main thread"""
    timer = QTimer(parent)
    timer.timeout.connect(callback)
    timer.setInterval(interval)
    return timer

# Function to enhance the MainScreen class with Training Wheels
# Function to enhance the MainScreen class with Training Wheels
# In ui_integration.py - Complete replacement for enhance_main_screen function

def enhance_main_screen(MainScreen):
    """Enhance the MainScreen class with Training Wheels features"""

    # Add method to show region selection screen
    def show_region_selection(self):
        """Show the region selection screen"""
        if hasattr(self, 'goal_input_section'):
            self.goal_input_section.setVisible(False)
        if hasattr(self, 'region_selection_screen'):
            self.region_selection_screen.setVisible(True)

    # Add method to show goal input
    def show_goal_input(self):
        """Show the goal input screen"""
        if hasattr(self, 'region_selection_screen'):
            self.region_selection_screen.setVisible(False)
        if hasattr(self, 'goal_input_section'):
            self.goal_input_section.setVisible(True)

    # Add these methods to the MainScreen class
    MainScreen.show_region_selection = show_region_selection
    MainScreen.show_goal_input = show_goal_input

    # Define session status checking function
    def check_session_status(self):
        """Periodically check session status"""
        if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
            status = tw.check_session_status()

            # Check for timeout
            if status.get("timed_out", False):
                # Notify user and ask to continue
                reply = QMessageBox.question(
                    self,
                    "Session Inactive",
                    "This session has been inactive for a while. Would you like to continue?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )

                if reply == QMessageBox.StandardButton.Yes:
                    # Resume if paused
                    if status.get("status") == "paused":
                        self.toggle_pause()

                    # Force a new step
                    tw.request_next_step()
                else:
                    # End session
                    self.stop_session()

    MainScreen.check_session_status = check_session_status

    # Define loading status update function
    def update_loading_status(self, status_text):
        """Update the loading overlay status"""
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.set_status(status_text)

    MainScreen.update_loading_status = update_loading_status

    # --- Keep the step list scrolled to the most-recent instruction ---
    def update_step_display(self):
        """Force the UI to scroll so the latest step is visible."""
        if (
                hasattr(self, "step_layout")
                and hasattr(self, "instruction_area")
                and self.step_layout.count() >= 2
        ):
            last_widget = self.step_layout.itemAt(self.step_layout.count() - 2).widget()
            self.instruction_area.ensureWidgetVisible(last_widget)
            QApplication.processEvents()  # paint immediately

    MainScreen.update_step_display = update_step_display

    # Define error handling function for knowledge retrieval
    def handle_knowledge_error(self, error_message):
        """Handle errors from knowledge retrieval"""
        # Hide loading overlay
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide()

        # Show error message
        QMessageBox.warning(
            self,
            "Knowledge Retrieval Error",
            f"There was an error retrieving knowledge: {error_message}\n\n"
            "The session will continue but may have limited guidance."
        )

        # Add an error step
        self.add_instruction({
            'instruction': f"Encountered an error while retrieving knowledge: {error_message}",
            'format': 'text'
        })

    MainScreen.handle_knowledge_error = handle_knowledge_error

    # Define knowledge results handling function
    def handle_knowledge_results(self, knowledge):
        """Handle knowledge retrieval results"""
        # Hide loading overlay
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide()

        # Add knowledge to training wheels
        if knowledge:
            tw.add_knowledge(knowledge)

            # Add an informational step
            self.add_instruction({
                'instruction': f"Found {len(knowledge)} knowledge items to help guide you through the process.",
                'format': 'text'
            })
        else:
            self.add_instruction({
                'instruction': "Could not find specific information for your goal. Will guide you based on general knowledge.",
                'format': 'text'
            })

        # Start guidance after knowledge is fetched
        tw.start_guidance()

    MainScreen.handle_knowledge_results = handle_knowledge_results

    # Critical: Define the guidance handler method before any connections
    def _handle_guidance(self, guidance):
        """Handle guidance signals from TrainingWheels"""
        try:
            logger.info(f"Received guidance: {str(guidance)[:50]}...")

            # Keep guidance as-is, just pass it to add_instruction
            # This allows us to preserve the structured format
            self.add_instruction(guidance)

            # Ensure the UI scrolls to show the latest instruction
            self.update_step_display()
        except Exception as e:
            logger.error(f"Error handling guidance: {e}")
            # Add a visible error to the UI
            error_content = {
                "instruction": f"Error displaying instruction: {str(e)}",
                "format": "text"
            }
            try:
                self.add_instruction(error_content)
            except Exception as add_error:
                logger.error(f"Failed to add error instruction: {add_error}")
    # Important: We need to correctly add the method to MainScreen
    # This is where the error was occurring
    MainScreen._handle_guidance = _handle_guidance

    # Define a signal connection function
    def connect_guidance_signal(self):
        """Ensure proper connection for guidance signal"""
        if hasattr(self, "guidance_received"):
            try:
                # Disconnect existing connections if any
                self.guidance_received.disconnect()
            except Exception as e:
                logger.debug(f"No previous connections to disconnect: {e}")
                pass  # It's okay if nothing was connected

            # Connect the signal to the handler
            try:
                # Verify the handler exists
                if hasattr(self, "_handle_guidance"):
                    self.guidance_received.connect(self._handle_guidance)
                    logger.info("Successfully connected guidance_received signal")
                    return True
                else:
                    logger.error("_handle_guidance method missing on MainScreen")
                    return False
            except Exception as e:
                logger.error(f"Failed to connect guidance_received signal: {e}")
                return False
        else:
            logger.error("MainScreen missing guidance_received signal")
            return False

    MainScreen.connect_guidance_signal = connect_guidance_signal

    # Store the original init_ui method
    original_init_ui = MainScreen.init_ui

    def enhanced_init_ui(self):
        """Enhanced init_ui method with Training Wheels integration"""
        # Call the original method
        original_init_ui(self)

        # Create region selection screen (initially hidden)
        self.region_selection_screen = RegionSelectionScreen(self)
        self.region_selection_screen.setVisible(False)

        # Create loading overlay
        self.loading_overlay = LoadingOverlay(self)
        self.loading_overlay.hide()

        # Add the new screen to the layout before the goal input section
        self.layout.insertWidget(1, self.region_selection_screen)

        # Create background worker for knowledge retrieval
        self.knowledge_worker = None

        # Modify next step button to use Training Wheels
        if hasattr(self, 'next_btn'):
            # Store the original next_step method
            original_next_step = self.next_step

            def enhanced_next_step():
                """Enhanced next_step method that uses Training Wheels"""
                if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
                    # Show a quick loading indicator
                    self.next_btn.setEnabled(False)
                    self.next_btn.setText("Processing...")

                    # Use Training Wheels for next step
                    try:
                        tw.request_next_step()
                    except Exception as e:
                        logger.error(f"Error requesting next step: {e}")

                    # Re-enable button after a short delay (using single shot timer)
                    QTimer.singleShot(1000, lambda: self._reenable_next_button())
                else:
                    # Use original method
                    original_next_step()

            # Helper method to re-enable button in the main thread
            def _reenable_next_button(self):
                """Re-enable the next button after processing"""
                self.next_btn.setEnabled(True)
                self.next_btn.setText("Next Step")

            # Add helper method to class
            self._reenable_next_button = _reenable_next_button.__get__(self)

            # Replace the next_step method
            self.next_step = enhanced_next_step

            # Update the next button click handler safely
            try:
                if hasattr(self.next_btn, 'clicked') and hasattr(self.next_btn.clicked, 'disconnect'):
                    self.next_btn.clicked.disconnect()
                    self.next_btn.clicked.connect(self.next_step)
            except Exception as e:
                logger.error(f"Error updating next button handler: {e}")

        # Add session status checker timer - create in main thread
        self.status_check_timer = create_main_thread_timer(self, 30000, self.check_session_status)
        # Don't start it yet - will start when session starts

    # Replace the init_ui method
    MainScreen.init_ui = enhanced_init_ui

    # Enhance the start_session method to use Training Wheels
    original_start_session = MainScreen.start_session

    def enhanced_start_session(self):
        """Enhanced start_session method that initializes Training Wheels"""
        # ── 1. keep original behaviour (shows goal etc.) ────────────────
        original_start_session(self)

        # ── 2. only continue if we have the needed attributes ───────────
        if not (hasattr(self, "user_id") and hasattr(self, "course_id") and hasattr(self, "goal")):
            return

        # ── 3. show the loading overlay full-window, on top ─────────────
        if hasattr(self, "loading_overlay"):
            self.loading_overlay.resize(self.size())
            self.loading_overlay.set_title("Setting up Training Wheels")
            self.loading_overlay.set_status("Initializing session…")
            self.loading_overlay.show()
            self.loading_overlay.raise_()
            QApplication.processEvents()

        # Add a visible "setting-up" line for the user
        self.add_instruction(
            {"instruction": "Setting up your training session…", "format": "text"}
        )

        # ── 4. Connect signal to handler - CRITICAL ─────────────────────
        # Ensure the handler method exists first (added above)
        if not hasattr(self, "_handle_guidance"):
            logger.error("_handle_guidance method missing - adding it now")
            # Add the method again as fallback
            self._handle_guidance = _handle_guidance.__get__(self)

        # Connect the signal
        result = self.connect_guidance_signal()
        if not result:
            logger.error("Failed to connect guidance signal - UI may not update correctly")
            # Add a visible error message
            self.add_instruction(
                {"instruction": "Warning: Signal connection failed - UI may not update automatically", "format": "text"}
            )

        # ── 5. callback passed to TrainingWheels (runs in worker thread) ─
        def step_callback(guidance: dict):
            """Callback from TrainingWheels that emits the guidance signal"""
            # Emit signal to GUI thread
            try:
                if hasattr(self, "guidance_received"):
                    logger.info(f"Emitting guidance_received signal with: {str(guidance)[:50]}...")
                    self.guidance_received.emit(guidance)
                else:
                    logger.error("Cannot emit guidance - signal not found!")
            except Exception as e:
                logger.error(f"Error in step_callback: {e}")

        # ── 6. start TrainingWheels session ─────────────────────────────
        try:
            session_id = tw.start_new_session(
                self.user_id,
                self.course_id,
                self.goal,
                step_callback,
                status_callback=getattr(self, "handle_tw_status_update", None),
            )

            # start the 30-sec status checker
            self.status_check_timer.start(30_000)
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            self.add_instruction(
                {"instruction": f"Error starting session: {e}", "format": "text"}
            )
            return

        # ── 7. kick off background knowledge worker ────────────────────
        self.knowledge_worker = BackgroundWorker(self.goal, self.course_id)
        self.knowledge_worker.status_update.connect(self.update_loading_status)
        self.knowledge_worker.error.connect(self.handle_knowledge_error)
        self.knowledge_worker.finished.connect(self.handle_knowledge_results)
        self.knowledge_worker.start()

        self.is_training_wheels_active = True

    # Replace the start_session method
    MainScreen.start_session = enhanced_start_session

    # Enhance the toggle_pause method
    original_toggle_pause = MainScreen.toggle_pause

    def enhanced_toggle_pause(self):
        """Enhanced toggle_pause method that pauses/resumes Training Wheels"""
        # Call original method
        original_toggle_pause(self)

        # Update Training Wheels state
        if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
            if self.is_session_paused:
                # Session is now paused
                tw.pause_guidance()

                # Add informational step
                self.add_instruction({
                    'instruction': "Session paused. Click Resume or Next Step when you're ready to continue.",
                    'format': 'text'
                })
            else:
                # Session is now resumed
                tw.resume_guidance()

                # Add informational step
                self.add_instruction({
                    'instruction': "Session resumed. Continuing guidance...",
                    'format': 'text'
                })

    # Replace the toggle_pause method
    MainScreen.toggle_pause = enhanced_toggle_pause

    # Enhance the stop_session method
    original_stop_session = MainScreen.stop_session

    def enhanced_stop_session(self):
        """Enhanced stop_session method that ends Training Wheels session"""
        # Check if Training Wheels is active
        had_tw_active = hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active

        if had_tw_active:
            # End Training Wheels session
            tw.end_guidance()
            self.is_training_wheels_active = False

            # Stop the status timer
            if hasattr(self, 'status_check_timer') and self.status_check_timer.isActive():
                self.status_check_timer.stop()

            # Also stop the knowledge worker if it's still running
            if hasattr(self, 'knowledge_worker') and self.knowledge_worker and self.knowledge_worker.isRunning():
                self.knowledge_worker.terminate()
                self.knowledge_worker.wait()

        # Call original method
        try:
            original_stop_session(self)
        except Exception as e:
            logger.error(f"Error in original stop_session: {e}")

    # Replace the stop_session method
    MainScreen.stop_session = enhanced_stop_session

    # Set user method enhancement to capture course_id
    original_set_user = MainScreen.set_user

    def enhanced_set_user(self, user_id):
        """Enhanced set_user method that shows region selection after login"""
        # Call original method
        original_set_user(self, user_id)

        # Add a debug log
        logger.info(f"Setting user: {user_id}, showing region selection screen")

        # Show region selection screen instead of goal input
        self.show_region_selection()

    # Replace the set_user method
    MainScreen.set_user = enhanced_set_user

    return MainScreen

# Function to integrate with the main application
def integrate_training_wheels(MainScreen):
    """Apply all Training Wheels enhancements to MainScreen"""
    MainScreen = enhance_main_screen(MainScreen)
    return MainScreen


if __name__ == "__main__":
    # Stand-alone test for S3 connection
    print("Testing S3 connection...")
    result = tw.test_s3_connection()
    print(result)