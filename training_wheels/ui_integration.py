# ui_integration.py - Fixed version with better error handling

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

# Choose the right enum for queued calls (Qt5 vs Qt6)
if hasattr(Qt, "ConnectionType"):
    QUEUED = Qt.ConnectionType.QueuedConnection
else:
    QUEUED = Qt.QueuedConnection

# Set up logging
logger = logging.getLogger(__name__)

# Import training wheels
import training_wheels as tw

# Use the same color scheme as learnchain_tutor.py
PRIMARY_COLOR = "#121212"
CARD_BACKGROUND = "rgba(30, 30, 30, 0.8)"
ACCENT_COLOR = "#FF3B30"
SECONDARY_COLOR = "#10b981"
TEXT_COLOR = "#FFFFFF"
SECONDARY_TEXT = "#CCCCCC"
LIGHT_GRAY = "#2A2A2A"
BORDER_COLOR = "#3A3A3A"
DISABLED_COLOR = "#555555"


class BackgroundWorker(QThread):
    """Background worker thread for handling knowledge retrieval with caching"""
    finished = pyqtSignal(object)  # Signal with knowledge results
    status_update = pyqtSignal(str)  # Signal for status updates
    error = pyqtSignal(str)  # Signal for errors

    # Simple cache for knowledge retrieval
    _knowledge_cache = {}

    def __init__(self, goal: str, course_id: str):
        super().__init__()
        self.goal = goal
        self.course_id = course_id

    def run(self):
        """Run the background task with caching"""
        try:
            # Create cache key
            cache_key = f"{self.course_id}_{hash(self.goal)}"
            
            # Check cache first
            if cache_key in self._knowledge_cache:
                logger.info(f"📦 Using cached knowledge for: {self.goal[:50]}...")
                self.status_update.emit("Using cached knowledge...")
                cached_knowledge = self._knowledge_cache[cache_key]
                self.status_update.emit(f"Found {len(cached_knowledge)} cached knowledge items")
                self.finished.emit(cached_knowledge)
                return

            self.status_update.emit("Fetching knowledge...")
            knowledge = fetch_knowledge_from_retriever(self.goal, self.course_id)
            
            # Cache the result for future use
            if knowledge:
                self._knowledge_cache[cache_key] = knowledge
                # Limit cache size to prevent memory issues
                if len(self._knowledge_cache) > 10:
                    oldest_key = next(iter(self._knowledge_cache))
                    del self._knowledge_cache[oldest_key]
            
            self.status_update.emit(f"Found {len(knowledge)} knowledge items")
            self.finished.emit(knowledge)
        except Exception as e:
            error_info = f"Error fetching knowledge: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_info)
            self.error.emit(str(e))
            self.finished.emit([])  # Return empty knowledge to keep things running


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
        container.setStyleSheet(f"""
            #loadingContainer {{
                background: {CARD_BACKGROUND};
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 16px;
            }}
        """)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(30, 30, 30, 30)
        container_layout.setSpacing(15)

        # Add loading label
        self.loading_label = QLabel("Loading...")
        self.loading_label.setFont(QFont("Inter", 14, QFont.Weight.Bold))
        self.loading_label.setStyleSheet(f"color: {ACCENT_COLOR};")
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.loading_label)

        # Add status label
        self.status_label = QLabel("Please wait")
        self.status_label.setFont(QFont("Inter", 11))
        self.status_label.setStyleSheet(f"color: {SECONDARY_TEXT};")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        container_layout.addWidget(self.status_label)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(85, 85, 85, 0.8);
                border-radius: 4px;
                border: none;
            }}
            QProgressBar::chunk {{
                background-color: {ACCENT_COLOR};
                border-radius: 4px;
            }}
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
        try:
            self.status_label.setText(status_text)
        except Exception as e:
            logger.error(f"Error updating status: {e}")

    def set_title(self, title_text):
        """Update the title/loading text"""
        try:
            self.loading_label.setText(title_text)
        except Exception as e:
            logger.error(f"Error updating title: {e}")


class GuidanceDisplay(QFrame):
    """Widget to display formatted guidance with code blocks"""

    def __init__(self, content, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: rgba(45, 45, 45, 0.6);
                border-radius: 8px;
                border: 1px solid {BORDER_COLOR};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)

        # Create text browser for HTML content
        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setStyleSheet(f"""
            QTextBrowser {{
                background-color: transparent;
                border: none;
                color: {TEXT_COLOR};
                font-size: 14px;
            }}
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

        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)

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
                      f'<code style="background-color: {LIGHT_GRAY}; padding: 2px 6px; border-radius: 3px; font-family: monospace; color: {ACCENT_COLOR};">\1</code>',
                      text)

        # Create HTML for code blocks
        for placeholder, language, code in code_blocks:
            code_html = self.create_code_block_html(code, language)
            text = text.replace(placeholder, code_html)

        # Make URLs clickable
        url_pattern = r'(https?://[^\s<>"{}|\\^`\[\]]+)'
        text = re.sub(url_pattern, f'<a href="\1" style="color: {ACCENT_COLOR}; text-decoration: underline;">\1</a>', text)

        # Convert newlines to <br>
        text = text.replace('\n', '<br>')

        return f"""
        <html>
        <body style="margin: 0; padding: 0; background-color: transparent; color: {TEXT_COLOR};">
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
            <div style="background-color: {LIGHT_GRAY}; padding: 4px 8px; border: 1px solid {BORDER_COLOR}; border-bottom: none; border-radius: 6px 6px 0 0; font-size: 12px; color: {SECONDARY_TEXT};">
                {language or 'code'}
            </div>
            <pre style="margin: 0; background-color: {LIGHT_GRAY}; padding: 10px; border: 1px solid {BORDER_COLOR}; border-radius: 0 0 6px 6px; overflow-x: auto; font-size: 13px; line-height: 1.45; color: {TEXT_COLOR};">
{code_escaped}</pre>
        </div>
        '''


# OPTIMIZED: Use the existing get_retriever_answer function with better performance
def fetch_knowledge_from_retriever(goal: str, course_id: str) -> List[Dict[str, Any]]:
    """Fetch knowledge using ultra-fast preemptive retriever with optimizations"""
    try:
        from retriever import get_retriever_answer, wait_for_full_initialization
        
        # Ensure preemptive initialization is complete
        logger.info("⏳ Ensuring preemptive initialization is complete...")
        ready = wait_for_full_initialization(timeout=5.0)
        
        if ready:
            logger.info("✅ MAXIMUM SPEED MODE: Preemptive initialization complete!")
        else:
            logger.warning("⚠️ Proceeding without full preemptive initialization")
        
        logger.info(f"🚀 ULTRA-FAST: Using get_retriever_answer for goal: '{goal}' (course: {course_id})")
        
        # Use maximum speed settings with reduced queries for faster response
        result = get_retriever_answer(
            question=goal,
            course_id=course_id,
            use_query_enhancement=True,
            use_enhanced_retrieval=True,
            use_multi_query=True,
            num_queries=3  # OPTIMIZED: Reduced from 5 to 3 for faster response
        )
        
        # Extract knowledge from results
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
        
        # Add individual query results if available (limit to top 3 for performance)
        individual_results = result.get("individual_results", [])
        for individual in individual_results[:3]:  # OPTIMIZED: Limit to top 3 results
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
            
            logger.info(f"🚀 PARALLEL retrieval via get_retriever_answer completed in {total_time:.2f}s!")
            logger.info(f"   → Processed {num_queries} queries simultaneously")
            logger.info(f"   → Got {valid_results} valid results")
        
        logger.info(f"Retrieved {len(all_knowledge)} knowledge items using get_retriever_answer")
        return all_knowledge
        
    except ImportError as e:
        logger.error(f"Could not import get_retriever_answer: {e}")
        return []
    except Exception as e:
        logger.error(f"Error in get_retriever_answer: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return []


def create_main_thread_timer(parent, interval, callback):
    """Create a timer that runs in the main thread"""
    timer = QTimer(parent)
    timer.timeout.connect(callback)
    timer.setInterval(interval)
    return timer


# Function to enhance the MainScreen class with Training Wheels
def enhance_main_screen(MainScreen):
    """Enhance the MainScreen class with Training Wheels features and performance optimizations"""

    # Add method to show goal input (no region selection anymore)
    def show_goal_input(self):
        """Show the goal input screen directly"""
        if hasattr(self, 'goal_input_section'):
            self.goal_input_section.setVisible(True)

    # Add this method to the MainScreen class
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
            try:
                self.loading_overlay.set_status(status_text)
            except Exception as e:
                logger.error(f"Error updating loading status: {e}")

    MainScreen.update_loading_status = update_loading_status

    # Keep the step list scrolled to the most-recent instruction
    def update_step_display(self):
        """Force the UI to scroll so the latest step is visible."""
        try:
            if (
                    hasattr(self, "step_layout")
                    and hasattr(self, "instruction_area")
                    and self.step_layout.count() >= 2
            ):
                last_widget = self.step_layout.itemAt(self.step_layout.count() - 2).widget()
                if last_widget:
                    self.instruction_area.ensureWidgetVisible(last_widget)
                    QApplication.processEvents()  # paint immediately
        except Exception as e:
            logger.error(f"Error updating step display: {e}")

    MainScreen.update_step_display = update_step_display

    # Define error handling function for knowledge retrieval
    def handle_knowledge_error(self, error_message):
        """Handle errors from knowledge retrieval"""
        try:
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
        except Exception as e:
            logger.error(f"Error in handle_knowledge_error: {e}")

    MainScreen.handle_knowledge_error = handle_knowledge_error

    # Define knowledge results handling function
    def handle_knowledge_results(self, knowledge):
        """Handle knowledge retrieval results"""
        try:
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
                    'instruction': "Could not find specific information for your goal. Will guide you based on general knowledge. You can still retry with a more specific goal.",
                    'format': 'text'
                })

            # Start guidance after knowledge is fetched
            tw.start_guidance()
        except Exception as e:
            logger.error(f"Error in handle_knowledge_results: {e}")
            # Add error instruction
            try:
                self.add_instruction({
                    'instruction': f"Error processing knowledge: {str(e)}",
                    'format': 'text'
                })
            except:
                pass

    MainScreen.handle_knowledge_results = handle_knowledge_results

    # Critical: Define the guidance handler method before any connections - FIXED VERSION
    def _handle_guidance(self, guidance):
        """Handle guidance signals from TrainingWheels - FIXED VERSION with robust data handling"""
        try:
            logger.info(f"Received guidance: {str(guidance)[:100]}...")

            # Validate and sanitize guidance structure
            if not isinstance(guidance, dict):
                logger.error(f"Invalid guidance format: {type(guidance)}")
                guidance = {
                    "instruction": str(guidance) if guidance else "Invalid guidance received",
                    "format": "text"
                }

            # Ensure required fields exist and are properly formatted
            if 'instruction' not in guidance:
                guidance['instruction'] = "No instruction provided"
            if 'format' not in guidance:
                guidance['format'] = 'text'

            # CRITICAL FIX: Handle the case where instruction is a list of blocks
            instruction_content = guidance.get('instruction')
            if isinstance(instruction_content, list):
                # Convert list of blocks to proper format for AnimatedStep
                logger.info("Instruction received as list of blocks - formatting properly")
                guidance['instruction'] = instruction_content  # Keep as list, AnimatedStep will handle it
            elif not isinstance(instruction_content, (str, dict)):
                # Convert any other type to string
                guidance['instruction'] = str(instruction_content)

            # Additional safety checks
            if guidance.get('format') not in ['text', 'code', 'json']:
                guidance['format'] = 'text'

            logger.info(f"Processed guidance format: {guidance.get('format')}, instruction type: {type(guidance.get('instruction'))}")

            # Keep guidance as-is, just pass it to add_instruction
            # This allows us to preserve the structured format
            self.add_instruction(guidance)

            # Ensure the UI scrolls to show the latest instruction
            self.update_step_display()
        except Exception as e:
            logger.error(f"Error handling guidance: {e}")
            logger.error(f"Guidance content: {guidance}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Add a visible error to the UI
            error_content = {
                "instruction": f"Error displaying instruction: {str(e)}",
                "format": "text"
            }
            try:
                self.add_instruction(error_content)
            except Exception as add_error:
                logger.error(f"Failed to add error instruction: {add_error}")
                # Last resort - try to show something
                try:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Error", f"Critical error displaying instruction: {str(e)}")
                except:
                    pass

    # Important: Add the method to MainScreen
    MainScreen._handle_guidance = _handle_guidance

    # Define a signal connection function - FIXED VERSION
    def connect_guidance_signal(self):
        """Ensure proper connection for guidance signal - FIXED VERSION"""
        try:
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
                        self.guidance_received.connect(self._handle_guidance, QUEUED)
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
        except Exception as e:
            logger.error(f"Error in connect_guidance_signal: {e}")
            return False

    MainScreen.connect_guidance_signal = connect_guidance_signal

    # Store the original init_ui method
    original_init_ui = MainScreen.init_ui

    def enhanced_init_ui(self):
        """Enhanced init_ui method with Training Wheels integration"""
        try:
            # Call the original method
            original_init_ui(self)

            # Create loading overlay
            self.loading_overlay = LoadingOverlay(self)
            self.loading_overlay.hide()

            # Create background worker for knowledge retrieval
            self.knowledge_worker = None

            # Modify next step button to use Training Wheels
            if hasattr(self, 'next_btn'):
                # Store the original next_step method
                original_next_step = self.next_step

                def enhanced_next_step():
                    """Enhanced next_step method that uses Training Wheels"""
                    try:
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
                    except Exception as e:
                        logger.error(f"Error in enhanced_next_step: {e}")

                # Helper method to re-enable button in the main thread
                def _reenable_next_button(self):
                    """Re-enable the next button after processing"""
                    try:
                        if hasattr(self, 'next_btn'):
                            self.next_btn.setEnabled(True)
                            self.next_btn.setText("Next Step")
                    except Exception as e:
                        logger.error(f"Error re-enabling next button: {e}")

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
        except Exception as e:
            logger.error(f"Error in enhanced_init_ui: {e}")

    # Replace the init_ui method
    MainScreen.init_ui = enhanced_init_ui

    # Enhance the start_session method to use Training Wheels
    original_start_session = MainScreen.start_session

    def enhanced_start_session(self):
        """Enhanced start_session method that initializes Training Wheels with optimizations"""
        try:
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
                if hasattr(self, 'status_check_timer'):
                    self.status_check_timer.start(30_000)
            except Exception as e:
                logger.error(f"Error starting session: {e}")
                self.add_instruction(
                    {"instruction": f"Error starting session: {e}", "format": "text"}
                )
                return

            # ── 7. kick off background knowledge worker with optimizations ─
            try:
                self.knowledge_worker = BackgroundWorker(self.goal, self.course_id)
                self.knowledge_worker.status_update.connect(self.update_loading_status)
                self.knowledge_worker.error.connect(self.handle_knowledge_error)
                self.knowledge_worker.finished.connect(self.handle_knowledge_results)
                self.knowledge_worker.start()
            except Exception as e:
                logger.error(f"Error starting knowledge worker: {e}")

            self.is_training_wheels_active = True
        except Exception as e:
            logger.error(f"Error in enhanced_start_session: {e}")
            # Try to show error to user
            try:
                self.add_instruction({
                    'instruction': f"Error starting training session: {str(e)}",
                    'format': 'text'
                })
            except:
                pass

    # Replace the start_session method
    MainScreen.start_session = enhanced_start_session

    # Enhance the toggle_pause method
    original_toggle_pause = MainScreen.toggle_pause

    def enhanced_toggle_pause(self):
        """Enhanced toggle_pause method that pauses/resumes Training Wheels"""
        try:
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
        except Exception as e:
            logger.error(f"Error in enhanced_toggle_pause: {e}")

    # Replace the toggle_pause method
    MainScreen.toggle_pause = enhanced_toggle_pause

    # Enhance the stop_session method
    original_stop_session = MainScreen.stop_session

    def enhanced_stop_session(self):
        """Enhanced stop_session method that ends Training Wheels session"""
        try:
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
        except Exception as e:
            logger.error(f"Error in enhanced_stop_session: {e}")

    # Replace the stop_session method
    MainScreen.stop_session = enhanced_stop_session

    # Set user method enhancement - automatically set fullscreen and go to goal input
    original_set_user = MainScreen.set_user

    def enhanced_set_user(self, user_id):
        """Enhanced set_user method that automatically sets fullscreen and shows goal input"""
        try:
            # Call original method
            original_set_user(self, user_id)

            # Automatically set fullscreen region
            try:
                # Get screen dimensions
                screen = QApplication.primaryScreen()
                geometry = screen.geometry()
                full_screen_region = (0, 0, geometry.width(), geometry.height())
                
                # Set the region in training wheels
                tw.set_capture_region(full_screen_region)
                
                logger.info(f"Auto-set fullscreen region: {full_screen_region}")
                
                # Go directly to goal input (no region selection screen)
                self.show_goal_input()
                
            except Exception as e:
                logger.error(f"Error setting fullscreen region: {e}")
                # Fallback to showing goal input anyway
                self.show_goal_input()
        except Exception as e:
            logger.error(f"Error in enhanced_set_user: {e}")

    # Replace the set_user method
    MainScreen.set_user = enhanced_set_user

    return MainScreen


# Function to integrate with the main application
def integrate_training_wheels(MainScreen):
    """Apply all Training Wheels enhancements to MainScreen"""
    MainScreen = enhance_main_screen(MainScreen)
    return MainScreen


# Trigger early initialization when UI starts
def trigger_early_initialization():
    """Trigger early initialization when UI starts"""
    try:
        import retriever
        logger.info("🚀 UI triggered preemptive initialization")
        # Non-blocking - just ensure it's started
        status = retriever.get_initialization_status()
        logger.info(f"📊 Initialization status: {status}")
    except Exception as e:
        logger.error(f"❌ Early initialization trigger failed: {e}")


if __name__ == "__main__":
    # Stand-alone test for S3 connection
    print("Testing S3 connection...")
    result = tw.test_s3_connection()
    print(result)

# Call immediately when module loads
trigger_early_initialization()