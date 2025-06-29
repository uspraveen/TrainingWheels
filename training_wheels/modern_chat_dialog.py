# modern_chat_dialog.py - Modern Chat Interface with Liquid Glass Design

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton,
    QLabel, QFrame, QScrollArea, QWidget, QCheckBox, QProgressBar,
    QGraphicsOpacityEffect, QGraphicsBlurEffect, QApplication
)
from PyQt6.QtGui import QFont, QPixmap, QIcon, QPainter, QColor, QBrush, QPen
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal

import json
import logging
from typing import Dict, List, Any

# Import modern color scheme
from learnchain_tutor import (
    GLASS_PRIMARY, GLASS_SECONDARY, GLASS_ACCENT, GLASS_ACCENT_DIM,
    GLASS_SURFACE, GLASS_TEXT_PRIMARY, GLASS_TEXT_SECONDARY, GLASS_BORDER,
    GLASS_OVERLAY, GLASS_BLUR_OVERLAY, LiquidGlassWidget, GlassButton, GlassInput
)

logger = logging.getLogger(__name__)

class ModernChatMessage(LiquidGlassWidget):
    """Modern chat message widget with liquid glass styling"""
    
    def __init__(self, message: str, role: str, parent=None):
        super().__init__(blur_radius=8, opacity=0.9, parent=parent)
        self.message = message
        self.role = role
        self.setup_ui()
        self.animate_entrance()
    
    def setup_ui(self):
        """Setup message UI based on role"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)
        
        # Role header
        header_layout = QHBoxLayout()
        
        # Role icon and label
        if self.role == "user":
            icon = "👤"
            role_name = "You"
            role_color = GLASS_ACCENT
            self.setStyleSheet(f"""
                LiquidGlassWidget {{
                    background: {GLASS_BLUR_OVERLAY};
                    border: 1px solid {GLASS_ACCENT};
                    border-radius: 16px;
                }}
            """)
        else:  # assistant
            icon = "🤖"
            role_name = "Assistant"
            role_color = GLASS_TEXT_PRIMARY
            self.setStyleSheet(f"""
                LiquidGlassWidget {{
                    background: rgba(22, 22, 24, 0.8);
                    border: 1px solid {GLASS_BORDER};
                    border-radius: 16px;
                }}
            """)
        
        role_label = QLabel(f"{icon} {role_name}")
        role_label.setStyleSheet(f"""
            QLabel {{
                color: {role_color};
                font-size: 14px;
                font-weight: 600;
                background: transparent;
                border: none;
            }}
        """)
        header_layout.addWidget(role_label)
        header_layout.addStretch()
        
        layout.addLayout(header_layout)
        
        # Message content
        content_label = QLabel(self.message)
        content_label.setWordWrap(True)
        content_label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_PRIMARY};
                font-size: 14px;
                line-height: 1.5;
                background: transparent;
                border: none;
                padding: 4px 0px;
            }}
        """)
        layout.addWidget(content_label)
    
    def animate_entrance(self):
        """Animate message entrance"""
        self.opacity_effect = QGraphicsOpacityEffect()
        self.setGraphicsEffect(self.opacity_effect)
        
        self.animation = QPropertyAnimation(self.opacity_effect, b"opacity")
        self.animation.setDuration(400)
        self.animation.setStartValue(0)
        self.animation.setEndValue(1)
        self.animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.animation.start()

class ModernThinkingIndicator(LiquidGlassWidget):
    """Modern thinking indicator with animated dots"""
    
    def __init__(self, parent=None):
        super().__init__(blur_radius=8, opacity=0.9, parent=parent)
        self.setup_ui()
        self.start_animation()
    
    def setup_ui(self):
        """Setup thinking indicator UI"""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        
        # Assistant icon
        icon_label = QLabel("🤖")
        icon_label.setStyleSheet(f"""
            QLabel {{
                font-size: 16px;
                background: transparent;
                border: none;
            }}
        """)
        layout.addWidget(icon_label)
        
        # Thinking text
        thinking_label = QLabel("Assistant is thinking")
        thinking_label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_SECONDARY};
                font-size: 14px;
                font-weight: 500;
                background: transparent;
                border: none;
            }}
        """)
        layout.addWidget(thinking_label)
        
        # Animated dots
        self.dots_label = QLabel("●●●")
        self.dots_label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_ACCENT};
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border: none;
            }}
        """)
        layout.addWidget(self.dots_label)
        
        layout.addStretch()
        
        self.setStyleSheet(f"""
            LiquidGlassWidget {{
                background: rgba(22, 22, 24, 0.6);
                border: 1px solid {GLASS_BORDER};
                border-radius: 16px;
            }}
        """)
    
    def start_animation(self):
        """Start dot animation"""
        self.dots_patterns = ["●  ", "●● ", "●●●", " ●●", "  ●", "   "]
        self.current_pattern = 0
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dots)
        self.timer.start(400)
    
    def update_dots(self):
        """Update dots pattern"""
        self.dots_label.setText(self.dots_patterns[self.current_pattern])
        self.current_pattern = (self.current_pattern + 1) % len(self.dots_patterns)
    
    def stop_animation(self):
        """Stop dot animation"""
        if hasattr(self, 'timer'):
            self.timer.stop()

class ModernChatDialog(QDialog):
    """Modern chat dialog with liquid glass design"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_widget = parent
        self.is_processing = False
        self.conversation_count = 0
        
        # Retriever settings
        self.use_query_enhancement = True
        self.use_enhanced_retrieval = True
        self.use_multi_query = True
        
        self.setup_modern_dialog()
        self.setup_ui()
    
    def setup_modern_dialog(self):
        """Setup modern dialog properties"""
        self.setWindowTitle("Chat with Assistant")
        self.setMinimumSize(500, 700)
        self.setModal(True)
        
        self.setStyleSheet(f"""
            QDialog {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {GLASS_PRIMARY}, stop:1 {GLASS_SECONDARY});
                border-radius: 16px;
            }}
        """)
    
    def setup_ui(self):
        """Setup modern chat UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(20)
        
        # Header
        self.create_header(layout)
        
        # Chat area
        self.create_chat_area(layout)
        
        # Settings panel
        self.create_settings_panel(layout)
        
        # Input area
        self.create_input_area(layout)
        
        # Button row
        self.create_button_row(layout)
        
        # Add welcome message
        self.add_assistant_message("Hello! I'm here to help with your training session. What can I assist you with?")
    
    def create_header(self, layout):
        """Create modern header"""
        header = LiquidGlassWidget(blur_radius=10, opacity=0.9)
        header.setFixedHeight(70)
        
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 0, 24, 0)
        
        # Title with icon
        title_layout = QHBoxLayout()
        
        icon_label = QLabel("💬")
        icon_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                background: transparent;
                border: none;
            }}
        """)
        title_layout.addWidget(icon_label)
        
        title_label = QLabel("Chat Assistant")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_PRIMARY};
                font-size: 20px;
                font-weight: 700;
                background: transparent;
                border: none;
            }}
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        header_layout.addLayout(title_layout)
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_SECONDARY};
                font-size: 14px;
                background: transparent;
                border: none;
            }}
        """)
        header_layout.addWidget(self.status_label)
        
        layout.addWidget(header)
    
    def create_chat_area(self, layout):
        """Create chat message area"""
        # Chat container
        self.chat_container = QScrollArea()
        self.chat_container.setWidgetResizable(True)
        self.chat_container.setFrameShape(QFrame.Shape.NoFrame)
        self.chat_container.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chat_container.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                background: rgba(255, 255, 255, 0.1);
                width: 8px;
                border-radius: 4px;
                margin: 0;
            }}
            QScrollBar::handle:vertical {{
                background: {GLASS_ACCENT};
                border-radius: 4px;
                min-height: 20px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        
        # Messages widget
        self.messages_widget = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_widget)
        self.messages_layout.setContentsMargins(0, 0, 0, 0)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch()
        
        self.chat_container.setWidget(self.messages_widget)
        layout.addWidget(self.chat_container, 1)
    
    def create_settings_panel(self, layout):
        """Create settings panel"""
        settings_panel = LiquidGlassWidget(blur_radius=8, opacity=0.85)
        settings_layout = QVBoxLayout(settings_panel)
        settings_layout.setContentsMargins(20, 16, 20, 16)
        settings_layout.setSpacing(12)
        
        # Settings title
        settings_title = QLabel("🔧 Retrieval Settings")
        settings_title.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_PRIMARY};
                font-size: 14px;
                font-weight: 600;
                background: transparent;
                border: none;
            }}
        """)
        settings_layout.addWidget(settings_title)
        
        # Settings toggles
        toggles_layout = QHBoxLayout()
        toggles_layout.setSpacing(20)
        
        # Checkbox styling
        checkbox_style = f"""
            QCheckBox {{
                color: {GLASS_TEXT_PRIMARY};
                font-size: 12px;
                background: transparent;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 3px;
                border: 1px solid {GLASS_BORDER};
                background: {GLASS_SURFACE};
            }}
            QCheckBox::indicator:checked {{
                background: {GLASS_ACCENT};
                border: 1px solid {GLASS_ACCENT};
            }}
            QCheckBox::indicator:checked::after {{
                content: "✓";
                color: {GLASS_PRIMARY};
                font-weight: bold;
            }}
        """
        
        # Query enhancement toggle
        self.enhance_toggle = QCheckBox("Query Enhancement")
        self.enhance_toggle.setChecked(True)
        self.enhance_toggle.setStyleSheet(checkbox_style)
        self.enhance_toggle.stateChanged.connect(self.update_settings)
        toggles_layout.addWidget(self.enhance_toggle)
        
        # Enhanced retrieval toggle
        self.retrieval_toggle = QCheckBox("Enhanced Retrieval")
        self.retrieval_toggle.setChecked(True)
        self.retrieval_toggle.setStyleSheet(checkbox_style)
        self.retrieval_toggle.stateChanged.connect(self.update_settings)
        toggles_layout.addWidget(self.retrieval_toggle)
        
        # Multi-query toggle
        self.multi_query_toggle = QCheckBox("Multi-Query")
        self.multi_query_toggle.setChecked(True)
        self.multi_query_toggle.setStyleSheet(checkbox_style)
        self.multi_query_toggle.stateChanged.connect(self.update_settings)
        toggles_layout.addWidget(self.multi_query_toggle)
        
        settings_layout.addLayout(toggles_layout)
        layout.addWidget(settings_panel)
    
    def create_input_area(self, layout):
        """Create input area"""
        input_container = QWidget()
        input_container.setStyleSheet("background: transparent;")
        input_layout = QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(8)
        
        # Progress indicator (hidden by default)
        self.progress_container = LiquidGlassWidget(blur_radius=6, opacity=0.9)
        self.progress_container.setFixedHeight(40)
        self.progress_container.setVisible(False)
        
        progress_layout = QHBoxLayout(self.progress_container)
        progress_layout.setContentsMargins(16, 0, 16, 0)
        
        self.progress_label = QLabel("Processing your question...")
        self.progress_label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_SECONDARY};
                font-size: 13px;
                background: transparent;
                border: none;
            }}
        """)
        progress_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {GLASS_SURFACE};
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background: {GLASS_ACCENT};
                border-radius: 2px;
            }}
        """)
        progress_layout.addWidget(self.progress_bar)
        
        input_layout.addWidget(self.progress_container)
        
        # Message input
        self.message_input = GlassInput("Type your question here...")
        self.message_input.setFixedHeight(50)
        self.message_input.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.message_input)
        
        layout.addWidget(input_container)
    
    def create_button_row(self, layout):
        """Create button row"""
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)
        
        button_layout.addStretch()
        
        # Send button
        self.send_btn = GlassButton("Send", "primary")
        self.send_btn.setFixedHeight(44)
        self.send_btn.setMinimumWidth(80)
        self.send_btn.clicked.connect(self.send_message)
        button_layout.addWidget(self.send_btn)
        
        # Close button
        self.close_btn = GlassButton("Close", "secondary")
        self.close_btn.setFixedHeight(44)
        self.close_btn.setMinimumWidth(80)
        self.close_btn.clicked.connect(self.close_chat)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def update_settings(self):
        """Update retriever settings"""
        self.use_query_enhancement = self.enhance_toggle.isChecked()
        self.use_enhanced_retrieval = self.retrieval_toggle.isChecked()
        self.use_multi_query = self.multi_query_toggle.isChecked()
        
        # Show system message
        settings_status = []
        if self.use_query_enhancement:
            settings_status.append("Query Enhancement")
        if self.use_enhanced_retrieval:
            settings_status.append("Enhanced Retrieval")
        if self.use_multi_query:
            settings_status.append("Multi-Query")
        
        if settings_status:
            self.add_system_message(f"Settings updated: {', '.join(settings_status)} enabled")
        else:
            self.add_system_message("All enhancement settings disabled")
    
    def add_conversation_header(self, number):
        """Add conversation separator"""
        separator = LiquidGlassWidget(blur_radius=5, opacity=0.7)
        separator.setFixedHeight(30)
        
        sep_layout = QHBoxLayout(separator)
        sep_layout.setContentsMargins(16, 0, 16, 0)
        
        line1 = QFrame()
        line1.setFrameStyle(QFrame.Shape.HLine)
        line1.setStyleSheet(f"color: {GLASS_BORDER};")
        sep_layout.addWidget(line1)
        
        label = QLabel(f"Conversation {number}")
        label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_SECONDARY};
                font-size: 12px;
                font-weight: 600;
                background: transparent;
                border: none;
                padding: 0 12px;
            }}
        """)
        sep_layout.addWidget(label)
        
        line2 = QFrame()
        line2.setFrameStyle(QFrame.Shape.HLine)
        line2.setStyleSheet(f"color: {GLASS_BORDER};")
        sep_layout.addWidget(line2)
        
        # Insert before stretch
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, separator)
        self.scroll_to_bottom()
    
    def add_user_message(self, message):
        """Add user message"""
        user_msg = ModernChatMessage(message, "user")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, user_msg)
        self.scroll_to_bottom()
        
        # Add to training wheels if available
        if hasattr(self.parent_widget, 'add_chat_to_session'):
            self.parent_widget.add_chat_to_session('user', message)
    
    def add_assistant_message(self, message):
        """Add assistant message"""
        # Remove thinking indicator if present
        self.remove_thinking_indicator()
        
        # Format message for better display
        formatted_message = self.format_message_content(message)
        
        assistant_msg = ModernChatMessage(formatted_message, "assistant")
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, assistant_msg)
        self.scroll_to_bottom()
        
        # Add to training wheels if available
        if hasattr(self.parent_widget, 'add_chat_to_session'):
            self.parent_widget.add_chat_to_session('assistant', message)
    
    def add_system_message(self, message):
        """Add system message"""
        system_msg = LiquidGlassWidget(blur_radius=5, opacity=0.7)
        system_msg.setFixedHeight(32)
        
        msg_layout = QHBoxLayout(system_msg)
        msg_layout.setContentsMargins(16, 0, 16, 0)
        
        label = QLabel(f"ℹ️ {message}")
        label.setStyleSheet(f"""
            QLabel {{
                color: {GLASS_TEXT_SECONDARY};
                font-size: 12px;
                font-style: italic;
                background: transparent;
                border: none;
            }}
        """)
        msg_layout.addWidget(label)
        
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, system_msg)
        self.scroll_to_bottom()
    
    def add_thinking_indicator(self):
        """Add thinking indicator"""
        self.thinking_indicator = ModernThinkingIndicator()
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, self.thinking_indicator)
        self.scroll_to_bottom()
    
    def remove_thinking_indicator(self):
        """Remove thinking indicator"""
        if hasattr(self, 'thinking_indicator'):
            self.thinking_indicator.stop_animation()
            self.thinking_indicator.setParent(None)
            delattr(self, 'thinking_indicator')
    
    def format_message_content(self, message):
        """Format message content for better display"""
        # Handle numbered lists
        lines = message.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Check if it's a numbered list item
                import re
                if re.match(r'^\d+\.\s', line):
                    formatted_lines.append(f"  {line}")  # Indent list items
                else:
                    formatted_lines.append(line)
        
        return '\n'.join(formatted_lines) if formatted_lines else message
    
    def scroll_to_bottom(self):
        """Scroll to bottom of chat"""
        QTimer.singleShot(50, lambda: self.chat_container.verticalScrollBar().setValue(
            self.chat_container.verticalScrollBar().maximum()
        ))
    
    def send_message(self):
        """Send message with modern processing"""
        if self.is_processing:
            return
        
        message = self.message_input.text().strip()
        if not message:
            return
        
        # Set processing state
        self.is_processing = True
        self.message_input.clear()
        self.message_input.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("Sending...")
        
        # Show progress
        self.progress_container.setVisible(True)
        self.status_label.setText("Processing...")
        
        # Add conversation header
        self.conversation_count += 1
        self.add_conversation_header(self.conversation_count)
        
        # Add user message
        self.add_user_message(message)
        
        # Add thinking indicator
        self.add_thinking_indicator()
        
        # Simulate processing with timer
        QTimer.singleShot(100, lambda: self.process_message(message))
    
    def process_message(self, message):
        """Process message using retriever"""
        try:
            # Get course ID
            course_id = getattr(self.parent_widget, 'course_id', "001")
            
            # Import retriever
            from retriever import query_course_knowledge
            
            # Query knowledge
            result = query_course_knowledge(
                course_id=course_id,
                question=message,
                use_query_enhancement=self.use_query_enhancement,
                use_enhanced_retrieval=self.use_enhanced_retrieval,
                k=10,
                use_parallel_processing=True
            )
            
            answer = result.get("answer", "I'm unable to find an answer to that question.")
            sources = result.get("source_documents", [])
            
            # Add assistant response
            self.add_assistant_message(answer)
            
            # Add sources if available
            if sources:
                sources_text = "Sources: " + ", ".join(sources[:3])  # Show top 3 sources
                self.add_system_message(sources_text)
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.add_assistant_message(
                "I encountered an error while processing your question. Please try again.")
        
        finally:
            # Reset processing state
            self.is_processing = False
            self.message_input.setEnabled(True)
            self.send_btn.setEnabled(True)
            self.send_btn.setText("Send")
            self.progress_container.setVisible(False)
            self.status_label.setText("Ready")
    
    def close_chat(self):
        """Close chat dialog"""
        self.hide()
    
    def closeEvent(self, event):
        """Handle close event"""
        self.remove_thinking_indicator()
        self.hide()
        event.ignore()  # Don't actually close, just hide

# Integration function to add to existing MainScreen
def add_modern_chat_to_main_screen(MainScreen):
    """Add modern chat functionality to MainScreen"""
    
    # Store original methods
    original_init_ui = MainScreen.init_ui
    
    def enhanced_init_ui(self):
        """Enhanced init_ui with modern chat"""
        original_init_ui(self)
        
        # Create modern chat dialog
        self.chat_dialog = ModernChatDialog(self)
        
        # Update chat button if it exists
        if hasattr(self, 'chat_btn'):
            self.chat_btn.clicked.disconnect()  # Disconnect old handler
            self.chat_btn.clicked.connect(self.open_modern_chat)
    
    def open_modern_chat(self):
        """Open modern chat dialog"""
        if hasattr(self, 'is_session_paused') and self.is_session_paused:
            if not hasattr(self, 'course_id') or not self.course_id:
                self.chat_dialog.add_system_message("Warning: No course ID set. Using default course.")
                self.chat_dialog.course_id = "001"
            
            self.chat_dialog.show()
            self.chat_dialog.raise_()
            self.chat_dialog.activateWindow()
    
    def add_chat_to_session(self, role: str, content: str):
        """Add chat interaction to training wheels session"""
        if hasattr(self, 'is_training_wheels_active') and self.is_training_wheels_active:
            import training_wheels as tw
            tw.add_chat_interaction(role, content)
    
    # Add methods to MainScreen
    MainScreen.init_ui = enhanced_init_ui
    MainScreen.open_modern_chat = open_modern_chat
    MainScreen.add_chat_to_session = add_chat_to_session
    
    return MainScreen

# Example usage:
# To integrate into existing code, call:
# MainScreen = add_modern_chat_to_main_screen(MainScreen)