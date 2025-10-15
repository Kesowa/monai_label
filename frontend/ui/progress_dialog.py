"""
Progress Dialog for Long-Running Operations
Shows progress and status for uploads, predictions, and finetuning
"""

from typing import Optional
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QPushButton, QTextEdit, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QMovie, QPixmap


class ProgressDialog(QDialog):
    """Progress dialog for long-running operations"""
    
    # Signals
    cancelled = pyqtSignal()
    
    def __init__(self, title: str, parent=None, cancellable: bool = True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumSize(400, 250)
        self.setMaximumSize(600, 400)
        
        self.cancellable = cancellable
        self.is_cancelled = False
        
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Setup the progress dialog UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Title
        self.title_label = QLabel(self.windowTitle())
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Status text
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(25)
        layout.addWidget(self.progress_bar)
        
        # Detailed progress text (expandable)
        self.details_frame = QFrame()
        self.details_frame.setFrameStyle(QFrame.StyledPanel)
        self.details_frame.setVisible(False)
        details_layout = QVBoxLayout(self.details_frame)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(100)
        self.details_text.setReadOnly(True)
        self.details_text.setFont(QFont("Courier New", 9))
        details_layout.addWidget(self.details_text)
        
        layout.addWidget(self.details_frame)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        # Show/Hide details button
        self.details_btn = QPushButton("Show Details")
        self.details_btn.clicked.connect(self.toggle_details)
        button_layout.addWidget(self.details_btn)
        
        button_layout.addStretch()
        
        # Cancel button
        if self.cancellable:
            self.cancel_btn = QPushButton("Cancel")
            self.cancel_btn.clicked.connect(self.cancel_operation)
            button_layout.addWidget(self.cancel_btn)
        
        # Close button (initially disabled)
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
    def setup_styles(self):
        """Apply styles to the dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #f8f9fa;
            }
            
            QLabel {
                color: #2c3e50;
            }
            
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                text-align: center;
                background-color: #ecf0f1;
                font-weight: bold;
            }
            
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 6px;
                margin: 1px;
            }
            
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 80px;
            }
            
            QPushButton:hover {
                background-color: #2980b9;
            }
            
            QPushButton:pressed {
                background-color: #21618c;
            }
            
            QPushButton:disabled {
                background-color: #bdc3c7;
                color: #7f8c8d;
            }
            
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                font-family: 'Courier New', monospace;
            }
            
            QFrame {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
            }
        """)
        
    def set_progress(self, value: int, status_text: str = ""):
        """Set progress value and status text"""
        self.progress_bar.setValue(value)
        
        if status_text:
            self.status_label.setText(status_text)
        
        # Enable close button when complete
        if value >= 100:
            self.close_btn.setEnabled(True)
            if self.cancellable:
                self.cancel_btn.setEnabled(False)
                
    def set_indeterminate(self, indeterminate: bool = True):
        """Set progress bar to indeterminate mode"""
        if indeterminate:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            
    def add_detail(self, detail_text: str):
        """Add detail text to the expandable section"""
        current_text = self.details_text.toPlainText()
        new_text = f"{current_text}\n{detail_text}" if current_text else detail_text
        
        # Keep only last 100 lines to prevent memory issues
        lines = new_text.split('\n')
        if len(lines) > 100:
            lines = lines[-100:]
            new_text = '\n'.join(lines)
        
        self.details_text.setPlainText(new_text)
        
        # Scroll to bottom
        scrollbar = self.details_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def toggle_details(self):
        """Toggle visibility of details section"""
        if self.details_frame.isVisible():
            self.details_frame.setVisible(False)
            self.details_btn.setText("Show Details")
            self.resize(400, 250)
        else:
            self.details_frame.setVisible(True)
            self.details_btn.setText("Hide Details")
            self.resize(600, 400)
            
    def cancel_operation(self):
        """Cancel the current operation"""
        self.is_cancelled = True
        self.cancelled.emit()
        
        if self.cancellable:
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            self.status_label.setText("Cancelling operation...")
            
    def set_error(self, error_message: str):
        """Set error state"""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: #e74c3c; font-weight: bold;")
        
        self.close_btn.setEnabled(True)
        if self.cancellable:
            self.cancel_btn.setEnabled(False)
            
        self.add_detail(f"ERROR: {error_message}")
        
    def set_success(self, success_message: str = "Operation completed successfully!"):
        """Set success state"""
        self.progress_bar.setValue(100)
        self.status_label.setText(success_message)
        self.status_label.setStyleSheet("color: #27ae60; font-weight: bold;")
        
        self.close_btn.setEnabled(True)
        if self.cancellable:
            self.cancel_btn.setEnabled(False)
            
        self.add_detail(f"SUCCESS: {success_message}")


class FinetuningProgressDialog(ProgressDialog):
    """Specialized progress dialog for model finetuning"""
    
    def __init__(self, parent=None):
        super().__init__("Model Finetuning Progress", parent, cancellable=True)
        
        # Add finetuning-specific elements
        self.setup_finetuning_ui()
        
    def setup_finetuning_ui(self):
        """Add finetuning-specific UI elements"""
        # Insert epoch progress before main progress bar
        main_layout = self.layout()
        
        # Find the progress bar and insert epoch info before it
        progress_index = -1
        for i in range(main_layout.count()):
            item = main_layout.itemAt(i)
            if item.widget() == self.progress_bar:
                progress_index = i
                break
                
        if progress_index >= 0:
            # Epoch info
            epoch_layout = QHBoxLayout()
            epoch_layout.addWidget(QLabel("Epoch:"))
            self.epoch_label = QLabel("0 / 0")
            self.epoch_label.setStyleSheet("font-weight: bold; color: #3498db;")
            epoch_layout.addWidget(self.epoch_label)
            epoch_layout.addStretch()
            
            # Loss info
            epoch_layout.addWidget(QLabel("Loss:"))
            self.loss_label = QLabel("N/A")
            self.loss_label.setStyleSheet("font-weight: bold; color: #e67e22;")
            epoch_layout.addWidget(self.loss_label)
            
            main_layout.insertLayout(progress_index, epoch_layout)
            
            # Epoch progress bar
            self.epoch_progress = QProgressBar()
            self.epoch_progress.setRange(0, 100)
            self.epoch_progress.setValue(0)
            self.epoch_progress.setMaximumHeight(15)
            main_layout.insertWidget(progress_index + 1, self.epoch_progress)
            
    def update_epoch_progress(self, current_epoch: int, total_epochs: int, 
                            epoch_progress: float, current_loss: float = None):
        """Update epoch-specific progress"""
        self.epoch_label.setText(f"{current_epoch} / {total_epochs}")
        self.epoch_progress.setValue(int(epoch_progress))
        
        if current_loss is not None:
            self.loss_label.setText(f"{current_loss:.4f}")
            
        # Update overall progress
        overall_progress = ((current_epoch - 1) / total_epochs * 100) + (epoch_progress / total_epochs)
        self.set_progress(int(overall_progress), f"Training epoch {current_epoch} of {total_epochs}")