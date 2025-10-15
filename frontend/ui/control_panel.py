"""
Control Panel Widget
Contains all the main controls for the application
"""

from typing import Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
    QLabel, QLineEdit, QComboBox, QSpinBox, QProgressBar,
    QTextEdit, QCheckBox, QFrame, QSizePolicy, QSpacerItem
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette


class ControlPanel(QWidget):
    """Main control panel with all application controls"""
    
    # Signals
    predict_requested = pyqtSignal()                    # Request prediction
    finetune_requested = pyqtSignal(str)               # Request finetuning with model name
    model_switch_requested = pyqtSignal(str)           # Request model switch
    
    def __init__(self):
        super().__init__()
        self.setMinimumWidth(350)
        self.setMaximumWidth(400)
        
        # State variables
        self.backend_connected = False
        self.image_loaded = False
        self.current_model = "default"
        self.annotation_count = 0
        
        self.setup_ui()
        self.setup_styles()
        
    def setup_ui(self):
        """Setup the control panel UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("🌳 AI Tree Detection Control Panel")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 10px 0px;")
        layout.addWidget(title_label)
        
        # Connection status
        self.connection_group = self.create_connection_status_group()
        layout.addWidget(self.connection_group)
        
        # Image information
        self.image_info_group = self.create_image_info_group()
        layout.addWidget(self.image_info_group)
        
        # AI Processing controls
        self.ai_group = self.create_ai_processing_group()
        layout.addWidget(self.ai_group)
        
        # Annotation tools
        self.annotation_group = self.create_annotation_tools_group()
        layout.addWidget(self.annotation_group)
        
        # Model management
        self.model_group = self.create_model_management_group()
        layout.addWidget(self.model_group)
        
        # Training data accumulation
        self.training_group = self.create_training_accumulation_group()
        layout.addWidget(self.training_group)

        # Finetuning controls
        self.finetune_group = self.create_finetuning_group()
        layout.addWidget(self.finetune_group)

        # Add stretch to push everything to top
        layout.addStretch()
        
        # Initially disable controls
        self.set_backend_connected(False)
        self.set_image_loaded(False)
    
    def create_connection_status_group(self) -> QGroupBox:
        """Create connection status group"""
        group = QGroupBox("🔗 Connection Status")
        layout = QVBoxLayout(group)
        
        self.connection_label = QLabel("🔴 Disconnected from Backend")
        self.connection_label.setStyleSheet("color: red; font-weight: bold; padding: 5px;")
        layout.addWidget(self.connection_label)
        
        self.reconnect_btn = QPushButton("🔄 Reconnect")
        self.reconnect_btn.clicked.connect(self.request_reconnect)
        layout.addWidget(self.reconnect_btn)
        
        return group
    
    def create_image_info_group(self) -> QGroupBox:
        """Create image information group"""
        group = QGroupBox("📊 Image Information")
        layout = QVBoxLayout(group)
        
        # Current file
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.current_file_label = QLabel("No file loaded")
        self.current_file_label.setStyleSheet("font-style: italic; color: #666;")
        file_layout.addWidget(self.current_file_label)
        layout.addLayout(file_layout)
        
        # Image dimensions
        dims_layout = QHBoxLayout()
        dims_layout.addWidget(QLabel("Size:"))
        self.image_dims_label = QLabel("-")
        dims_layout.addWidget(self.image_dims_label)
        layout.addLayout(dims_layout)
        
        # Patch info
        patch_layout = QHBoxLayout()
        patch_layout.addWidget(QLabel("Patches:"))
        self.patch_count_label = QLabel("0")
        patch_layout.addWidget(self.patch_count_label)
        layout.addLayout(patch_layout)
        
        return group
    
    def create_ai_processing_group(self) -> QGroupBox:
        """Create AI processing controls"""
        group = QGroupBox("🤖 AI Processing")
        layout = QVBoxLayout(group)
        
        # Predict button
        self.predict_btn = QPushButton("🔍 Run Prediction")
        self.predict_btn.setMinimumHeight(40)
        self.predict_btn.clicked.connect(self.predict_requested.emit)
        layout.addWidget(self.predict_btn)
        
        # Prediction progress
        self.prediction_progress = QProgressBar()
        self.prediction_progress.setVisible(False)
        layout.addWidget(self.prediction_progress)
        
        # Results info
        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("Detections:"))
        self.detections_count_label = QLabel("0")
        self.detections_count_label.setStyleSheet("font-weight: bold; color: #27ae60;")
        results_layout.addWidget(self.detections_count_label)
        results_layout.addStretch()
        layout.addLayout(results_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Min Confidence:"))
        self.confidence_spinbox = QSpinBox()
        self.confidence_spinbox.setRange(0, 100)
        self.confidence_spinbox.setValue(50)
        self.confidence_spinbox.setSuffix("%")
        conf_layout.addWidget(self.confidence_spinbox)
        layout.addLayout(conf_layout)
        
        return group
    
    def create_annotation_tools_group(self) -> QGroupBox:
        """Create annotation tools group"""
        group = QGroupBox("✏️ Annotation Tools")
        layout = QVBoxLayout(group)
        
        # Tool buttons
        button_layout1 = QHBoxLayout()
        
        self.add_bbox_btn = QPushButton("➕ Add Box")
        self.add_bbox_btn.setToolTip("Click to start drawing new bounding boxes")
        button_layout1.addWidget(self.add_bbox_btn)
        
        self.delete_bbox_btn = QPushButton("🗑️ Delete")
        self.delete_bbox_btn.setToolTip("Delete selected bounding box")
        button_layout1.addWidget(self.delete_bbox_btn)
        
        layout.addLayout(button_layout1)
        
        button_layout2 = QHBoxLayout()
        
        self.clear_all_btn = QPushButton("🧹 Clear All")
        self.clear_all_btn.setToolTip("Clear all bounding boxes")
        button_layout2.addWidget(self.clear_all_btn)
        
        self.undo_btn = QPushButton("↶ Undo")
        self.undo_btn.setToolTip("Undo last action")
        button_layout2.addWidget(self.undo_btn)
        
        layout.addLayout(button_layout2)
        
        # Annotation stats
        stats_frame = QFrame()
        stats_frame.setFrameStyle(QFrame.StyledPanel)
        stats_layout = QVBoxLayout(stats_frame)
        
        # Box count
        count_layout = QHBoxLayout()
        count_layout.addWidget(QLabel("Boxes:"))
        self.bbox_count_label = QLabel("0")
        self.bbox_count_label.setStyleSheet("font-weight: bold; color: #3498db;")
        count_layout.addWidget(self.bbox_count_label)
        count_layout.addStretch()
        stats_layout.addLayout(count_layout)
        
        # Average confidence
        avg_conf_layout = QHBoxLayout()
        avg_conf_layout.addWidget(QLabel("Avg Confidence:"))
        self.avg_confidence_label = QLabel("0%")
        avg_conf_layout.addWidget(self.avg_confidence_label)
        avg_conf_layout.addStretch()
        stats_layout.addLayout(avg_conf_layout)
        
        layout.addWidget(stats_frame)
        
        return group
    
    def create_model_management_group(self) -> QGroupBox:
        """Create model management controls"""
        group = QGroupBox("🧠 Model Management")
        layout = QVBoxLayout(group)

        # Current model display
        current_layout = QHBoxLayout()
        current_layout.addWidget(QLabel("Active Model:"))
        self.active_model_label = QLabel("default")
        self.active_model_label.setStyleSheet("font-weight: bold; color: #8e44ad;")
        current_layout.addWidget(self.active_model_label)
        layout.addLayout(current_layout)

        # Model selector
        selector_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["default", "pretrained"])
        selector_layout.addWidget(self.model_selector)

        self.switch_model_btn = QPushButton("🔄 Switch")
        self.switch_model_btn.clicked.connect(self.switch_model)
        selector_layout.addWidget(self.switch_model_btn)

        layout.addLayout(selector_layout)

        # Refresh models button
        self.refresh_models_btn = QPushButton("🔄 Refresh Models")
        self.refresh_models_btn.clicked.connect(self.refresh_models)
        layout.addWidget(self.refresh_models_btn)

        return group

    def create_training_accumulation_group(self) -> QGroupBox:
        """Create training data accumulation controls"""
        group = QGroupBox("📝 Training Data")
        layout = QVBoxLayout(group)

        # Status display
        status_layout = QHBoxLayout()
        status_layout.addWidget(QLabel("Accumulated:"))
        self.training_count_label = QLabel("0 patches")
        self.training_count_label.setStyleSheet("font-weight: bold; color: #e67e22;")
        status_layout.addWidget(self.training_count_label)
        status_layout.addStretch()
        layout.addLayout(status_layout)

        # Save button
        self.save_training_btn = QPushButton("💾 Save for Training")
        self.save_training_btn.setMinimumHeight(35)
        self.save_training_btn.setToolTip("Save current patch annotations for model finetuning")
        layout.addWidget(self.save_training_btn)

        # Info label
        info_label = QLabel("Save annotated patches to accumulate training data")
        info_label.setStyleSheet("font-size: 9px; color: #666; font-style: italic;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        return group
    
    def create_finetuning_group(self) -> QGroupBox:
        """Create finetuning controls"""
        group = QGroupBox("🎓 Model Finetuning")
        layout = QVBoxLayout(group)
        
        # Model name input
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("New Model Name:"))
        self.model_name_input = QLineEdit()
        self.model_name_input.setPlaceholderText("my_custom_model")
        name_layout.addWidget(self.model_name_input)
        layout.addLayout(name_layout)
        
        # Finetuning parameters
        params_frame = QFrame()
        params_frame.setFrameStyle(QFrame.StyledPanel)
        params_layout = QVBoxLayout(params_frame)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 100)
        self.epochs_spinbox.setValue(10)
        epochs_layout.addWidget(self.epochs_spinbox)
        epochs_layout.addStretch()
        params_layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_input = QLineEdit("0.001")
        self.lr_input.setMaximumWidth(80)
        lr_layout.addWidget(self.lr_input)
        lr_layout.addStretch()
        params_layout.addLayout(lr_layout)
        
        layout.addWidget(params_frame)
        
        # Finetune button
        self.finetune_btn = QPushButton("🚀 Start Finetuning")
        self.finetune_btn.setMinimumHeight(40)
        self.finetune_btn.clicked.connect(self.start_finetuning)
        layout.addWidget(self.finetune_btn)
        
        # Finetuning progress
        self.finetune_progress = QProgressBar()
        self.finetune_progress.setVisible(False)
        layout.addWidget(self.finetune_progress)
        
        # Status text
        self.finetune_status_label = QLabel("")
        self.finetune_status_label.setStyleSheet("font-style: italic; color: #666;")
        self.finetune_status_label.setWordWrap(True)
        layout.addWidget(self.finetune_status_label)
        
        return group
    
    def setup_styles(self):
        """Apply styles to the control panel"""
        self.setStyleSheet("""
            QGroupBox {
                font-size: 11px;
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin: 8px 0px;
                padding-top: 15px;
                background-color: #fafafa;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px 0 8px;
                color: #2c3e50;
            }
            
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 10px;
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
            
            QLineEdit {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            
            QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            
            QSpinBox {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 4px;
                background-color: white;
            }
            
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                text-align: center;
                background-color: #ecf0f1;
            }
            
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 3px;
            }
        """)
    
    def set_backend_connected(self, connected: bool):
        """Update UI based on backend connection status"""
        self.backend_connected = connected

        if connected:
            self.connection_label.setText("🟢 Connected to Backend")
            self.connection_label.setStyleSheet("color: green; font-weight: bold; padding: 5px;")
            self.reconnect_btn.setEnabled(False)
            self.predict_btn.setText("🔍 Run Prediction")
            self.finetune_btn.setText("🚀 Start Finetuning")
        else:
            self.connection_label.setText("🟡 Offline Mode")
            self.connection_label.setStyleSheet("color: orange; font-weight: bold; padding: 5px;")
            self.reconnect_btn.setEnabled(True)
            self.predict_btn.setText("🔍 Run Prediction (Offline)")
            self.finetune_btn.setText("🚀 Start Finetuning (Offline)")

        # Update other controls
        self._update_controls_state()
    
    def set_image_loaded(self, loaded: bool):
        """Update UI based on whether image is loaded"""
        self.image_loaded = loaded
        self._update_controls_state()
    
    def _update_controls_state(self):
        """Update the enabled state of controls"""
        # AI processing controls
        self.predict_btn.setEnabled(self.backend_connected and self.image_loaded)
        
        # Annotation tools
        for btn in [self.add_bbox_btn, self.delete_bbox_btn, 
                   self.clear_all_btn, self.undo_btn]:
            btn.setEnabled(self.image_loaded)
        
        # Model management
        self.switch_model_btn.setEnabled(self.backend_connected)
        self.refresh_models_btn.setEnabled(self.backend_connected)
        
        # Training data saving
        self.save_training_btn.setEnabled(
            self.backend_connected and self.image_loaded and
            self.annotation_count > 0
        )

        # Finetuning (enabled if backend connected - doesn't need current image)
        self.finetune_btn.setEnabled(self.backend_connected)
    
    def update_image_info(self, file_name: str, dimensions: tuple, patch_count: int):
        """Update image information display"""
        self.current_file_label.setText(file_name)
        self.image_dims_label.setText(f"{dimensions[0]} x {dimensions[1]}")
        self.patch_count_label.setText(str(patch_count))
    
    def update_annotation_count(self, count: int):
        """Update annotation count and statistics"""
        self.annotation_count = count
        self.bbox_count_label.setText(str(count))
        self.detections_count_label.setText(str(count))
        self._update_controls_state()
    
    def update_active_model(self, model_name: str):
        """Update active model display"""
        self.current_model = model_name
        self.active_model_label.setText(model_name)
        
        # Update selector if needed
        current_index = self.model_selector.findText(model_name)
        if current_index >= 0:
            self.model_selector.setCurrentIndex(current_index)
    
    def show_prediction_progress(self, show: bool = True):
        """Show/hide prediction progress bar"""
        self.prediction_progress.setVisible(show)
        if show:
            self.prediction_progress.setRange(0, 0)  # Indeterminate progress
    
    def show_finetuning_progress(self, show: bool = True):
        """Show/hide finetuning progress bar"""
        self.finetune_progress.setVisible(show)
        if show:
            self.finetune_progress.setRange(0, 0)  # Indeterminate progress
    
    def set_finetuning_status(self, status: str):
        """Set finetuning status text"""
        self.finetune_status_label.setText(status)

    def update_training_count(self, count: int):
        """Update training data count display"""
        self.training_count_label.setText(f"{count} patch{'es' if count != 1 else ''}")

    @pyqtSlot()
    def request_reconnect(self):
        """Request reconnection to backend"""
        # This would trigger a reconnection attempt
        pass
    
    @pyqtSlot()
    def switch_model(self):
        """Switch to selected model"""
        selected_model = self.model_selector.currentText()
        if selected_model and selected_model != self.current_model:
            self.model_switch_requested.emit(selected_model)
    
    @pyqtSlot()
    def refresh_models(self):
        """Refresh the list of available models"""
        # This would query the backend for available models
        pass
    
    @pyqtSlot()
    def start_finetuning(self):
        """Start the finetuning process"""
        model_name = self.model_name_input.text().strip()
        if not model_name:
            model_name = "custom_model"
        
        self.finetune_requested.emit(model_name)