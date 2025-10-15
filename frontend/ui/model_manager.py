"""
Model Manager Widget
Handles model selection, switching, and management
"""

from typing import List, Dict, Optional
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, 
    QLabel, QComboBox, QListWidget, QListWidgetItem, QFrame,
    QTextEdit, QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QFont


class ModelManagerWidget(QWidget):
    """Widget for managing available models"""
    
    # Signals
    model_selected = pyqtSignal(str)        # Model selected for switching
    model_deleted = pyqtSignal(str)         # Model deletion requested
    model_exported = pyqtSignal(str)        # Model export requested
    
    def __init__(self):
        super().__init__()
        self.models_list = []
        self.active_model = ""
        
        self.setup_ui()
        self.setup_styles()
    
    def setup_ui(self):
        """Setup the model manager UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("🧠 Model Management Center")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Available models group
        models_group = QGroupBox("📋 Available Models")
        models_layout = QVBoxLayout(models_group)
        
        # Models list
        self.models_list_widget = QListWidget()
        self.models_list_widget.itemClicked.connect(self.on_model_selected)
        models_layout.addWidget(self.models_list_widget)
        
        # Model actions
        actions_layout = QHBoxLayout()
        
        self.load_model_btn = QPushButton("📂 Load")
        self.load_model_btn.clicked.connect(self.load_selected_model)
        actions_layout.addWidget(self.load_model_btn)
        
        self.delete_model_btn = QPushButton("🗑️ Delete")
        self.delete_model_btn.clicked.connect(self.delete_selected_model)
        actions_layout.addWidget(self.delete_model_btn)
        
        self.export_model_btn = QPushButton("💾 Export")
        self.export_model_btn.clicked.connect(self.export_selected_model)
        actions_layout.addWidget(self.export_model_btn)
        
        models_layout.addLayout(actions_layout)
        
        layout.addWidget(models_group)
        
        # Active model info
        active_group = QGroupBox("🔧 Active Model Info")
        active_layout = QVBoxLayout(active_group)
        
        # Model name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.active_name_label = QLabel("None")
        self.active_name_label.setStyleSheet("font-weight: bold; color: #2c3e50;")
        name_layout.addWidget(self.active_name_label)
        name_layout.addStretch()
        active_layout.addLayout(name_layout)
        
        # Model type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Type:"))
        self.active_type_label = QLabel("DeepForest")
        type_layout.addWidget(self.active_type_label)
        type_layout.addStretch()
        active_layout.addLayout(type_layout)
        
        # Performance metrics (if available)
        metrics_frame = QFrame()
        metrics_frame.setFrameStyle(QFrame.StyledPanel)
        metrics_layout = QVBoxLayout(metrics_frame)
        
        self.metrics_label = QLabel("Performance metrics will appear here after model evaluation")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        metrics_layout.addWidget(self.metrics_label)
        
        active_layout.addWidget(metrics_frame)
        
        layout.addWidget(active_group)
        
        # Model training history
        history_group = QGroupBox("📊 Training History")
        history_layout = QVBoxLayout(history_group)
        
        self.history_text = QTextEdit()
        self.history_text.setMaximumHeight(100)
        self.history_text.setReadOnly(True)
        self.history_text.setPlaceholderText("Training history will appear here...")
        history_layout.addWidget(self.history_text)
        
        layout.addWidget(history_group)
        
        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh Model List")
        self.refresh_btn.clicked.connect(self.refresh_models)
        layout.addWidget(self.refresh_btn)
        
    def setup_styles(self):
        """Apply styles to the widget"""
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #bdc3c7;
                border-radius: 8px;
                margin: 5px 0px;
                padding-top: 15px;
                background-color: #fafafa;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #2c3e50;
            }
            
            QListWidget {
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                background-color: white;
                alternate-background-color: #f8f9fa;
            }
            
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #ecf0f1;
            }
            
            QListWidget::item:selected {
                background-color: #3498db;
                color: white;
            }
            
            QListWidget::item:hover {
                background-color: #ebf3fd;
            }
            
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #2980b9;
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
                font-size: 9px;
            }
        """)
    
    def set_models_list(self, models: List[Dict]):
        """Set the list of available models"""
        self.models_list = models
        self.update_models_display()
    
    def update_models_display(self):
        """Update the models list widget"""
        self.models_list_widget.clear()
        
        for model in self.models_list:
            item_text = f"📦 {model.get('name', 'Unknown')}"
            if model.get('active', False):
                item_text += " ✅ (Active)"
            
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, model)
            
            # Color code based on model type or status
            if model.get('active', False):
                item.setBackground(Qt.lightGreen)
            elif model.get('type') == 'finetuned':
                item.setBackground(Qt.lightBlue)
            
            self.models_list_widget.addItem(item)
    
    def set_active_model(self, model_name: str, model_info: Optional[Dict] = None):
        """Set the active model information"""
        self.active_model = model_name
        self.active_name_label.setText(model_name)
        
        if model_info:
            self.active_type_label.setText(model_info.get('type', 'DeepForest'))
            
            # Update metrics if available
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                metrics_text = f"Accuracy: {metrics.get('accuracy', 'N/A')}\n"
                metrics_text += f"Precision: {metrics.get('precision', 'N/A')}\n"
                metrics_text += f"Recall: {metrics.get('recall', 'N/A')}"
                self.metrics_label.setText(metrics_text)
            else:
                self.metrics_label.setText("Performance metrics not available")
    
    def add_training_log(self, log_entry: str):
        """Add an entry to the training history"""
        current_text = self.history_text.toPlainText()
        new_text = f"{current_text}\n{log_entry}" if current_text else log_entry
        
        # Keep only last 20 lines
        lines = new_text.split('\n')
        if len(lines) > 20:
            lines = lines[-20:]
            new_text = '\n'.join(lines)
        
        self.history_text.setPlainText(new_text)
        
        # Scroll to bottom
        scrollbar = self.history_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    @pyqtSlot(QListWidgetItem)
    def on_model_selected(self, item: QListWidgetItem):
        """Handle model selection"""
        model_data = item.data(Qt.UserRole)
        if model_data:
            # Enable/disable buttons based on selection
            self.load_model_btn.setEnabled(True)
            self.delete_model_btn.setEnabled(model_data.get('name') != self.active_model)
            self.export_model_btn.setEnabled(True)
    
    @pyqtSlot()
    def load_selected_model(self):
        """Load the selected model"""
        current_item = self.models_list_widget.currentItem()
        if current_item:
            model_data = current_item.data(Qt.UserRole)
            model_name = model_data.get('name', '')
            if model_name:
                self.model_selected.emit(model_name)
    
    @pyqtSlot()
    def delete_selected_model(self):
        """Delete the selected model"""
        current_item = self.models_list_widget.currentItem()
        if current_item:
            model_data = current_item.data(Qt.UserRole)
            model_name = model_data.get('name', '')
            
            if model_name == self.active_model:
                QMessageBox.warning(self, "Warning", "Cannot delete the active model!")
                return
            
            # Confirm deletion
            reply = QMessageBox.question(
                self, 
                "Confirm Deletion", 
                f"Are you sure you want to delete model '{model_name}'?\n\nThis action cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.model_deleted.emit(model_name)
    
    @pyqtSlot()
    def export_selected_model(self):
        """Export the selected model"""
        current_item = self.models_list_widget.currentItem()
        if current_item:
            model_data = current_item.data(Qt.UserRole)
            model_name = model_data.get('name', '')
            if model_name:
                self.model_exported.emit(model_name)
    
    @pyqtSlot()
    def refresh_models(self):
        """Refresh the models list"""
        # This would typically make an API call to get updated models
        pass