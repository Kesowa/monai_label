#!/usr/bin/env python3
"""
PyQt5 Professional TIFF Annotation Application
Interactive AI Workflow - Tree Detection and Annotation

COORDINATE SYSTEM:
- All bboxes stored in IMAGE coordinates (patch-local pixel coordinates)
- Conversion to WIDGET coordinates only for display
- Global coordinates used when communicating with backend
"""

import sys
import os
import signal
from typing import List, Dict, Optional
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QFileDialog,
    QMessageBox, QGroupBox, QFrame,
    QSplitter, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap

# Custom widgets
from ui.tiff_viewer import TiffViewer, BoundingBoxEditor
from ui.control_panel import ControlPanel
from ui.progress_dialog import ProgressDialog
from utils.api_client import FastAPIClient
from utils.tiff_processor import TiffProcessor


class UploadThread(QThread):
    """Background thread for uploading files without blocking UI"""
    upload_complete = pyqtSignal(object)  # Emits response or None
    upload_progress = pyqtSignal(str)  # Emits status messages
    
    def __init__(self, api_client, file_path):
        super().__init__()
        self.api_client = api_client
        self.file_path = file_path
        
    def run(self):
        """Run the upload in background thread"""
        try:
            self.upload_progress.emit("Uploading to backend...")
            response = self.api_client.upload_tiff(self.file_path)
            self.upload_complete.emit(response)
        except Exception as e:
            print(f"Upload thread error: {e}")
            self.upload_complete.emit(None)


class PredictThread(QThread):
    """Background thread for running predictions without blocking UI"""
    predict_complete = pyqtSignal(object)  # Emits response or None
    predict_progress = pyqtSignal(str)  # Emits status messages
    
    def __init__(self, api_client, tiff_processor, patch_index):
        super().__init__()
        self.api_client = api_client
        self.tiff_processor = tiff_processor
        self.patch_index = patch_index
        
    def run(self):
        """Run the prediction in background thread"""
        import tempfile
        temp_file = None
        
        try:
            self.predict_progress.emit("Preparing patch for prediction...")
            
            # Save current patch to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Save the patch
            if not self.tiff_processor.save_patch_to_file(self.patch_index, temp_path):
                raise Exception("Failed to save patch")
            
            self.predict_progress.emit("Uploading patch and running AI prediction...")
            
            # Send patch for prediction
            response = self.api_client.predict_patch(temp_path)
            
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            self.predict_complete.emit(response)
            
        except Exception as e:
            print(f"Predict thread error: {e}")
            if temp_file:
                try:
                    os.remove(temp_file.name)
                except:
                    pass
            self.predict_complete.emit(None)


class MainWindow(QMainWindow):
    """Main application window with professional UI layout"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive AI Workflow - Professional Tree Detection")
        self.setMinimumSize(1400, 900)
        
        # Initialize components
        self.api_client = FastAPIClient()
        self.tiff_processor = TiffProcessor(patch_size=(700, 700))
        
        # State variables
        self.current_image_id = None
        self.current_image_path = None
        self.current_annotations = []  # Always in IMAGE coordinates
        self.current_patch_index = 0
        self.total_patches = 0
        
        # Loading state management
        self._is_loading = False
        self._pending_patch_load = None
        
        # Debounce timer for patch changes
        self.patch_change_timer = QTimer()
        self.patch_change_timer.setSingleShot(True)
        self.patch_change_timer.timeout.connect(self._do_patch_load)
        
        # Setup UI
        self.setup_ui()
        self.setup_styles()
        self.connect_signals()
        
        # Check backend connection
        QTimer.singleShot(100, self.check_backend_connection)
        
    def setup_ui(self):
        """Setup the main UI layout"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create main splitter
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        # Left panel - Controls
        self.control_panel = ControlPanel()
        main_splitter.addWidget(self.control_panel)
        
        # Right panel - Image viewer
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Toolbar
        toolbar = self.create_toolbar()
        right_layout.addWidget(toolbar)
        
        # TIFF Viewer with bounding box editor
        self.tiff_viewer = TiffViewer()
        self.bbox_editor = BoundingBoxEditor(self.tiff_viewer)
        
        # Add viewer to scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.tiff_viewer)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumSize(800, 600)
        right_layout.addWidget(scroll_area)
        
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([400, 1000])
        
        # Status bar
        self.statusBar().showMessage("Ready - Load a TIFF file to begin")
        
    def create_toolbar(self) -> QWidget:
        """Create the main toolbar"""
        toolbar = QFrame()
        toolbar.setFrameStyle(QFrame.StyledPanel)
        toolbar_layout = QHBoxLayout(toolbar)
        
        # File operations
        self.load_btn = QPushButton("📁 Load TIFF")
        self.load_btn.setMinimumHeight(40)
        toolbar_layout.addWidget(self.load_btn)
        
        toolbar_layout.addWidget(QLabel("|"))
        
        # Patch navigation
        toolbar_layout.addWidget(QLabel("Patch:"))
        
        self.patch_spinbox = QSpinBox()
        self.patch_spinbox.setEnabled(False)
        self.patch_spinbox.setMinimumWidth(100)
        toolbar_layout.addWidget(self.patch_spinbox)
        
        self.patch_info_label = QLabel("0 / 0")
        toolbar_layout.addWidget(self.patch_info_label)
        
        toolbar_layout.addWidget(QLabel("|"))
        
        toolbar_layout.addStretch()
        
        # Connection status
        self.connection_status = QLabel("🔴 Disconnected")
        toolbar_layout.addWidget(self.connection_status)
        
        return toolbar
        
    def setup_styles(self):
        """Apply professional styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            
            QPushButton {
                background-color: #4a90e2;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            
            QPushButton:hover {
                background-color: #357abd;
            }
            
            QPushButton:pressed {
                background-color: #2868a3;
            }
            
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 8px;
                margin: 5px 0px;
                padding-top: 15px;
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            
            QFrame[frameShape="4"] {
                border: 1px solid #cccccc;
                background-color: white;
            }
            
            QScrollArea {
                border: 1px solid #cccccc;
                background-color: white;
            }
        """)
        
    def connect_signals(self):
        """Connect all signal handlers"""
        # File operations
        self.load_btn.clicked.connect(self.load_tiff_file)
        
        # Patch navigation
        self.patch_spinbox.valueChanged.connect(self.on_patch_changed)
        
        # Control panel signals
        self.control_panel.predict_requested.connect(self.predict_current_patch)
        self.control_panel.finetune_requested.connect(self.start_finetuning)
        self.control_panel.model_switch_requested.connect(self.switch_model)
        
        # Bbox editor controls
        self.control_panel.add_bbox_btn.clicked.connect(self.start_bbox_creation)
        self.control_panel.delete_bbox_btn.clicked.connect(self.delete_selected_bbox)
        self.control_panel.clear_all_btn.clicked.connect(self.clear_all_bboxes)
        self.control_panel.undo_btn.clicked.connect(self.undo_last_action)
        
        # Bounding box editor signals
        self.bbox_editor.bbox_created.connect(self.on_bbox_created)
        self.bbox_editor.bbox_deleted.connect(self.on_bbox_deleted)
        self.bbox_editor.bboxes_changed.connect(self.on_bboxes_changed)

        # Model management signals
        self.control_panel.refresh_models_btn.clicked.connect(self.refresh_model_list)
        self.control_panel.save_training_btn.clicked.connect(self.save_current_patch_annotations)
        
    def check_backend_connection(self):
        """Check if backend is running"""
        try:
            response = self.api_client.health_check()
            if response and response.get('status') == 'healthy':
                self.connection_status.setText("🟢 Connected")
                self.connection_status.setStyleSheet("color: green; font-weight: bold;")
                self.control_panel.set_backend_connected(True)
                self.statusBar().showMessage("Backend connected - Full functionality available")
            else:
                self.show_backend_warning()
        except Exception:
            self.show_backend_warning()

    def show_backend_warning(self):
        """Show backend connection warning"""
        self.connection_status.setText("🟡 Offline Mode")
        self.connection_status.setStyleSheet("color: orange; font-weight: bold;")
        self.control_panel.set_backend_connected(False)
        self.statusBar().showMessage("Offline mode - Some features limited")
    
    def load_tiff_file(self):
        """Load a TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select TIFF File", 
            "", 
            "TIFF Files (*.tiff *.tif);;All Files (*)"
        )
        
        if file_path:
            self.process_tiff_file(file_path)
    
    def process_tiff_file(self, file_path: str):
        """Process the selected TIFF file"""
        progress = ProgressDialog("Processing TIFF File", self)
        progress.show()
        QApplication.processEvents()

        try:
            # Process TIFF locally first (fast, non-blocking)
            progress.set_progress(20, "Analyzing file and checking memory...")
            QApplication.processEvents()
            
            success = self.tiff_processor.load_tiff(file_path)

            if not success:
                raise Exception("Failed to load TIFF - memory constraints")

            patch_info = self.tiff_processor.get_patch_info()
            self.current_image_path = file_path

            # Setup patch navigation
            self.total_patches = patch_info['total_patches']
            self.setup_patch_navigation()

            progress.set_progress(60, "Loading first patch...")
            QApplication.processEvents()

            # Load first patch immediately (UI is responsive)
            self.load_patch(0)

            progress.set_progress(100, "Complete!")
            progress.close()

            # No need to upload entire file - predictions work patch-by-patch
            backend_status = "Ready for predictions" if self.control_panel.backend_connected else "Offline mode"
            self.statusBar().showMessage(
                f"Loaded: {Path(file_path).name} ({self.total_patches} patches) - {backend_status}"
            )

        except MemoryError:
            QMessageBox.critical(
                self, "Memory Error",
                "The TIFF file is too large for available memory.\n\n"
                "Suggestions:\n"
                "• Close other applications\n"
                "• Try a smaller file\n"
                "• Add more RAM"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process TIFF:\n{str(e)}")

        finally:
            if progress.isVisible():
                progress.close()
    
    def setup_patch_navigation(self):
        """Setup patch navigation controls"""
        if self.total_patches > 0:
            self.patch_spinbox.setRange(0, self.total_patches - 1)
            self.patch_spinbox.setValue(0)
            self.patch_spinbox.setEnabled(True)
            self.update_patch_info()
    
    def load_patch(self, patch_index: int):
        """Load a specific patch"""
        if not self.current_image_path:
            return
            
        try:
            # Clear previous display
            self.tiff_viewer.clear()
            
            # Force GC before loading new patch
            import gc
            gc.collect()
            
            # Get patch from processor
            patch_data = self.tiff_processor.get_patch(patch_index)
            
            if not patch_data:
                raise Exception("Failed to load patch data")
            
            # Display in viewer
            self.tiff_viewer.set_image(patch_data['image'])

            # Update current patch info
            self.current_patch_index = patch_index
            self.update_patch_info()

            # Clear existing annotations
            self.bbox_editor.clear_bboxes()
            self.current_annotations.clear()

            # Reset creation mode
            if self.bbox_editor.creating_bbox:
                self.bbox_editor.stop_bbox_creation()
                self.control_panel.add_bbox_btn.setText("➕ Add Box")
                self.control_panel.add_bbox_btn.setStyleSheet("")

            # Enable controls
            self.control_panel.set_image_loaded(True)
            self.control_panel.update_annotation_count(0)
            
        except MemoryError:
            QMessageBox.critical(
                self, "Memory Error",
                f"Out of memory loading patch {patch_index}.\n\n"
                "Try closing other applications."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load patch:\n{str(e)}")
    
    def update_patch_info(self):
            """Update patch information display"""
            self.patch_info_label.setText(f"{self.current_patch_index + 1} / {self.total_patches}")
            self.patch_spinbox.blockSignals(True)
            self.patch_spinbox.setValue(self.current_patch_index)
            self.patch_spinbox.blockSignals(False)
    
    @pyqtSlot(int)
    def on_patch_changed(self, value: int):
        """Handle patch navigation with debouncing"""
        if self._is_loading:
            self._pending_patch_load = value
            return
        
        # Debounce rapid changes
        self._pending_patch_load = value
        self.patch_change_timer.start(150)  # 150ms debounce
    
    def _do_patch_load(self):
        """Actually load the patch (after debounce)"""
        if self._pending_patch_load is not None:
            value = self._pending_patch_load
            self._pending_patch_load = None
            
            if value != self.current_patch_index:
                self._is_loading = True
                try:
                    self.load_patch(value)
                finally:
                    self._is_loading = False
    
    @pyqtSlot()
    def predict_current_patch(self):
        """Run prediction on current patch"""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No TIFF file loaded")
            return

        # Check if backend is connected
        if not self.control_panel.backend_connected:
            QMessageBox.information(
                self, "Offline Mode",
                "AI predictions require backend connection.\n"
                "You can still create bounding boxes manually."
            )
            return

        # Start background prediction on current patch (no full upload needed!)
        self.statusBar().showMessage("Preparing current patch for AI prediction...")
        self.start_background_prediction()
    
    def start_background_prediction(self):
        """Start prediction in background thread on current patch"""
        self.predict_thread = PredictThread(
            self.api_client,
            self.tiff_processor,
            self.current_patch_index
        )
        self.predict_thread.predict_progress.connect(self.on_predict_progress)
        self.predict_thread.predict_complete.connect(self.on_predict_complete)
        self.predict_thread.start()
    
    def on_predict_progress(self, message: str):
        """Handle prediction progress updates"""
        self.statusBar().showMessage(message)
    
    def on_predict_complete(self, response):
        """Handle prediction completion"""
        if response and response.get('success'):
            predictions = response.get('predictions', [])

            # Predictions are already in patch-local coordinates (no filtering needed)
            # Just set them directly
            self.bbox_editor.set_bboxes(predictions)
            self.current_annotations = predictions.copy()

            self.statusBar().showMessage(f"Prediction complete: {len(predictions)} detections on current patch")
        else:
            self.statusBar().showMessage("Prediction failed")
            QMessageBox.warning(self, "Prediction Failed", "Failed to get predictions from backend")
    
    @pyqtSlot(dict)
    def on_bbox_created(self, bbox: Dict):
        """Handle new bounding box creation (receives IMAGE coords)"""
        self.current_annotations.append(bbox.copy())
    
    @pyqtSlot(int)
    def on_bbox_deleted(self, index: int):
        """Handle bounding box deletion"""
        if 0 <= index < len(self.current_annotations):
            del self.current_annotations[index]
    
    @pyqtSlot()
    def on_bboxes_changed(self):
        """Handle any change to bboxes"""
        # Sync with bbox editor
        self.current_annotations = self.bbox_editor.get_bboxes()
        self.control_panel.update_annotation_count(len(self.current_annotations))
    
    @pyqtSlot(str)
    def start_finetuning(self, model_name: str):
        """Start model finetuning using accumulated patch annotations"""
        # Check if backend is connected
        if not self.control_panel.backend_connected:
            QMessageBox.information(
                self, "Offline Mode",
                "Model finetuning requires backend connection."
            )
            return

        # Check if we have accumulated annotations
        try:
            response = self.api_client.get_patch_annotations_count()
            if not response or not response.get('success'):
                QMessageBox.warning(
                    self, "No Annotations",
                    "No accumulated patch annotations found.\n\n"
                    "Save annotations from patches using the 'Save for Training' button first."
                )
                return

            annotation_count = response.get('count', 0)
            if annotation_count == 0:
                QMessageBox.warning(
                    self, "No Annotations",
                    "No accumulated patch annotations found.\n\n"
                    "Save annotations from patches using the 'Save for Training' button first."
                )
                return

            # Confirm finetuning
            reply = QMessageBox.question(
                self,
                "Start Finetuning",
                f"Start finetuning new model '{model_name}'?\n\n"
                f"• Using {annotation_count} accumulated patch annotation(s)\n"
                f"• Epochs: {self.control_panel.epochs_spinbox.value()}\n"
                f"• Learning Rate: {self.control_panel.lr_input.text()}\n\n"
                "This will create a new model version that you can switch to.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply != QMessageBox.Yes:
                return

            # Start finetuning
            epochs = self.control_panel.epochs_spinbox.value()
            lr = float(self.control_panel.lr_input.text())

            self.control_panel.show_finetuning_progress(True)
            self.control_panel.set_finetuning_status("Starting finetuning...")

            response = self.api_client.start_patch_finetuning(model_name, epochs, lr)

            if response and response.get('success'):
                task_id = response.get('task_id')
                self.statusBar().showMessage(f"Finetuning started: {response.get('message')}")

                # Poll for status updates
                self.start_finetuning_status_poll(task_id, model_name)
            else:
                self.control_panel.show_finetuning_progress(False)
                self.control_panel.set_finetuning_status("Failed to start finetuning")
                QMessageBox.critical(self, "Error", "Failed to start finetuning")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start finetuning:\n{str(e)}")
            self.control_panel.show_finetuning_progress(False)
    
    def convert_annotations_to_global(self) -> List[Dict]:
        """Convert patch-local IMAGE coordinates to global image coordinates"""
        patch_bounds = self.tiff_processor.get_patch_bounds(self.current_patch_index)
        if not patch_bounds:
            return []

        global_annotations = []
        for annotation in self.current_annotations:
            # Convert from patch-local IMAGE coords to global coords
            global_annotation = {
                'xmin': annotation['xmin'] + patch_bounds['xmin'],
                'ymin': annotation['ymin'] + patch_bounds['ymin'],
                'xmax': annotation['xmax'] + patch_bounds['xmin'],
                'ymax': annotation['ymax'] + patch_bounds['ymin'],
                'confidence': annotation.get('confidence'),
                'label': annotation.get('label', 'Tree')
            }
            global_annotations.append(global_annotation)

        return global_annotations
    
    def start_finetuning_status_poll(self, task_id: str, model_name: str):
        """Poll for finetuning status updates"""
        self.finetuning_task_id = task_id
        self.finetuning_model_name = model_name

        # Create timer for polling
        if not hasattr(self, 'finetune_poll_timer'):
            self.finetune_poll_timer = QTimer()
            self.finetune_poll_timer.timeout.connect(self.check_finetuning_status)

        self.finetune_poll_timer.start(2000)  # Poll every 2 seconds

    def check_finetuning_status(self):
        """Check finetuning task status"""
        if not hasattr(self, 'finetuning_task_id'):
            return

        try:
            response = self.api_client.get_finetuning_status(self.finetuning_task_id)

            if response and response.get('success'):
                task = response.get('task', {})
                status = task.get('status')
                message = task.get('message', '')
                progress = task.get('progress', 0)

                self.control_panel.set_finetuning_status(message)

                if status == 'completed':
                    self.finetune_poll_timer.stop()
                    self.control_panel.show_finetuning_progress(False)

                    QMessageBox.information(
                        self, "Finetuning Complete",
                        f"Model '{self.finetuning_model_name}' has been created!\n\n"
                        f"{message}\n\n"
                        "You can now switch to this model in the Model Management section."
                    )

                    # Refresh model list
                    self.refresh_model_list()

                    del self.finetuning_task_id
                    del self.finetuning_model_name

                elif status == 'failed':
                    self.finetune_poll_timer.stop()
                    self.control_panel.show_finetuning_progress(False)

                    QMessageBox.critical(
                        self, "Finetuning Failed",
                        f"Finetuning failed:\n{message}"
                    )

                    del self.finetuning_task_id
                    del self.finetuning_model_name

        except Exception as e:
            print(f"Error checking finetuning status: {e}")

    def refresh_model_list(self):
        """Refresh the available models list"""
        try:
            response = self.api_client.list_models()
            if response and response.get('success'):
                models = response.get('models', [])
                active_model = response.get('active_model', 'default')

                # Update combo box
                self.control_panel.model_selector.clear()
                for model in models:
                    self.control_panel.model_selector.addItem(model['name'])

                # Set active model
                self.control_panel.update_active_model(active_model)

        except Exception as e:
            print(f"Error refreshing model list: {e}")

    @pyqtSlot(str)
    def switch_model(self, model_name: str):
        """Switch to a different model"""
        try:
            response = self.api_client.switch_model(model_name)
            if response and response.get('success'):
                QMessageBox.information(self, "Success", f"Switched to model: {model_name}")
                self.control_panel.update_active_model(model_name)
                self.statusBar().showMessage(f"Active model: {model_name}")
            else:
                raise Exception("Failed to switch model")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to switch model:\n{str(e)}")
    
    def start_bbox_creation(self):
        """Toggle bounding box creation mode"""
        if self.bbox_editor.creating_bbox:
            self.bbox_editor.stop_bbox_creation()
            self.control_panel.add_bbox_btn.setText("➕ Add Box")
            self.control_panel.add_bbox_btn.setStyleSheet("")
            self.statusBar().showMessage("Click boxes to delete them")
        else:
            self.bbox_editor.start_bbox_creation()
            self.control_panel.add_bbox_btn.setText("⏹️ Stop Adding")
            self.control_panel.add_bbox_btn.setStyleSheet("background-color: #e74c3c;")
            self.statusBar().showMessage("Click and drag to create bounding boxes")
    
    def delete_selected_bbox(self):
        """Delete the currently selected bounding box"""
        self.bbox_editor.delete_selected_bbox()
        self.statusBar().showMessage("Deleted selected bounding box")
    
    def clear_all_bboxes(self):
        """Clear all bounding boxes"""
        if not self.bbox_editor.bboxes:
            return
            
        reply = QMessageBox.question(
            self,
            "Clear All Annotations",
            "Clear all bounding boxes?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.bbox_editor.clear_bboxes()
            self.current_annotations.clear()
            self.statusBar().showMessage("Cleared all annotations")

    def undo_last_action(self):
        """Undo the last annotation action"""
        if self.bbox_editor.undo():
            self.current_annotations = self.bbox_editor.get_bboxes()
            self.statusBar().showMessage("Undid last action")
        else:
            self.statusBar().showMessage("Nothing to undo")

    def save_current_patch_annotations(self):
        """Save current patch annotations for training"""
        if not self.current_annotations:
            QMessageBox.warning(self, "No Annotations", "No annotations to save")
            return

        if not self.control_panel.backend_connected:
            QMessageBox.information(
                self, "Offline Mode",
                "Saving annotations requires backend connection."
            )
            return

        try:
            # Save current patch to temporary file
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            if not self.tiff_processor.save_patch_to_file(self.current_patch_index, temp_path):
                raise Exception("Failed to save patch")

            # Save annotations to backend
            response = self.api_client.save_patch_annotations(
                self.current_patch_index,
                temp_path,
                self.current_annotations
            )

            if response and response.get('success'):
                # Get updated count
                count_response = self.api_client.get_patch_annotations_count()
                count = count_response.get('count', 0) if count_response else 0

                QMessageBox.information(
                    self, "Success",
                    f"Annotations saved for training!\n\n"
                    f"Total accumulated patches: {count}\n\n"
                    "You can continue annotating more patches or start finetuning."
                )

                # Update status
                self.control_panel.update_training_count(count)
                self.statusBar().showMessage(f"Saved annotations ({count} patches accumulated)")
            else:
                raise Exception("Failed to save annotations")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save annotations:\n{str(e)}")

    def closeEvent(self, event):
        """Handle window close event"""
        # Stop any running background threads
        if hasattr(self, 'upload_thread') and self.upload_thread.isRunning():
            self.upload_thread.quit()
            self.upload_thread.wait(1000)  # Wait max 1 second
        
        if hasattr(self, 'predict_thread') and self.predict_thread.isRunning():
            self.predict_thread.quit()
            self.predict_thread.wait(1000)  # Wait max 1 second
        
        # Clean up memory
        self.tiff_processor.cleanup_memory(aggressive=True)
        event.accept()


def signal_handler(signum, frame):
    """Handle keyboard interrupt signal"""
    print("\nClosing application...")
    QApplication.quit()


def main():
    """Main application entry point"""
    signal.signal(signal.SIGINT, signal_handler)

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("Interactive AI Workflow")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("AI Research Lab")

        window = MainWindow()
        window.show()

        return app.exec_()

    except Exception as e:
        print(f"\nApplication error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())