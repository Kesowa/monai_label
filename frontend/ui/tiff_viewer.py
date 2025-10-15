"""
TIFF Viewer Widget with Bounding Box Support
Handles image display, zoom, and annotation overlay

COORDINATE SYSTEM:
- IMAGE coordinates: Actual pixel coordinates in the source image (0 to image width/height)
- WIDGET coordinates: Display coordinates accounting for zoom and centering
- All bboxes stored in IMAGE coordinates, converted to WIDGET only for display
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from PyQt5.QtWidgets import QWidget, QLabel, QScrollArea
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPoint, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QMouseEvent, QPaintEvent, QImage


class TiffViewer(QLabel):
    """Custom image viewer with zoom and pan capabilities"""
    
    # Signals
    image_clicked = pyqtSignal(QPoint)  # Emitted with IMAGE coordinates
    zoom_changed = pyqtSignal(float)    # Emitted when zoom level changes
    
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("border: 1px solid #cccccc; background-color: white;")
        
        # Image data
        self._original_pixmap = None
        self._scaled_pixmap = None
        self._image_width = 0
        self._image_height = 0
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 10.0
        
        # Pan functionality
        self._pan_start_pos = QPoint()
        self._panning = False
        self._image_offset = QPoint(0, 0)
        
        # Enable mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Reference to bbox editor (set externally)
        self._bbox_editor = None
        
    def set_image(self, image_array: np.ndarray):
        """Set image from numpy array"""
        try:
            # Clear previous display
            self.clear()
            
            # Convert numpy array to QPixmap
            if len(image_array.shape) == 3:
                height, width, channels = image_array.shape
                if channels == 4:
                    image_array = image_array[:, :, :3]
                
                # Normalize to 0-255 range
                if image_array.dtype != np.uint8:
                    img_min, img_max = image_array.min(), image_array.max()
                    if img_max > img_min:
                        image_array = ((image_array - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
                    else:
                        image_array = np.zeros_like(image_array, dtype=np.uint8)
                
                # Convert to QImage then QPixmap
                bytes_per_line = 3 * width
                q_image = QImage(image_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
                # Make a copy to avoid memory issues
                q_image = q_image.copy()
                self._original_pixmap = QPixmap.fromImage(q_image)
                
            else:
                # Grayscale
                height, width = image_array.shape
                if image_array.dtype != np.uint8:
                    img_min, img_max = image_array.min(), image_array.max()
                    if img_max > img_min:
                        image_array = ((image_array - img_min) * 255 / (img_max - img_min)).astype(np.uint8)
                    else:
                        image_array = np.zeros_like(image_array, dtype=np.uint8)
                
                q_image = QImage(image_array.tobytes(), width, height, width, QImage.Format_Grayscale8)
                q_image = q_image.copy()
                self._original_pixmap = QPixmap.fromImage(q_image)
            
            # Store original image dimensions
            self._image_width = width
            self._image_height = height
            
            # Reset zoom and position
            self._zoom_factor = 1.0
            self._image_offset = QPoint(0, 0)
            
            # Update display
            self._update_display()
            
        except Exception as e:
            print(f"Error setting image: {e}")
            import traceback
            traceback.print_exc()
            self.setText("Failed to load image")
            self._image_width = 0
            self._image_height = 0
    
    def clear(self):
        """Clear the current image"""
        self._original_pixmap = None
        self._scaled_pixmap = None
        self._image_width = 0
        self._image_height = 0
        self.setPixmap(QPixmap())
    
    def _update_display(self):
        """Update the displayed image based on zoom and pan"""
        if not self._original_pixmap:
            return
        
        old_zoom = self._zoom_factor
        
        # Avoid redundant updates
        if self._scaled_pixmap and abs(old_zoom - self._zoom_factor) < 0.001:
            return
        
        # Scale the image
        scaled_size = self._original_pixmap.size() * self._zoom_factor
        self._scaled_pixmap = self._original_pixmap.scaled(
            scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        # Set the pixmap
        self.setPixmap(self._scaled_pixmap)
        
        # Update widget size for scroll area
        if self._scaled_pixmap:
            self.setFixedSize(self._scaled_pixmap.size())
        
        # Emit zoom changed signal only if it actually changed
        if abs(old_zoom - self._zoom_factor) >= 0.001:
            self.zoom_changed.emit(self._zoom_factor)
    
    def zoom_in(self, factor: float = 1.2):
        """Zoom in by given factor"""
        new_zoom = self._zoom_factor * factor
        if new_zoom <= self._max_zoom:
            self._zoom_factor = new_zoom
            self._update_display()
    
    def zoom_out(self, factor: float = 1.2):
        """Zoom out by given factor"""
        new_zoom = self._zoom_factor / factor
        if new_zoom >= self._min_zoom:
            self._zoom_factor = new_zoom
            self._update_display()
    
    def set_zoom(self, zoom_factor: float):
        """Set specific zoom level"""
        old_zoom = self._zoom_factor
        zoom_factor = max(self._min_zoom, min(self._max_zoom, zoom_factor))
        
        # Avoid redundant updates
        if abs(old_zoom - zoom_factor) < 0.001:
            return
        
        self._zoom_factor = zoom_factor
        self._update_display()
    
    def fit_to_window(self):
        """Fit image to window size"""
        if not self._original_pixmap:
            return
        
        widget_size = self.size()
        image_size = self._original_pixmap.size()
        
        # Calculate zoom to fit
        zoom_x = widget_size.width() / image_size.width()
        zoom_y = widget_size.height() / image_size.height()
        zoom = min(zoom_x, zoom_y, 1.0)
        
        self.set_zoom(zoom)
    
    def get_zoom_factor(self) -> float:
        """Get current zoom factor"""
        return self._zoom_factor
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get original image dimensions"""
        return (self._image_width, self._image_height)
    
    def image_to_widget_coords(self, image_point: QPoint) -> QPoint:
        """Convert IMAGE coordinates to WIDGET coordinates"""
        if not self._scaled_pixmap:
            return image_point
        
        # Scale by zoom factor
        scaled_x = int(image_point.x() * self._zoom_factor)
        scaled_y = int(image_point.y() * self._zoom_factor)
        
        # Account for centering offset
        widget_size = self.size()
        image_size = self._scaled_pixmap.size()
        
        offset_x = (widget_size.width() - image_size.width()) // 2
        offset_y = (widget_size.height() - image_size.height()) // 2
        
        return QPoint(
            scaled_x + offset_x + self._image_offset.x(),
            scaled_y + offset_y + self._image_offset.y()
        )
    
    def widget_to_image_coords(self, widget_point: QPoint) -> QPoint:
        """Convert WIDGET coordinates to IMAGE coordinates"""
        if not self._scaled_pixmap or self._zoom_factor == 0:
            return QPoint(0, 0)
        
        # Account for centering offset
        widget_size = self.size()
        image_size = self._scaled_pixmap.size()
        
        offset_x = (widget_size.width() - image_size.width()) // 2
        offset_y = (widget_size.height() - image_size.height()) // 2
        
        adjusted_x = widget_point.x() - offset_x - self._image_offset.x()
        adjusted_y = widget_point.y() - offset_y - self._image_offset.y()
        
        # Unscale by zoom factor
        image_x = int(adjusted_x / self._zoom_factor)
        image_y = int(adjusted_y / self._zoom_factor)
        
        # Clamp to image bounds
        image_x = max(0, min(image_x, self._image_width - 1))
        image_y = max(0, min(image_y, self._image_height - 1))
        
        return QPoint(image_x, image_y)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events"""
        if event.button() == Qt.LeftButton:
            if self._bbox_editor:
                # Convert to image coordinates immediately
                image_pos = self.widget_to_image_coords(event.pos())
                self._bbox_editor.handle_mouse_press(image_pos)
            else:
                image_pos = self.widget_to_image_coords(event.pos())
                self.image_clicked.emit(image_pos)
                
        elif event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start_pos = event.pos()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move events"""
        if self._panning:
            delta = event.pos() - self._pan_start_pos
            self._image_offset += delta
            self._pan_start_pos = event.pos()
            self.update()
            
        elif self._bbox_editor and self._bbox_editor.creating_bbox:
            # Convert to image coordinates
            image_pos = self.widget_to_image_coords(event.pos())
            self._bbox_editor.handle_mouse_move(image_pos)
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release events"""
        if event.button() == Qt.MiddleButton:
            self._panning = False
            
        elif event.button() == Qt.LeftButton and self._bbox_editor:
            if self._bbox_editor.creating_bbox:
                # Convert to image coordinates
                image_pos = self.widget_to_image_coords(event.pos())
                self._bbox_editor.handle_mouse_release(image_pos)
    
    def wheelEvent(self, event):
        """Handle wheel events for zooming"""
        if event.angleDelta().y() > 0:
            self.zoom_in(1.1)
        else:
            self.zoom_out(1.1)
    
    def paintEvent(self, event):
        """Custom paint event that draws bboxes on top of the image"""
        # Call parent paint event first to draw the image
        super().paintEvent(event)
        
        # Then draw bboxes if we have a bbox editor
        if self._bbox_editor:
            self._bbox_editor.draw_bboxes(self)


class BoundingBoxEditor(QWidget):
    """
    Handles bounding box creation, editing, and visualization
    
    IMPORTANT: All bboxes are stored in IMAGE coordinates
    Conversion to WIDGET coordinates happens only during drawing
    """
    
    # Signals
    bbox_created = pyqtSignal(dict)       # New bbox created (IMAGE coords)
    bbox_deleted = pyqtSignal(int)        # Bbox deleted (index)
    bbox_modified = pyqtSignal(int, dict) # Bbox modified (index, IMAGE coords)
    bboxes_changed = pyqtSignal()         # Any change to bboxes list
    
    def __init__(self, tiff_viewer: TiffViewer):
        super().__init__()
        self.tiff_viewer = tiff_viewer
        self.bboxes = []  # All bboxes in IMAGE coordinates
        self.selected_bbox_index = -1
        self.creating_bbox = False
        self.bbox_start_pos = None  # IMAGE coordinates
        self.current_bbox = None    # IMAGE coordinates (for preview)
        
        # Undo system
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_stack = 50
        
        # Visual properties
        self.bbox_color = QColor(0, 255, 0, 100)
        self.bbox_border_color = QColor(0, 200, 0, 255)
        self.selected_color = QColor(255, 0, 0, 100)
        self.selected_border_color = QColor(255, 0, 0, 255)
        self.preview_color = QColor(255, 255, 0, 150)
        self.border_width = 2
        
        # Connect to viewer
        self.tiff_viewer._bbox_editor = self
    
    def set_bboxes(self, bboxes: List[Dict]):
        """Set the list of bounding boxes (must be in IMAGE coordinates)"""
        self.bboxes = [bbox.copy() for bbox in bboxes]
        self.selected_bbox_index = -1
        self.tiff_viewer.update()
        self.bboxes_changed.emit()
    
    def add_bbox(self, bbox: Dict):
        """Add a new bounding box (must be in IMAGE coordinates)"""
        self.save_state_for_undo()
        self.bboxes.append(bbox.copy())
        self.bbox_created.emit(bbox)
        self.tiff_viewer.update()
        self.bboxes_changed.emit()
    
    def delete_bbox(self, index: int):
        """Delete a bounding box"""
        if 0 <= index < len(self.bboxes):
            self.save_state_for_undo()
            del self.bboxes[index]
            
            if self.selected_bbox_index == index:
                self.selected_bbox_index = -1
            elif self.selected_bbox_index > index:
                self.selected_bbox_index -= 1
            
            self.bbox_deleted.emit(index)
            self.tiff_viewer.update()
            self.bboxes_changed.emit()
    
    def clear_bboxes(self):
        """Clear all bounding boxes"""
        if self.bboxes:
            self.save_state_for_undo()
            self.bboxes.clear()
            self.selected_bbox_index = -1
            self.tiff_viewer.update()
            self.bboxes_changed.emit()
    
    def start_bbox_creation(self):
        """Start creating a new bounding box"""
        self.creating_bbox = True
        self.tiff_viewer.setCursor(Qt.CrossCursor)
    
    def stop_bbox_creation(self):
        """Stop creating bounding boxes"""
        self.creating_bbox = False
        self.bbox_start_pos = None
        self.current_bbox = None
        self.tiff_viewer.setCursor(Qt.ArrowCursor)
        self.tiff_viewer.update()
    
    def save_state_for_undo(self):
        """Save current state for undo functionality"""
        self.undo_stack.append([bbox.copy() for bbox in self.bboxes])
        
        # Limit undo stack size
        if len(self.undo_stack) > self.max_undo_stack:
            self.undo_stack.pop(0)
        
        self.redo_stack.clear()
    
    def undo(self):
        """Undo the last action"""
        if self.undo_stack:
            self.redo_stack.append([bbox.copy() for bbox in self.bboxes])
            self.bboxes = self.undo_stack.pop()
            self.selected_bbox_index = -1
            self.tiff_viewer.update()
            self.bboxes_changed.emit()
            return True
        return False
    
    def redo(self):
        """Redo the last undone action"""
        if self.redo_stack:
            self.undo_stack.append([bbox.copy() for bbox in self.bboxes])
            self.bboxes = self.redo_stack.pop()
            self.selected_bbox_index = -1
            self.tiff_viewer.update()
            self.bboxes_changed.emit()
            return True
        return False
    
    def get_bboxes(self) -> List[Dict]:
        """Get current list of bounding boxes (IMAGE coordinates)"""
        return [bbox.copy() for bbox in self.bboxes]
    
    def handle_mouse_press(self, image_pos: QPoint):
        """Handle mouse press (receives IMAGE coordinates)"""
        # Check if clicking on existing bbox
        clicked_index = self.find_bbox_at_position(image_pos)
        
        if clicked_index >= 0:
            # Delete the clicked bbox
            self.delete_bbox(clicked_index)
            return
        
        # Start creating new bbox if in creation mode
        if self.creating_bbox:
            self.bbox_start_pos = image_pos
            self.current_bbox = {
                'xmin': image_pos.x(),
                'ymin': image_pos.y(),
                'xmax': image_pos.x(),
                'ymax': image_pos.y(),
                'confidence': None,
                'label': 'Tree'
            }
            self.tiff_viewer.update()
        else:
            # Deselect all
            self.selected_bbox_index = -1
            self.tiff_viewer.update()
    
    def handle_mouse_move(self, image_pos: QPoint):
        """Handle mouse move (receives IMAGE coordinates)"""
        if self.creating_bbox and self.current_bbox is not None:
            self.current_bbox['xmax'] = image_pos.x()
            self.current_bbox['ymax'] = image_pos.y()
            self.tiff_viewer.update()
    
    def handle_mouse_release(self, image_pos: QPoint):
        """Handle mouse release (receives IMAGE coordinates)"""
        if self.creating_bbox and self.current_bbox is not None:
            self.current_bbox['xmax'] = image_pos.x()
            self.current_bbox['ymax'] = image_pos.y()
            
            # Normalize coordinates (ensure min < max)
            xmin = min(self.current_bbox['xmin'], self.current_bbox['xmax'])
            xmax = max(self.current_bbox['xmin'], self.current_bbox['xmax'])
            ymin = min(self.current_bbox['ymin'], self.current_bbox['ymax'])
            ymax = max(self.current_bbox['ymin'], self.current_bbox['ymax'])
            
            # Check minimum size (in image coordinates)
            width = xmax - xmin
            height = ymax - ymin
            
            if width > 5 and height > 5:
                final_bbox = {
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'confidence': None,
                    'label': 'Tree'
                }
                self.add_bbox(final_bbox)
            
            # Reset creation state
            self.current_bbox = None
            self.bbox_start_pos = None
            self.tiff_viewer.update()
    
    def find_bbox_at_position(self, image_pos: QPoint) -> int:
        """Find bbox at given IMAGE coordinate position"""
        for i, bbox in enumerate(self.bboxes):
            if (bbox['xmin'] <= image_pos.x() <= bbox['xmax'] and
                bbox['ymin'] <= image_pos.y() <= bbox['ymax']):
                return i
        return -1
    
    def delete_selected_bbox(self):
        """Delete currently selected bounding box"""
        if self.selected_bbox_index >= 0:
            self.delete_bbox(self.selected_bbox_index)
    
    def draw_bboxes(self, viewer: TiffViewer):
        """Draw all bounding boxes (converts IMAGE to WIDGET coords for display)"""
        if not self.bboxes and not self.current_bbox:
            return
        
        painter = QPainter(viewer)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw existing bboxes
        for i, bbox in enumerate(self.bboxes):
            self._draw_single_bbox(painter, bbox, i == self.selected_bbox_index, viewer)
        
        # Draw preview bbox during creation
        if self.creating_bbox and self.current_bbox:
            self._draw_preview_bbox(painter, self.current_bbox, viewer)
        
        painter.end()
    
    def _draw_single_bbox(self, painter: QPainter, bbox: Dict, is_selected: bool, viewer: TiffViewer):
        """Draw a single bbox (converting IMAGE coords to WIDGET coords)"""
        # Convert IMAGE coordinates to WIDGET coordinates
        top_left = viewer.image_to_widget_coords(QPoint(int(bbox['xmin']), int(bbox['ymin'])))
        bottom_right = viewer.image_to_widget_coords(QPoint(int(bbox['xmax']), int(bbox['ymax'])))
        
        rect = QRect(top_left, bottom_right)
        
        # Choose colors
        if is_selected:
            fill_color = self.selected_color
            border_color = self.selected_border_color
        else:
            fill_color = self.bbox_color
            border_color = self.bbox_border_color
        
        # Draw filled rectangle
        painter.setBrush(QBrush(fill_color))
        painter.setPen(QPen(border_color, self.border_width))
        painter.drawRect(rect)
        
        # Draw confidence score if available
        if bbox.get('confidence') is not None:
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            confidence_text = f"{bbox['confidence']:.2f}"
            painter.drawText(rect.topLeft() + QPoint(5, 15), confidence_text)
    
    def _draw_preview_bbox(self, painter: QPainter, bbox: Dict, viewer: TiffViewer):
        """Draw preview bbox during creation"""
        # Convert IMAGE coordinates to WIDGET coordinates
        top_left = viewer.image_to_widget_coords(QPoint(int(bbox['xmin']), int(bbox['ymin'])))
        bottom_right = viewer.image_to_widget_coords(QPoint(int(bbox['xmax']), int(bbox['ymax'])))
        
        rect = QRect(top_left, bottom_right)
        
        # Draw dashed preview rectangle
        painter.setBrush(QBrush(self.preview_color))
        pen = QPen(QColor(255, 255, 0), self.border_width, Qt.DashLine)
        painter.setPen(pen)
        painter.drawRect(rect)