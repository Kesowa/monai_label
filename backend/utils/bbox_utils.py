import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class AnnotationManager:
    """Manages bounding box annotations and their storage"""
    
    def __init__(self, annotations_dir: str = "storage/annotations"):
        self.annotations_dir = annotations_dir
        os.makedirs(annotations_dir, exist_ok=True)
        
        # In-memory cache for quick access
        self.annotations_cache: Dict[str, dict] = {}
    
    def save_annotation(self, annotation) -> bool:
        """Save annotation to disk and cache"""
        try:
            annotation_dict = annotation.dict() if hasattr(annotation, 'dict') else annotation
            
            # Ensure annotation has required fields
            if not annotation_dict.get('id'):
                annotation_dict['id'] = str(uuid.uuid4())
            
            annotation_dict['modified_at'] = datetime.now().isoformat()
            
            # Save to file
            annotation_file = os.path.join(
                self.annotations_dir, 
                f"{annotation_dict['id']}.json"
            )
            
            with open(annotation_file, 'w') as f:
                json.dump(annotation_dict, f, indent=2)
            
            # Update cache
            self.annotations_cache[annotation_dict['id']] = annotation_dict
            
            logger.info(f"Saved annotation {annotation_dict['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
            return False
    
    def get_annotation(self, annotation_id: str) -> Optional[dict]:
        """Get a specific annotation by ID"""
        # Check cache first
        if annotation_id in self.annotations_cache:
            return self.annotations_cache[annotation_id]
        
        # Load from disk
        annotation_file = os.path.join(self.annotations_dir, f"{annotation_id}.json")
        if os.path.exists(annotation_file):
            try:
                with open(annotation_file, 'r') as f:
                    annotation = json.load(f)
                
                # Update cache
                self.annotations_cache[annotation_id] = annotation
                return annotation
                
            except Exception as e:
                logger.error(f"Error loading annotation {annotation_id}: {e}")
        
        return None
    
    def get_annotations_by_image(self, image_id: str) -> List[dict]:
        """Get all annotations for a specific image"""
        annotations = []
        
        # Check all annotation files
        for filename in os.listdir(self.annotations_dir):
            if filename.endswith('.json'):
                annotation_id = os.path.splitext(filename)[0]
                annotation = self.get_annotation(annotation_id)
                
                if annotation and annotation.get('image_id') == image_id:
                    annotations.append(annotation)
        
        # Sort by creation time (newest first)
        annotations.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return annotations
    
    def delete_annotation(self, annotation_id: str) -> bool:
        """Delete an annotation"""
        try:
            annotation_file = os.path.join(self.annotations_dir, f"{annotation_id}.json")
            
            if os.path.exists(annotation_file):
                os.remove(annotation_file)
            
            if annotation_id in self.annotations_cache:
                del self.annotations_cache[annotation_id]
            
            logger.info(f"Deleted annotation {annotation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting annotation {annotation_id}: {e}")
            return False
    
    def validate_bounding_box(self, bbox: dict) -> bool:
        """Validate a bounding box structure"""
        required_fields = ['xmin', 'ymin', 'xmax', 'ymax']
        
        for field in required_fields:
            if field not in bbox:
                return False
            
            try:
                float(bbox[field])
            except (ValueError, TypeError):
                return False
        
        # Check logical consistency
        if bbox['xmin'] >= bbox['xmax'] or bbox['ymin'] >= bbox['ymax']:
            return False
        
        return True
    
    def convert_bbox_format(self, bbox: dict, from_format: str, to_format: str, 
                           image_width: int = None, image_height: int = None) -> dict:
        """
        Convert bounding box between different formats
        
        Args:
            bbox: Bounding box dictionary
            from_format: 'absolute' (pixel coords) or 'normalized' (0-1 range)
            to_format: 'absolute' (pixel coords) or 'normalized' (0-1 range)
            image_width: Required for conversion between formats
            image_height: Required for conversion between formats
        """
        if from_format == to_format:
            return bbox.copy()
        
        if not image_width or not image_height:
            raise ValueError("Image dimensions required for format conversion")
        
        result = bbox.copy()
        
        if from_format == 'normalized' and to_format == 'absolute':
            # Convert from normalized (0-1) to absolute pixel coordinates
            result['xmin'] = bbox['xmin'] * image_width
            result['ymin'] = bbox['ymin'] * image_height
            result['xmax'] = bbox['xmax'] * image_width
            result['ymax'] = bbox['ymax'] * image_height
            
        elif from_format == 'absolute' and to_format == 'normalized':
            # Convert from absolute pixel coordinates to normalized (0-1)
            result['xmin'] = bbox['xmin'] / image_width
            result['ymin'] = bbox['ymin'] / image_height
            result['xmax'] = bbox['xmax'] / image_width
            result['ymax'] = bbox['ymax'] / image_height
        
        return result
    
    def merge_overlapping_boxes(self, bboxes: List[dict], iou_threshold: float = 0.5) -> List[dict]:
        """Merge overlapping bounding boxes using IoU threshold"""
        if not bboxes:
            return []
        
        def calculate_iou(box1, box2):
            """Calculate Intersection over Union (IoU) of two bounding boxes"""
            # Calculate intersection area
            x1 = max(box1['xmin'], box2['xmin'])
            y1 = max(box1['ymin'], box2['ymin'])
            x2 = min(box1['xmax'], box2['xmax'])
            y2 = min(box1['ymax'], box2['ymax'])
            
            if x1 >= x2 or y1 >= y2:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            
            # Calculate union area
            area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
            area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        def merge_boxes(box1, box2):
            """Merge two bounding boxes"""
            return {
                'xmin': min(box1['xmin'], box2['xmin']),
                'ymin': min(box1['ymin'], box2['ymin']),
                'xmax': max(box1['xmax'], box2['xmax']),
                'ymax': max(box1['ymax'], box2['ymax']),
                'confidence': max(box1.get('confidence', 0.5), box2.get('confidence', 0.5)),
                'label': box1.get('label', 'Tree')
            }
        
        merged = []
        for current_box in bboxes:
            merged_with_existing = False
            
            for i, existing_box in enumerate(merged):
                if calculate_iou(current_box, existing_box) > iou_threshold:
                    # Merge boxes
                    merged[i] = merge_boxes(current_box, existing_box)
                    merged_with_existing = True
                    break
            
            if not merged_with_existing:
                merged.append(current_box.copy())
        
        return merged
    
    def filter_boxes_by_confidence(self, bboxes: List[dict], min_confidence: float = 0.5) -> List[dict]:
        """Filter bounding boxes by confidence threshold"""
        return [bbox for bbox in bboxes if bbox.get('confidence', 0.5) >= min_confidence]
    
    def get_annotation_statistics(self, image_id: str) -> dict:
        """Get statistics for annotations of an image"""
        annotations = self.get_annotations_by_image(image_id)
        
        if not annotations:
            return {
                "total_annotations": 0,
                "total_boxes": 0,
                "average_confidence": 0,
                "labels": {}
            }
        
        total_boxes = 0
        confidences = []
        label_counts = {}
        
        for annotation in annotations:
            boxes = annotation.get('bounding_boxes', [])
            total_boxes += len(boxes)
            
            for box in boxes:
                conf = box.get('confidence', 0.5)
                confidences.append(conf)
                
                label = box.get('label', 'Tree')
                label_counts[label] = label_counts.get(label, 0) + 1
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            "total_annotations": len(annotations),
            "total_boxes": total_boxes,
            "average_confidence": round(avg_confidence, 3),
            "labels": label_counts
        }
    
    def export_annotations(self, image_id: str, format: str = 'json') -> str:
        """Export annotations in various formats"""
        annotations = self.get_annotations_by_image(image_id)
        
        if format.lower() == 'json':
            return json.dumps(annotations, indent=2)
        
        elif format.lower() == 'csv':
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(['annotation_id', 'image_id', 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'label'])
            
            # Data rows
            for annotation in annotations:
                for box in annotation.get('bounding_boxes', []):
                    writer.writerow([
                        annotation['id'],
                        annotation['image_id'],
                        box['xmin'],
                        box['ymin'],
                        box['xmax'],
                        box['ymax'],
                        box.get('confidence', 0.5),
                        box.get('label', 'Tree')
                    ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {format}")