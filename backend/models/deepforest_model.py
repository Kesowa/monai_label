import os
import json
import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Proj, transform
from shapely.geometry import Polygon
from deepforest import main
from typing import List, Optional, Tuple
import logging
import math

logger = logging.getLogger(__name__)

class InteractiveDeepForest:
    """
    Adapted DeepForest model for interactive annotation workflow
    Based on the original rs.py implementation
    """
    
    def __init__(self, model_path: str = "my_model.pt", slice_size: Tuple[int, int] = (700, 700)):
        """Initialize the DeepForest model"""
        self.model = main.deepforest()
        self.model.use_release()
        self.model_path = model_path
        self.slice_size = slice_size
        
        # Load the finetuned model if it exists
        if os.path.exists(model_path):
            self.load_model(model_path)
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model file {model_path} not found, using pretrained model")
            
        # WGS84 projection for coordinate conversion
        self.wgs84_proj = Proj(proj='latlong', datum='WGS84')
    
    def _validate_bbox(self, bbox: dict) -> bool:
        """
        Validate bounding box data to ensure it's valid for Pydantic models
        
        Args:
            bbox: Dictionary containing bbox coordinates
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check all required fields exist
            required_fields = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'label']
            for field in required_fields:
                if field not in bbox:
                    logger.warning(f"Missing field '{field}' in bbox")
                    return False
            
            # Check if all numeric values are valid (not NaN, not Inf)
            numeric_fields = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence']
            for field in numeric_fields:
                value = bbox[field]
                # Check if it's a valid number
                if not isinstance(value, (int, float)):
                    logger.warning(f"Field '{field}' is not numeric: {type(value)}")
                    return False
                # Check for NaN and Inf
                if math.isnan(value) or math.isinf(value):
                    logger.warning(f"Field '{field}' has invalid value: {value}")
                    return False
            
            # Check logical consistency
            if bbox['xmin'] >= bbox['xmax']:
                logger.warning(f"Invalid bbox: xmin ({bbox['xmin']}) >= xmax ({bbox['xmax']})")
                return False
            if bbox['ymin'] >= bbox['ymax']:
                logger.warning(f"Invalid bbox: ymin ({bbox['ymin']}) >= ymax ({bbox['ymax']})")
                return False
            
            # Check confidence is in valid range
            if not (0.0 <= bbox['confidence'] <= 1.0):
                logger.warning(f"Confidence out of range [0, 1]: {bbox['confidence']}")
                return False
            
            # Check label is a string
            if not isinstance(bbox['label'], str):
                logger.warning(f"Label is not a string: {type(bbox['label'])}")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating bbox: {e}")
            return False
        
    def load_model(self, model_path: str):
        """Load a saved model state dictionary"""
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model_path = model_path
            logger.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            raise
    
    def predict_image(self, image_path: str, return_coords: str = "pixel", is_patch: bool = False) -> List[dict]:
        """
        Predict bounding boxes for a TIFF image

        Args:
            image_path: Path to the TIFF image
            return_coords: "pixel" for pixel coordinates, "geo" for geographic coordinates
            is_patch: If True, treat as a single patch without sliding window

        Returns:
            List of bounding box predictions
        """
        try:
            predictions = []

            # Open the TIFF file using rasterio
            with rasterio.open(image_path) as src:
                self.transform_matrix = src.transform
                self.src_crs = src.crs
                self.image_height = src.height
                self.image_width = src.width

                # If this is already a patch, predict directly without sliding window
                if is_patch or (self.image_height <= self.slice_size[1] and self.image_width <= self.slice_size[0]):
                    # Read entire image
                    image_data = src.read()
                    image_data = image_data.transpose(1, 2, 0)

                    # Ensure RGB format
                    if image_data.shape[2] == 4:
                        image_data = image_data[:, :, :3]
                    elif image_data.shape[2] == 1:
                        image_data = np.stack([image_data[:,:,0]] * 3, axis=2)

                    # Run prediction directly
                    slice_predictions = self.model.predict_image(
                        image=image_data,
                        return_plot=False
                    )

                    if slice_predictions is not None and not slice_predictions.empty:
                        for _, row in slice_predictions.iterrows():
                            try:
                                bbox = {
                                    "xmin": float(row['xmin']),
                                    "ymin": float(row['ymin']),
                                    "xmax": float(row['xmax']),
                                    "ymax": float(row['ymax']),
                                    "confidence": float(row.get('score', 0.5)),
                                    "label": "Tree"
                                }

                                if self._validate_bbox(bbox):
                                    predictions.append(bbox)

                            except (ValueError, TypeError, KeyError) as e:
                                logger.debug(f"Error processing prediction row: {e}")
                                continue
                else:
                    # Process large image with sliding window
                    for y in range(0, self.image_height, self.slice_size[1]):
                        for x in range(0, self.image_width, self.slice_size[0]):
                            x1, y1 = x, y
                            x2 = min(x + self.slice_size[0], self.image_width)
                            y2 = min(y + self.slice_size[1], self.image_height)

                            # Read window from image
                            window = Window(x1, y1, x2 - x1, y2 - y1)
                            slice_image = src.read(window=window)

                            # Transpose to (height, width, channels)
                            slice_image = slice_image.transpose(1, 2, 0)

                            # Ensure RGB format (strip alpha if present)
                            if slice_image.shape[2] == 4:
                                slice_image = slice_image[:, :, :3]

                            # Run prediction on slice
                            slice_predictions = self.model.predict_image(
                                image=slice_image,
                                return_plot=False
                            )

                            if slice_predictions is not None and not slice_predictions.empty:
                                # Adjust bounding boxes for slice offset
                                for _, row in slice_predictions.iterrows():
                                    try:
                                        bbox = {
                                            "xmin": float(row['xmin'] + x1),
                                            "ymin": float(row['ymin'] + y1),
                                            "xmax": float(row['xmax'] + x1),
                                            "ymax": float(row['ymax'] + y1),
                                            "confidence": float(row.get('score', 0.5)),
                                            "label": "Tree"
                                        }

                                        # Validate bbox before adding to predictions
                                        if not self._validate_bbox(bbox):
                                            logger.debug(f"Skipping invalid bbox at position ({x1}, {y1})")
                                            continue

                                        # Convert to geographic coordinates if requested
                                        if return_coords == "geo":
                                            bbox = self._convert_to_geo_coords(bbox)
                                            # Validate again after coordinate conversion
                                            if not self._validate_bbox(bbox):
                                                logger.debug(f"Skipping invalid bbox after geo conversion")
                                                continue

                                        predictions.append(bbox)

                                    except (ValueError, TypeError, KeyError) as e:
                                        logger.debug(f"Error processing prediction row: {e}")
                                        continue

            logger.info(f"Generated {len(predictions)} predictions for {image_path}")
            return predictions

        except Exception as e:
            logger.error(f"Error predicting on image {image_path}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _convert_to_geo_coords(self, bbox: dict) -> dict:
        """Convert pixel coordinates to geographic coordinates"""
        try:
            # Convert pixel coordinates to projected coordinates
            xmin_proj, ymin_proj = rasterio.transform.xy(
                self.transform_matrix, bbox['ymin'], bbox['xmin']
            )
            xmax_proj, ymax_proj = rasterio.transform.xy(
                self.transform_matrix, bbox['ymax'], bbox['xmax']
            )
            
            # Convert to lat/lon
            xmin_geo, ymin_geo = transform(
                Proj(self.src_crs), self.wgs84_proj, xmin_proj, ymin_proj
            )
            xmax_geo, ymax_geo = transform(
                Proj(self.src_crs), self.wgs84_proj, xmax_proj, ymax_proj
            )
            
            return {
                "xmin": xmin_geo,
                "ymin": ymin_geo,
                "xmax": xmax_geo,
                "ymax": ymax_geo,
                "confidence": bbox['confidence'],
                "label": bbox['label']
            }
        except Exception as e:
            logger.error(f"Error converting coordinates: {e}")
            return bbox
    
    def pixel_to_geo_coordinates(self, x_pixel: float, y_pixel: float) -> Tuple[float, float]:
        """Convert pixel coordinates to geographic coordinates (lat/lon)"""
        # Convert pixel coordinates to projected coordinates
        x_proj, y_proj = rasterio.transform.xy(self.transform_matrix, y_pixel, x_pixel)
        
        # Convert projected coordinates to lat/lon (WGS84)
        lon, lat = transform(Proj(self.src_crs), self.wgs84_proj, x_proj, y_proj)
        return lon, lat
    
    def create_training_data(self, image_path: str, annotations: List[dict]) -> str:
        """
        Create training data from annotations for finetuning
        
        Args:
            image_path: Path to the TIFF image
            annotations: List of bounding box annotations
            
        Returns:
            Path to the created training data file
        """
        try:
            # Create training data directory if it doesn't exist
            train_dir = "storage/training_data"
            os.makedirs(train_dir, exist_ok=True)
            
            # Create annotations in format expected by DeepForest
            training_annotations = []
            
            for i, bbox in enumerate(annotations):
                training_annotations.append({
                    "image_path": image_path,
                    "xmin": bbox["xmin"],
                    "ymin": bbox["ymin"],
                    "xmax": bbox["xmax"],
                    "ymax": bbox["ymax"],
                    "label": bbox.get("label", "Tree")
                })
            
            # Save training annotations
            train_file = os.path.join(train_dir, "training_annotations.csv")
            
            import pandas as pd
            df = pd.DataFrame(training_annotations)
            df.to_csv(train_file, index=False)
            
            logger.info(f"Created training data file: {train_file}")
            return train_file
            
        except Exception as e:
            logger.error(f"Error creating training data: {e}")
            raise
    
    def finetune_model(self, training_data_path: str, output_path: str, epochs: int = 10):
        """
        Finetune the model with new training data

        Args:
            training_data_path: Path to CSV file with training annotations
            output_path: Path to save the finetuned model
            epochs: Number of training epochs
        """
        try:
            logger.info(f"Starting model finetuning with {training_data_path}")

            # Load training data
            import pandas as pd
            train_df = pd.read_csv(training_data_path)

            logger.info(f"Training data: {len(train_df)} annotations")

            # Configure training parameters
            self.model.config["train"]["csv_file"] = training_data_path
            self.model.config["train"]["root_dir"] = "/"  # Use absolute paths in CSV
            self.model.config["train"]["epochs"] = epochs
            self.model.config["train"]["lr"] = 0.001
            self.model.config["batch_size"] = 4

            # Fix the verbose parameter issue in PyTorch Lightning
            # Monkey patch the configure_optimizers to remove verbose parameter
            original_configure_optimizers = self.model.configure_optimizers

            def patched_configure_optimizers():
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.model.config["train"]["lr"],
                    momentum=0.9
                )
                # Use StepLR instead of ReduceLROnPlateau since we don't have validation data
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=5,  # Reduce LR every 5 epochs
                    gamma=0.1     # Multiply LR by 0.1
                )
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'epoch'
                    }
                }

            self.model.configure_optimizers = patched_configure_optimizers

            # Force CPU training to avoid MPS float64 issues on Apple Silicon
            # MPS doesn't support float64, which causes issues with some operations
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

            # Override create_trainer to force CPU
            original_create_trainer = self.model.create_trainer

            def patched_create_trainer():
                from pytorch_lightning import Trainer
                from pytorch_lightning.callbacks import ModelCheckpoint

                # Force CPU to avoid MPS compatibility issues
                self.model.trainer = Trainer(
                    max_epochs=self.model.config["train"]["epochs"],
                    accelerator='cpu',  # Force CPU
                    devices=1,
                    enable_checkpointing=False,
                    logger=False,
                    enable_progress_bar=True
                )

            self.model.create_trainer = patched_create_trainer

            # Create trainer
            self.model.create_trainer()

            # Load training dataset
            train_dataset = self.model.load_dataset(
                csv_file=training_data_path,
                root_dir="/",
                augment=True
            )

            logger.info(f"Training for {epochs} epochs on CPU (MPS not supported for training)...")

            # Train using the trainer
            self.model.trainer.fit(self.model, train_dataset)

            # Restore original methods
            self.model.create_trainer = original_create_trainer

            # Restore original configure_optimizers
            self.model.configure_optimizers = original_configure_optimizers

            # Save the finetuned model
            torch.save(self.model.state_dict(), output_path)
            self.model_path = output_path

            logger.info(f"Model finetuning completed. Saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error during model finetuning: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        return {
            "model_path": self.model_path,
            "slice_size": self.slice_size,
            "model_type": "DeepForest",
            "loaded": os.path.exists(self.model_path) if self.model_path else False
        }