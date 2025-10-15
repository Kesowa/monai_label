import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import logging
from threading import Lock

from .deepforest_model import InteractiveDeepForest

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages multiple model versions and handles model switching"""
    
    def __init__(self, models_dir: str = "storage/models"):
        self.models_dir = models_dir
        self.models: Dict[str, InteractiveDeepForest] = {}
        self.active_model_name = "default"
        self.finetuning_tasks: Dict[str, dict] = {}
        self.lock = Lock()
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        # Initialize with default model (copy from root my_model.pt if exists)
        self._initialize_default_model()
        
    def _initialize_default_model(self):
        """Initialize the default model"""
        try:
            default_model_path = os.path.join(self.models_dir, "default.pt")
            
            # Copy the original model if it exists
            if os.path.exists("my_model.pt") and not os.path.exists(default_model_path):
                shutil.copy2("my_model.pt", default_model_path)
                logger.info("Copied my_model.pt to default model")
            
            # Load the default model
            if os.path.exists(default_model_path):
                self.models["default"] = InteractiveDeepForest(default_model_path)
                logger.info("Loaded default model")
            else:
                # Create a new model with pretrained weights
                self.models["default"] = InteractiveDeepForest()
                logger.info("Created default model with pretrained weights")
                
            self.active_model_name = "default"
            
        except Exception as e:
            logger.error(f"Error initializing default model: {e}")
            # Fallback: create a basic model
            self.models["default"] = InteractiveDeepForest()
            self.active_model_name = "default"
    
    def get_active_model(self) -> InteractiveDeepForest:
        """Get the currently active model"""
        with self.lock:
            return self.models[self.active_model_name]
    
    def get_active_model_name(self) -> str:
        """Get the name of the currently active model"""
        return self.active_model_name
    
    def list_models(self) -> List[dict]:
        """List all available models"""
        models_info = []
        
        # Add loaded models
        for name, model in self.models.items():
            models_info.append({
                "name": name,
                "path": model.model_path,
                "loaded": True,
                "active": name == self.active_model_name,
                "info": model.get_model_info()
            })
        
        # Add unloaded models from disk
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pt'):
                model_name = os.path.splitext(filename)[0]
                if model_name not in self.models:
                    models_info.append({
                        "name": model_name,
                        "path": os.path.join(self.models_dir, filename),
                        "loaded": False,
                        "active": False,
                        "info": {"model_type": "DeepForest"}
                    })
        
        return sorted(models_info, key=lambda x: x["name"])
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model"""
        try:
            with self.lock:
                # If model is already loaded, just switch
                if model_name in self.models:
                    self.active_model_name = model_name
                    logger.info(f"Switched to model: {model_name}")
                    return True
                
                # Try to load model from disk
                model_path = os.path.join(self.models_dir, f"{model_name}.pt")
                if os.path.exists(model_path):
                    self.models[model_name] = InteractiveDeepForest(model_path)
                    self.active_model_name = model_name
                    logger.info(f"Loaded and switched to model: {model_name}")
                    return True
                
                logger.warning(f"Model {model_name} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error switching to model {model_name}: {e}")
            return False
    
    def create_model_version(self, base_model_name: str, new_model_name: str) -> bool:
        """Create a new model version from an existing model"""
        try:
            if base_model_name not in self.models:
                logger.error(f"Base model {base_model_name} not found")
                return False
            
            base_model = self.models[base_model_name]
            new_model_path = os.path.join(self.models_dir, f"{new_model_name}.pt")
            
            # Copy the model file
            shutil.copy2(base_model.model_path, new_model_path)
            
            # Create new model instance
            self.models[new_model_name] = InteractiveDeepForest(new_model_path)
            
            logger.info(f"Created model version: {new_model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating model version {new_model_name}: {e}")
            return False
    
    def finetune_model(self, task_id: str, image_id: str, annotation, new_model_name: str):
        """Background task for model finetuning"""
        try:
            # Update task status
            self.finetuning_tasks[task_id] = {
                "status": "started",
                "progress": 0,
                "message": "Initializing finetuning...",
                "start_time": datetime.now().isoformat(),
                "image_id": image_id,
                "model_name": new_model_name
            }
            
            logger.info(f"Starting finetuning task {task_id}")
            
            # Get image path
            image_path = f"storage/uploads/{image_id}.tiff"
            if not os.path.exists(image_path):
                image_path = f"storage/uploads/{image_id}.tif"
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found for {image_id}")
            
            # Update progress
            self.finetuning_tasks[task_id]["status"] = "preparing_data"
            self.finetuning_tasks[task_id]["progress"] = 20
            self.finetuning_tasks[task_id]["message"] = "Preparing training data..."
            
            # Create training data
            current_model = self.get_active_model()
            training_data_path = current_model.create_training_data(
                image_path, 
                [box.dict() for box in annotation.bounding_boxes]
            )
            
            # Update progress
            self.finetuning_tasks[task_id]["status"] = "training"
            self.finetuning_tasks[task_id]["progress"] = 40
            self.finetuning_tasks[task_id]["message"] = "Training model..."
            
            # Create new model version
            if not self.create_model_version(self.active_model_name, f"{new_model_name}_temp"):
                raise Exception("Failed to create model version")
            
            # Get the new model for finetuning
            temp_model = self.models[f"{new_model_name}_temp"]
            new_model_path = os.path.join(self.models_dir, f"{new_model_name}.pt")
            
            # Update progress
            self.finetuning_tasks[task_id]["progress"] = 60
            self.finetuning_tasks[task_id]["message"] = "Fine-tuning in progress..."
            
            # Finetune the model
            temp_model.finetune_model(training_data_path, new_model_path, epochs=5)
            
            # Update progress
            self.finetuning_tasks[task_id]["progress"] = 90
            self.finetuning_tasks[task_id]["message"] = "Saving model..."
            
            # Load the finetuned model
            self.models[new_model_name] = InteractiveDeepForest(new_model_path)
            
            # Clean up temp model
            if f"{new_model_name}_temp" in self.models:
                del self.models[f"{new_model_name}_temp"]
            
            # Complete task
            self.finetuning_tasks[task_id] = {
                "status": "completed",
                "progress": 100,
                "message": f"Model {new_model_name} finetuned successfully!",
                "start_time": self.finetuning_tasks[task_id]["start_time"],
                "end_time": datetime.now().isoformat(),
                "image_id": image_id,
                "model_name": new_model_name
            }
            
            logger.info(f"Completed finetuning task {task_id}")
            
        except Exception as e:
            logger.error(f"Error in finetuning task {task_id}: {e}")
            self.finetuning_tasks[task_id] = {
                "status": "failed",
                "progress": 0,
                "message": f"Finetuning failed: {str(e)}",
                "start_time": self.finetuning_tasks[task_id]["start_time"],
                "end_time": datetime.now().isoformat(),
                "image_id": image_id,
                "model_name": new_model_name,
                "error": str(e)
            }
    
    def get_finetuning_status(self, task_id: str) -> dict:
        """Get the status of a finetuning task"""
        if task_id in self.finetuning_tasks:
            return {
                "success": True,
                "task": self.finetuning_tasks[task_id]
            }
        else:
            return {
                "success": False,
                "message": "Task not found"
            }
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Clean up old finetuning task records"""
        try:
            current_time = datetime.now()
            tasks_to_remove = []
            
            for task_id, task in self.finetuning_tasks.items():
                if "end_time" in task:
                    end_time = datetime.fromisoformat(task["end_time"])
                    age_hours = (current_time - end_time).total_seconds() / 3600
                    
                    if age_hours > max_age_hours:
                        tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.finetuning_tasks[task_id]
                logger.info(f"Cleaned up old task: {task_id}")
                
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {e}")