import os
import json
import uuid
from datetime import datetime
from typing import List, Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from models.deepforest_model import InteractiveDeepForest
from models.model_manager import ModelManager
from utils.image_utils import TiffProcessor
from utils.bbox_utils import AnnotationManager

# Initialize FastAPI app
app = FastAPI(
    title="Interactive AI Workflow",
    description="TIFF file processing with DeepForest predictions and interactive annotation",
    version="1.0.0"
)

# Add CORS middleware for separate frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for desktop app
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Create necessary directories
os.makedirs("storage/uploads", exist_ok=True)
os.makedirs("storage/models", exist_ok=True)
os.makedirs("storage/annotations", exist_ok=True)
os.makedirs("storage/tiles", exist_ok=True)

# Initialize components
model_manager = ModelManager()
tiff_processor = TiffProcessor()
annotation_manager = AnnotationManager()

# Pydantic models
class BoundingBox(BaseModel):
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    confidence: Optional[float] = None
    label: str = "Tree"

class Annotation(BaseModel):
    id: str
    bounding_boxes: List[BoundingBox]
    image_id: str
    created_at: str
    modified_at: str

class FinetuneRequest(BaseModel):
    image_id: str
    annotation_id: str
    model_name: str

# Serve static files directly through API routes
import os
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

@app.get("/")
async def root():
    """Serve the main frontend page"""
    frontend_path = os.path.join(frontend_dir, "index.html")
    return FileResponse(frontend_path)

@app.api_route("/static/{path:path}", methods=["GET", "HEAD"])
async def serve_static(path: str):
    """Serve static files"""
    static_path = os.path.join(frontend_dir, "static", path)
    if os.path.exists(static_path):
        return FileResponse(static_path)
    else:
        raise HTTPException(status_code=404, detail="Static file not found")

@app.api_route("/tiles/{path:path}", methods=["GET", "HEAD"])
async def serve_tiles(path: str):
    """Serve generated tiles"""
    tiles_path = os.path.join("storage", "tiles", path)
    if os.path.exists(tiles_path):
        return FileResponse(tiles_path)
    else:
        raise HTTPException(status_code=404, detail="Tile not found")

@app.post("/api/upload")
async def upload_tiff(file: UploadFile = File(...)):
    """Upload a TIFF file and generate tiles for display"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.tiff', '.tif')):
            raise HTTPException(status_code=400, detail="Only TIFF files are supported")
        
        # Generate unique ID for this upload
        image_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        filename = f"{image_id}{file_extension}"
        filepath = f"storage/uploads/{filename}"
        
        # Save uploaded file (stream to avoid memory issues with large files)
        with open(filepath, "wb") as buffer:
            while True:
                chunk = await file.read(8192)  # Read in 8KB chunks
                if not chunk:
                    break
                buffer.write(chunk)
        
        # Process TIFF and generate tiles
        tile_info = tiff_processor.process_tiff(filepath, image_id)
        
        return JSONResponse({
            "success": True,
            "image_id": image_id,
            "filename": file.filename,
            "tile_info": tile_info,
            "message": "TIFF file uploaded and processed successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/api/image/{image_id}/info")
async def get_image_info(image_id: str):
    """Get information about an uploaded image"""
    try:
        info = tiff_processor.get_image_info(image_id)
        if not info:
            raise HTTPException(status_code=404, detail="Image not found")
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict/{image_id}")
async def predict_bounding_boxes(image_id: str):
    """Generate bounding box predictions for an uploaded TIFF image"""
    try:
        # Get image path
        image_path = tiff_processor.get_image_path(image_id)
        if not image_path:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Get current active model
        current_model = model_manager.get_active_model()
        
        # Run predictions
        predictions = current_model.predict_image(image_path)
        
        # Save predictions as initial annotation
        annotation_id = str(uuid.uuid4())
        annotation = Annotation(
            id=annotation_id,
            bounding_boxes=predictions,
            image_id=image_id,
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat()
        )
        
        annotation_manager.save_annotation(annotation)
        
        return JSONResponse({
            "success": True,
            "annotation_id": annotation_id,
            "predictions": [box.dict() for box in predictions],
            "count": len(predictions)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict_patch")
async def predict_patch(file: UploadFile = File(...)):
    """Generate bounding box predictions for a single patch/slice image"""
    temp_filepath = None
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Only image files are supported")

        # Save patch temporarily
        temp_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_filename = f"{temp_id}{file_extension}"
        temp_filepath = f"storage/uploads/{temp_filename}"

        # Save uploaded patch
        with open(temp_filepath, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Get current active model
        current_model = model_manager.get_active_model()

        # Run predictions on the patch (treat as single patch, no sliding window)
        predictions = current_model.predict_image(temp_filepath, is_patch=True)

        # Clean up temporary file
        try:
            if temp_filepath and os.path.exists(temp_filepath):
                os.remove(temp_filepath)
        except Exception as cleanup_error:
            print(f"Warning: Failed to cleanup temp file: {cleanup_error}")

        return JSONResponse({
            "success": True,
            "predictions": predictions,
            "count": len(predictions)
        })

    except Exception as e:
        # Clean up temporary file on error
        if temp_filepath:
            try:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
            except:
                pass

        # Log the error with traceback
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in predict_patch endpoint:\n{error_details}")

        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/annotations/{image_id}")
async def get_annotations(image_id: str):
    """Get existing annotations for an image"""
    try:
        annotations = annotation_manager.get_annotations_by_image(image_id)
        return JSONResponse({
            "success": True,
            "annotations": [ann.dict() for ann in annotations]
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/annotations/{image_id}")
async def save_annotations(image_id: str, annotation: Annotation):
    """Save modified annotations"""
    try:
        annotation.image_id = image_id
        annotation.modified_at = datetime.now().isoformat()
        annotation_manager.save_annotation(annotation)

        return JSONResponse({
            "success": True,
            "message": "Annotations saved successfully"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PatchAnnotation(BaseModel):
    patch_index: int
    patch_image_path: str
    bounding_boxes: List[BoundingBox]
    created_at: str = ""
    modified_at: str = ""

@app.post("/api/annotations/patch/save")
async def save_patch_annotation(annotation: PatchAnnotation):
    """Save annotations for a specific patch"""
    try:
        # Create annotations directory if it doesn't exist
        os.makedirs("storage/annotations/patches", exist_ok=True)

        # Save patch annotation
        annotation_file = f"storage/annotations/patches/patch_{annotation.patch_index}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        annotation_data = {
            "patch_index": annotation.patch_index,
            "patch_image_path": annotation.patch_image_path,
            "bounding_boxes": [box.dict() for box in annotation.bounding_boxes],
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat()
        }

        with open(annotation_file, 'w') as f:
            json.dump(annotation_data, f, indent=2)

        return JSONResponse({
            "success": True,
            "message": "Patch annotations saved successfully",
            "annotation_file": annotation_file
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving patch annotations: {str(e)}")

@app.get("/api/annotations/patches/count")
async def get_patch_annotations_count():
    """Get count of saved patch annotations"""
    try:
        patches_dir = "storage/annotations/patches"
        if not os.path.exists(patches_dir):
            return JSONResponse({"success": True, "count": 0, "files": []})

        annotation_files = [f for f in os.listdir(patches_dir) if f.endswith('.json')]

        return JSONResponse({
            "success": True,
            "count": len(annotation_files),
            "files": annotation_files
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/finetune")
async def start_finetuning(request: FinetuneRequest, background_tasks: BackgroundTasks):
    """Start model finetuning with modified annotations"""
    try:
        # Get annotation data
        annotation = annotation_manager.get_annotation(request.annotation_id)
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")

        # Start background finetuning task
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            model_manager.finetune_model,
            task_id=task_id,
            image_id=request.image_id,
            annotation=annotation,
            new_model_name=request.model_name
        )

        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "message": "Finetuning started. Check status with /api/finetune/status/{task_id}"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PatchFinetuneRequest(BaseModel):
    model_name: str
    epochs: int = 10
    learning_rate: float = 0.001

@app.post("/api/finetune/patches")
async def start_patch_finetuning(request: PatchFinetuneRequest, background_tasks: BackgroundTasks):
    """Start model finetuning using accumulated patch annotations"""
    try:
        # Check if we have patch annotations
        patches_dir = "storage/annotations/patches"
        if not os.path.exists(patches_dir):
            raise HTTPException(status_code=400, detail="No patch annotations found")

        annotation_files = [f for f in os.listdir(patches_dir) if f.endswith('.json')]

        if len(annotation_files) == 0:
            raise HTTPException(status_code=400, detail="No patch annotations found")

        # Start background finetuning task
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            _finetune_from_patches,
            task_id=task_id,
            model_name=request.model_name,
            epochs=request.epochs,
            learning_rate=request.learning_rate
        )

        return JSONResponse({
            "success": True,
            "task_id": task_id,
            "message": f"Finetuning started with {len(annotation_files)} patch annotations. Check status with /api/finetune/status/{{task_id}}",
            "annotation_count": len(annotation_files)
        })

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error starting finetuning: {str(e)}")

def _finetune_from_patches(task_id: str, model_name: str, epochs: int, learning_rate: float):
    """Background task for finetuning from patch annotations"""
    import pandas as pd
    import torch

    try:
        # Update task status
        model_manager.finetuning_tasks[task_id] = {
            "status": "started",
            "progress": 0,
            "message": "Loading patch annotations...",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name
        }

        print(f"Starting patch finetuning task {task_id}")

        # Load all patch annotations
        patches_dir = "storage/annotations/patches"
        annotation_files = [f for f in os.listdir(patches_dir) if f.endswith('.json')]

        all_annotations = []
        patch_images = []

        for ann_file in annotation_files:
            with open(os.path.join(patches_dir, ann_file), 'r') as f:
                patch_data = json.load(f)

                patch_image_path = patch_data.get('patch_image_path')
                if patch_image_path and os.path.exists(patch_image_path):
                    for bbox in patch_data['bounding_boxes']:
                        all_annotations.append({
                            'image_path': patch_image_path,
                            'xmin': bbox['xmin'],
                            'ymin': bbox['ymin'],
                            'xmax': bbox['xmax'],
                            'ymax': bbox['ymax'],
                            'label': bbox.get('label', 'Tree')
                        })

                    if patch_image_path not in patch_images:
                        patch_images.append(patch_image_path)

        if len(all_annotations) == 0:
            raise Exception("No valid annotations found")

        # Update progress
        model_manager.finetuning_tasks[task_id]["progress"] = 20
        model_manager.finetuning_tasks[task_id]["message"] = f"Preparing training data ({len(all_annotations)} boxes from {len(patch_images)} patches)..."

        # Create training CSV
        training_dir = "storage/training_data"
        os.makedirs(training_dir, exist_ok=True)

        training_csv = os.path.join(training_dir, f"training_{task_id}.csv")
        df = pd.DataFrame(all_annotations)
        df.to_csv(training_csv, index=False)

        print(f"Created training CSV with {len(all_annotations)} annotations")

        # Update progress
        model_manager.finetuning_tasks[task_id]["status"] = "training"
        model_manager.finetuning_tasks[task_id]["progress"] = 40
        model_manager.finetuning_tasks[task_id]["message"] = "Training model..."

        # Get current model
        current_model = model_manager.get_active_model()

        # Create output path for new model
        new_model_path = os.path.join(model_manager.models_dir, f"{model_name}.pt")

        # Update progress
        model_manager.finetuning_tasks[task_id]["progress"] = 50
        model_manager.finetuning_tasks[task_id]["message"] = f"Fine-tuning for {epochs} epochs..."

        print(f"Starting training with {epochs} epochs, lr={learning_rate}")

        # Use the finetune_model method from InteractiveDeepForest
        # This uses the proper DeepForest training flow
        current_model.finetune_model(
            training_data_path=training_csv,
            output_path=new_model_path,
            epochs=epochs
        )

        # Update progress
        model_manager.finetuning_tasks[task_id]["progress"] = 90
        model_manager.finetuning_tasks[task_id]["message"] = "Loading finetuned model..."

        # Load the new model into manager
        from models.deepforest_model import InteractiveDeepForest
        model_manager.models[model_name] = InteractiveDeepForest(new_model_path)

        # Complete task
        model_manager.finetuning_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": f"Model '{model_name}' finetuned successfully with {len(all_annotations)} annotations!",
            "start_time": model_manager.finetuning_tasks[task_id]["start_time"],
            "end_time": datetime.now().isoformat(),
            "model_name": model_name,
            "annotation_count": len(all_annotations),
            "patch_count": len(patch_images)
        }

        print(f"Completed patch finetuning task {task_id}")

    except Exception as e:
        print(f"Error in patch finetuning task {task_id}: {e}")
        import traceback
        traceback.print_exc()

        model_manager.finetuning_tasks[task_id] = {
            "status": "failed",
            "progress": 0,
            "message": f"Finetuning failed: {str(e)}",
            "start_time": model_manager.finetuning_tasks[task_id].get("start_time", datetime.now().isoformat()),
            "end_time": datetime.now().isoformat(),
            "model_name": model_name,
            "error": str(e)
        }

@app.get("/api/finetune/status/{task_id}")
async def get_finetuning_status(task_id: str):
    """Get the status of a finetuning task"""
    try:
        status = model_manager.get_finetuning_status(task_id)
        return JSONResponse(status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available model versions"""
    try:
        models = model_manager.list_models()
        return JSONResponse({
            "success": True,
            "models": models,
            "active_model": model_manager.get_active_model_name()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ModelSwitchRequest(BaseModel):
    model_name: str

@app.post("/api/models/switch")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model version"""
    try:
        success = model_manager.switch_model(request.model_name)
        if success:
            return JSONResponse({
                "success": True,
                "message": f"Switched to model: {request.model_name}",
                "active_model": request.model_name
            })
        else:
            raise HTTPException(status_code=404, detail="Model not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_model": model_manager.get_active_model_name()
    })

if __name__ == "__main__":
    # Simple startup without reload for stability
    print("🚀 Starting Interactive AI Workflow Backend...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 50)

    try:
        # Set environment variable for large request size limit
        import os
        os.environ['UVICORN_LIMIT_MAX_REQUEST_SIZE'] = str(1024 * 1024 * 1024)  # 1GB

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")