"""
FastAPI Client
Handles communication with the FastAPI backend
Includes timeout handling and basic retry logic
"""

import requests
import json
import time
from typing import List, Dict, Optional, Union
from pathlib import Path


class FastAPIClient:
    """Client for communicating with the FastAPI backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Timeout settings
        self.default_timeout = 30
        self.upload_timeout = 120
        self.predict_timeout = 180
        
        # Retry settings
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
    def _request_with_retry(self, method: str, url: str, max_retries: int = None, 
                           **kwargs) -> Optional[requests.Response]:
        """Make request with retry logic"""
        max_retries = max_retries or self.max_retries
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
            except requests.exceptions.Timeout as e:
                last_exception = e
                break
            except requests.exceptions.HTTPError as e:
                last_exception = e
                break
            except Exception as e:
                last_exception = e
                break
        
        print(f"Request failed after {max_retries} attempts: {last_exception}")
        return None
        
    def health_check(self) -> Optional[Dict]:
        """Check if backend is healthy"""
        try:
            response = self._request_with_retry(
                'GET', 
                f"{self.base_url}/api/health",
                timeout=5,
                max_retries=1
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def upload_tiff(self, file_path: str) -> Optional[Dict]:
        """Upload a TIFF file to the backend"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Use requests directly (not session) for file uploads to avoid header conflicts
            with open(file_path, 'rb') as f:
                files = {'file': (file_path.name, f, 'image/tiff')}
                
                # Don't set Content-Type header - let requests handle it for multipart/form-data
                headers = {
                    'Accept': 'application/json'
                }
                
                # Make request directly without session to avoid header conflicts
                max_retries = self.max_retries
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/upload",
                            files=files,
                            headers=headers,
                            timeout=self.upload_timeout
                        )
                        response.raise_for_status()
                        return response.json()
                    except requests.exceptions.ConnectionError as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        break
                    except requests.exceptions.Timeout as e:
                        last_exception = e
                        break
                    except requests.exceptions.HTTPError as e:
                        last_exception = e
                        print(f"HTTP Error during upload: {e}")
                        if hasattr(e.response, 'text'):
                            print(f"Response: {e.response.text}")
                        break
                    except Exception as e:
                        last_exception = e
                        break
                
                print(f"Upload failed after {max_retries} attempts: {last_exception}")
                return None
            
        except Exception as e:
            print(f"Upload failed: {e}")
            return None
    
    def get_image_info(self, image_id: str) -> Optional[Dict]:
        """Get information about an uploaded image"""
        try:
            response = self._request_with_retry(
                'GET',
                f"{self.base_url}/api/image/{image_id}/info",
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to get image info: {e}")
            return None
    
    def predict_image(self, image_id: str) -> Optional[Dict]:
        """Generate predictions for an image"""
        try:
            response = self._request_with_retry(
                'POST',
                f"{self.base_url}/api/predict/{image_id}",
                timeout=self.predict_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
    
    def predict_patch(self, patch_image_path: str) -> Optional[Dict]:
        """Generate predictions for a single patch image"""
        try:
            patch_path = Path(patch_image_path)
            if not patch_path.exists():
                raise FileNotFoundError(f"Patch not found: {patch_image_path}")
            
            # Upload patch and get predictions
            with open(patch_path, 'rb') as f:
                files = {'file': (patch_path.name, f, 'image/tiff')}
                
                headers = {
                    'Accept': 'application/json'
                }
                
                # Make request directly without session
                max_retries = self.max_retries
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        response = requests.post(
                            f"{self.base_url}/api/predict_patch",
                            files=files,
                            headers=headers,
                            timeout=self.predict_timeout
                        )
                        response.raise_for_status()
                        return response.json()
                    except requests.exceptions.ConnectionError as e:
                        last_exception = e
                        if attempt < max_retries - 1:
                            time.sleep(self.retry_delay * (attempt + 1))
                            continue
                        break
                    except requests.exceptions.Timeout as e:
                        last_exception = e
                        break
                    except requests.exceptions.HTTPError as e:
                        last_exception = e
                        print(f"HTTP Error during patch prediction: {e}")
                        if hasattr(e.response, 'text'):
                            print(f"Response: {e.response.text}")
                        break
                    except Exception as e:
                        last_exception = e
                        break
                
                print(f"Patch prediction failed after {max_retries} attempts: {last_exception}")
                return None
            
        except Exception as e:
            print(f"Patch prediction failed: {e}")
            return None
    
    def get_annotations(self, image_id: str) -> Optional[Dict]:
        """Get existing annotations for an image"""
        try:
            response = self._request_with_retry(
                'GET',
                f"{self.base_url}/api/annotations/{image_id}",
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to get annotations: {e}")
            return None
    
    def save_annotations(self, image_id: str, annotations: List[Dict]) -> Optional[Dict]:
        """Save annotations for an image"""
        try:
            annotation_data = {
                "id": f"annotation_{image_id}",
                "bounding_boxes": annotations,
                "image_id": image_id,
                "created_at": "",
                "modified_at": ""
            }
            
            response = self._request_with_retry(
                'POST',
                f"{self.base_url}/api/annotations/{image_id}",
                json=annotation_data,
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to save annotations: {e}")
            return None
    
    def start_finetuning(self, image_id: str, annotations: List[Dict], 
                        model_name: str) -> Optional[Dict]:
        """Start model finetuning"""
        try:
            finetune_data = {
                "image_id": image_id,
                "annotation_id": f"annotation_{image_id}",
                "model_name": model_name
            }
            
            # Save annotations first
            save_result = self.save_annotations(image_id, annotations)
            if not save_result or not save_result.get('success'):
                raise Exception("Failed to save annotations")
            
            response = self._request_with_retry(
                'POST',
                f"{self.base_url}/api/finetune",
                json=finetune_data,
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to start finetuning: {e}")
            return None
    
    def get_finetuning_status(self, task_id: str) -> Optional[Dict]:
        """Get finetuning task status"""
        try:
            response = self._request_with_retry(
                'GET',
                f"{self.base_url}/api/finetune/status/{task_id}",
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to get finetuning status: {e}")
            return None
    
    def list_models(self) -> Optional[Dict]:
        """List available models"""
        try:
            response = self._request_with_retry(
                'GET',
                f"{self.base_url}/api/models",
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to list models: {e}")
            return None
    
    def switch_model(self, model_name: str) -> Optional[Dict]:
        """Switch to a different model"""
        try:
            response = self._request_with_retry(
                'POST',
                f"{self.base_url}/api/models/switch",
                json={"model_name": model_name},
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to switch model: {e}")
            return None

    def save_patch_annotations(self, patch_index: int, patch_image_path: str, bounding_boxes: List[Dict]) -> Optional[Dict]:
        """Save annotations for a specific patch"""
        try:
            annotation_data = {
                "patch_index": patch_index,
                "patch_image_path": patch_image_path,
                "bounding_boxes": bounding_boxes
            }

            response = self._request_with_retry(
                'POST',
                f"{self.base_url}/api/annotations/patch/save",
                json=annotation_data,
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to save patch annotations: {e}")
            return None

    def get_patch_annotations_count(self) -> Optional[Dict]:
        """Get count of saved patch annotations"""
        try:
            response = self._request_with_retry(
                'GET',
                f"{self.base_url}/api/annotations/patches/count",
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to get patch annotations count: {e}")
            return None

    def start_patch_finetuning(self, model_name: str, epochs: int = 10, learning_rate: float = 0.001) -> Optional[Dict]:
        """Start finetuning using accumulated patch annotations"""
        try:
            finetune_data = {
                "model_name": model_name,
                "epochs": epochs,
                "learning_rate": learning_rate
            }

            response = self._request_with_retry(
                'POST',
                f"{self.base_url}/api/finetune/patches",
                json=finetune_data,
                timeout=self.default_timeout
            )
            return response.json() if response else None
        except Exception as e:
            print(f"Failed to start patch finetuning: {e}")
            return None
    
    def get_tile(self, image_id: str, zoom: int, x: int, y: int) -> Optional[bytes]:
        """Get a specific tile from the backend"""
        try:
            response = self._request_with_retry(
                'GET',
                f"{self.base_url}/tiles/{image_id}/{zoom}/{x}/{y}.jpg",
                timeout=self.default_timeout
            )
            return response.content if response else None
        except Exception as e:
            print(f"Failed to get tile: {e}")
            return None