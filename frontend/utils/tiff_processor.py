"""
TIFF Processor
Handles local TIFF file processing and patch extraction
Optimized for large files with aggressive memory management
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import math
import gc
import os

# Try to import psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available - memory monitoring disabled")


class TiffProcessor:
    """Handles TIFF file processing and patch extraction with memory management"""

    def __init__(self, patch_size: Tuple[int, int] = (700, 700)):
        self.patch_size = patch_size
        self.current_tiff_path = None
        self.image_data = None
        self.image_metadata = None
        self.patches_info = []
        self.memory_mode = "normal"  # "normal", "memory_efficient", "streaming"

        # Memory limits (in bytes)
        self.max_memory_for_full_load = 150 * 1024 * 1024  # 150MB
        self.min_available_memory = 500 * 1024 * 1024     # 500MB free required
        
        # Optimized caching
        self.patch_cache = {}
        self.max_cache_size = 3  # Reduced from 5
        self.cache_access_order = []
        self.current_patch_index = -1

    def check_memory_requirements(self, file_size: int, width: int, height: int, bands: int) -> str:
        """Determine the best loading strategy based on memory requirements"""
        if not HAS_PSUTIL:
            return "streaming"

        try:
            available_memory = psutil.virtual_memory().available
            bytes_per_pixel = bands * np.dtype(np.uint8).itemsize
            total_pixels = width * height
            estimated_memory = int(total_pixels * bytes_per_pixel * 1.5)  # 50% overhead

            if estimated_memory > self.max_memory_for_full_load:
                return "streaming"
            elif available_memory < self.min_available_memory:
                return "memory_efficient"
            else:
                return "normal"

        except Exception as e:
            print(f"Memory check failed: {e}, using streaming mode")
            return "streaming"

    def cleanup_memory(self, aggressive: bool = False):
        """Clean up memory - optionally aggressive cleanup"""
        if aggressive:
            # Clear everything
            self.patch_cache.clear()
            self.cache_access_order.clear()
            if self.memory_mode == "streaming" and self.image_data is not None:
                self.image_data = None
        else:
            # Keep only most recent patch
            if len(self.patch_cache) > 1:
                # Keep only current patch
                if self.current_patch_index in self.patch_cache:
                    current_patch = self.patch_cache[self.current_patch_index]
                    self.patch_cache.clear()
                    self.patch_cache[self.current_patch_index] = current_patch
                    self.cache_access_order = [self.current_patch_index]
                else:
                    self.patch_cache.clear()
                    self.cache_access_order.clear()
        
        gc.collect()

    def get_cached_patch(self, patch_index: int) -> Optional[np.ndarray]:
        """Get a patch from cache if available"""
        if patch_index in self.patch_cache:
            # Update LRU order
            if patch_index in self.cache_access_order:
                self.cache_access_order.remove(patch_index)
            self.cache_access_order.append(patch_index)
            return self.patch_cache[patch_index]
        return None

    def cache_patch(self, patch_index: int, patch_data: np.ndarray):
        """Cache a patch using LRU strategy"""
        # Remove oldest if cache is full
        while len(self.patch_cache) >= self.max_cache_size:
            oldest_patch = self.cache_access_order.pop(0)
            if oldest_patch in self.patch_cache:
                del self.patch_cache[oldest_patch]

        # Add new patch
        self.patch_cache[patch_index] = patch_data.copy()
        if patch_index not in self.cache_access_order:
            self.cache_access_order.append(patch_index)

    def load_tiff(self, tiff_path: str) -> bool:
        """Load a TIFF file and prepare patch information with memory management"""
        try:
            # Clean up previous data
            self.cleanup_memory(aggressive=True)

            tiff_path = Path(tiff_path)
            if not tiff_path.exists():
                raise FileNotFoundError(f"TIFF file not found: {tiff_path}")

            file_size = tiff_path.stat().st_size
            self.current_tiff_path = str(tiff_path)

            # Read TIFF metadata
            with rasterio.open(self.current_tiff_path) as src:
                self.image_metadata = {
                    'width': src.width,
                    'height': src.height,
                    'count': src.count,
                    'dtype': src.dtypes[0],
                    'crs': src.crs,
                    'transform': src.transform,
                    'bounds': src.bounds,
                    'file_size': file_size
                }

                # Determine loading strategy
                self.memory_mode = self.check_memory_requirements(
                    file_size, src.width, src.height, src.count
                )

                print(f"Loading mode: {self.memory_mode}")
                print(f"File: {file_size / (1024*1024):.1f}MB, Size: {src.width}x{src.height}")

                if self.memory_mode == "normal":
                    # Safe to load full image
                    try:
                        if src.count >= 3:
                            self.image_data = src.read([1, 2, 3])
                        else:
                            band_data = src.read(1)
                            self.image_data = np.stack([band_data, band_data, band_data])

                        self.image_data = self.image_data.transpose(1, 2, 0)
                        print("Full image loaded into memory")
                    except MemoryError:
                        print("Memory error during full load, switching to streaming mode")
                        self.memory_mode = "streaming"
                        self.image_data = None
                else:
                    self.image_data = None
                    print("Using on-demand patch loading")

            # Calculate patch information
            self._calculate_patches()

            print(f"TIFF loaded: {len(self.patches_info)} patches")
            return True

        except MemoryError:
            print("Out of memory - file too large")
            self.cleanup_memory(aggressive=True)
            return False
        except Exception as e:
            print(f"Error loading TIFF: {e}")
            self.cleanup_memory(aggressive=True)
            return False
    
    def _calculate_patches(self):
        """Calculate patch positions and information"""
        if not self.image_metadata:
            return

        width = self.image_metadata['width']
        height = self.image_metadata['height']
        patch_width, patch_height = self.patch_size

        self.patches_info = []
        patch_id = 0

        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                x1, y1 = x, y
                x2 = min(x + patch_width, width)
                y2 = min(y + patch_height, height)

                patch_info = {
                    'id': patch_id,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2
                }

                self.patches_info.append(patch_info)
                patch_id += 1
    
    def get_patch_info(self) -> Dict:
        """Get information about patches"""
        return {
            'total_patches': len(self.patches_info),
            'patch_size': self.patch_size,
            'image_size': (self.image_metadata['width'], self.image_metadata['height']) if self.image_metadata else (0, 0),
            'patches_per_row': math.ceil(self.image_metadata['width'] / self.patch_size[0]) if self.image_metadata else 0,
            'patches_per_col': math.ceil(self.image_metadata['height'] / self.patch_size[1]) if self.image_metadata else 0
        }
    
    def get_patch(self, patch_index: int) -> Optional[Dict]:
        """Get a specific patch by index with caching and memory management"""
        if patch_index < 0 or patch_index >= len(self.patches_info):
            return None

        # Aggressive cleanup if jumping to distant patch
        if self.current_patch_index >= 0 and abs(patch_index - self.current_patch_index) > 2:
            self.cleanup_memory(aggressive=False)

        self.current_patch_index = patch_index
        patch_info = self.patches_info[patch_index]

        # Check cache first
        cached_patch = self.get_cached_patch(patch_index)
        if cached_patch is not None:
            return {
                'image': cached_patch,
                'info': patch_info,
                'patch_index': patch_index
            }

        try:
            if self.image_data is not None:
                # Extract from loaded image
                patch_image = self.image_data[
                    patch_info['ymin']:patch_info['ymax'],
                    patch_info['xmin']:patch_info['xmax']
                ].copy()
            else:
                # Read patch directly from file
                patch_image = self._read_patch_from_file(patch_info)

            # Ensure correct format
            if len(patch_image.shape) == 3 and patch_image.shape[2] == 4:
                patch_image = patch_image[:, :, :3]
            elif len(patch_image.shape) == 2:
                patch_image = np.stack([patch_image, patch_image, patch_image], axis=2)

            # Ensure uint8
            if patch_image.dtype != np.uint8:
                if patch_image.max() <= 1.0:
                    patch_image = (patch_image * 255).astype(np.uint8)
                else:
                    patch_image = patch_image.astype(np.uint8)

            # Cache the patch
            self.cache_patch(patch_index, patch_image)

            return {
                'image': patch_image,
                'info': patch_info,
                'patch_index': patch_index
            }

        except MemoryError:
            print(f"Memory error loading patch {patch_index}")
            self.cleanup_memory(aggressive=True)
            return None
        except Exception as e:
            print(f"Error extracting patch {patch_index}: {e}")
            return None
    
    def _read_patch_from_file(self, patch_info: Dict) -> np.ndarray:
        """Read a patch directly from the TIFF file"""
        try:
            with rasterio.open(self.current_tiff_path) as src:
                window = Window(
                    patch_info['xmin'],
                    patch_info['ymin'],
                    patch_info['width'],
                    patch_info['height']
                )

                # Memory check
                if HAS_PSUTIL:
                    patch_pixels = patch_info['width'] * patch_info['height']
                    estimated_bytes = patch_pixels * src.count * 2
                    available_memory = psutil.virtual_memory().available
                    
                    if estimated_bytes > available_memory * 0.8:
                        raise MemoryError(f"Insufficient memory for patch")

                # Read the patch
                if src.count >= 3:
                    patch_data = src.read([1, 2, 3], window=window)
                else:
                    band_data = src.read(1, window=window)
                    patch_data = np.stack([band_data, band_data, band_data])

                result = patch_data.transpose(1, 2, 0)
                del patch_data
                gc.collect()

                return result

        except Exception as e:
            print(f"Error reading patch from file: {e}")
            raise
    
    def get_patch_bounds(self, patch_index: int) -> Optional[Dict]:
        """Get the bounds of a specific patch"""
        if patch_index < 0 or patch_index >= len(self.patches_info):
            return None
        return self.patches_info[patch_index].copy()
    
    def find_patch_at_coordinates(self, x: int, y: int) -> int:
        """Find which patch contains the given coordinates"""
        for i, patch in enumerate(self.patches_info):
            if (patch['xmin'] <= x < patch['xmax'] and 
                patch['ymin'] <= y < patch['ymax']):
                return i
        return -1
    
    def get_overlapping_patches(self, bbox: Dict) -> List[int]:
        """Get all patches that overlap with a bounding box"""
        overlapping = []
        for i, patch in enumerate(self.patches_info):
            if self._bbox_overlaps_patch(bbox, patch):
                overlapping.append(i)
        return overlapping
    
    def _bbox_overlaps_patch(self, bbox: Dict, patch: Dict) -> bool:
        """Check if a bounding box overlaps with a patch"""
        return not (bbox['xmax'] <= patch['xmin'] or bbox['xmin'] >= patch['xmax'] or
                   bbox['ymax'] <= patch['ymin'] or bbox['ymin'] >= patch['ymax'])
    
    def convert_global_to_patch_coords(self, global_bbox: Dict, patch_index: int) -> Optional[Dict]:
        """Convert global coordinates to patch-local coordinates"""
        if patch_index < 0 or patch_index >= len(self.patches_info):
            return None
            
        patch = self.patches_info[patch_index]
        
        if not self._bbox_overlaps_patch(global_bbox, patch):
            return None
        
        local_bbox = {
            'xmin': max(0, global_bbox['xmin'] - patch['xmin']),
            'ymin': max(0, global_bbox['ymin'] - patch['ymin']),
            'xmax': min(patch['width'], global_bbox['xmax'] - patch['xmin']),
            'ymax': min(patch['height'], global_bbox['ymax'] - patch['ymin']),
            'confidence': global_bbox.get('confidence'),
            'label': global_bbox.get('label', 'Tree')
        }
        
        return local_bbox
    
    def convert_patch_to_global_coords(self, patch_bbox: Dict, patch_index: int) -> Optional[Dict]:
        """Convert patch-local coordinates to global coordinates"""
        if patch_index < 0 or patch_index >= len(self.patches_info):
            return None
            
        patch = self.patches_info[patch_index]
        
        global_bbox = {
            'xmin': patch_bbox['xmin'] + patch['xmin'],
            'ymin': patch_bbox['ymin'] + patch['ymin'],
            'xmax': patch_bbox['xmax'] + patch['xmin'],
            'ymax': patch_bbox['ymax'] + patch['ymin'],
            'confidence': patch_bbox.get('confidence'),
            'label': patch_bbox.get('label', 'Tree')
        }
        
        return global_bbox
    
    def get_image_metadata(self) -> Optional[Dict]:
        """Get image metadata"""
        return self.image_metadata.copy() if self.image_metadata else None
    
    def is_loaded(self) -> bool:
        """Check if a TIFF file is currently loaded"""
        return self.current_tiff_path is not None and len(self.patches_info) > 0
    
    def save_patch_to_file(self, patch_index: int, output_path: str) -> bool:
        """
        Save a specific patch to a file for uploading to backend
        
        Args:
            patch_index: Index of the patch to save
            output_path: Path where to save the patch file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            patch_data = self.get_patch(patch_index)
            if not patch_data:
                return False
            
            patch_image = patch_data['image']
            
            # Save as TIFF using rasterio to maintain quality
            from rasterio.transform import from_bounds
            import rasterio
            
            patch_info = patch_data['info']
            height, width = patch_image.shape[:2]
            
            # Prepare data in rasterio format (bands, height, width)
            if len(patch_image.shape) == 3:
                data_to_save = patch_image.transpose(2, 0, 1)
            else:
                data_to_save = patch_image.reshape(1, height, width)
            
            # Write the patch
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=data_to_save.shape[0],
                dtype=patch_image.dtype,
                compress='deflate'
            ) as dst:
                dst.write(data_to_save)
            
            return True
            
        except Exception as e:
            print(f"Error saving patch to file: {e}")
            return False

    def get_memory_info(self) -> Dict:
        """Get memory usage information"""
        if not HAS_PSUTIL:
            return {
                'total_memory': 0,
                'available_memory': 0,
                'used_memory': 0,
                'memory_percent': 0,
                'cache_size': len(self.patch_cache),
                'memory_mode': self.memory_mode,
                'psutil_available': False
            }

        try:
            vm = psutil.virtual_memory()
            return {
                'total_memory': vm.total,
                'available_memory': vm.available,
                'used_memory': vm.used,
                'memory_percent': vm.percent,
                'cache_size': len(self.patch_cache),
                'memory_mode': self.memory_mode,
                'psutil_available': True
            }
        except Exception:
            return {
                'total_memory': 0,
                'available_memory': 0,
                'used_memory': 0,
                'memory_percent': 0,
                'cache_size': len(self.patch_cache),
                'memory_mode': self.memory_mode,
                'psutil_available': False
            }