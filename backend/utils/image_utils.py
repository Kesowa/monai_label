import os
import json
import math
from PIL import Image, ImageOps
import rasterio
from rasterio.enums import Resampling
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TiffProcessor:
    """Handles TIFF file processing and tile generation for web display"""
    
    def __init__(self, tile_size: int = 256, max_zoom: int = 6):
        self.tile_size = tile_size
        self.max_zoom = max_zoom
        self.tiles_dir = "storage/tiles"
        os.makedirs(self.tiles_dir, exist_ok=True)
        
        # Store image information
        self.image_info: Dict[str, dict] = {}
    
    def process_tiff(self, tiff_path: str, image_id: str) -> dict:
        """
        Process a TIFF file and generate tiles for web display
        
        Args:
            tiff_path: Path to the TIFF file
            image_id: Unique identifier for the image
            
        Returns:
            Dictionary with tile information and metadata
        """
        try:
            logger.info(f"Processing TIFF file: {tiff_path}")
            
            # Create directory for this image's tiles
            image_tiles_dir = os.path.join(self.tiles_dir, image_id)
            os.makedirs(image_tiles_dir, exist_ok=True)
            
            # Read TIFF metadata
            with rasterio.open(tiff_path) as src:
                width = src.width
                height = src.height
                crs = src.crs
                transform = src.transform
                bounds = src.bounds
                band_count = src.count
                dtype = str(src.dtypes[0])
                
                # Read the image data
                if band_count >= 3:
                    # RGB or RGBA
                    data = src.read([1, 2, 3])
                else:
                    # Single band - convert to RGB
                    band = src.read(1)
                    data = [band, band, band]
                
                # Convert to PIL Image
                image = self._array_to_pil_image(data)
            
            # Generate tiles at different zoom levels
            tile_info = self._generate_tiles(image, image_tiles_dir, image_id)
            
            # Store image metadata
            metadata = {
                "width": width,
                "height": height,
                "crs": str(crs) if crs else None,
                "transform": list(transform) if transform else None,
                "bounds": bounds,
                "band_count": band_count,
                "dtype": dtype,
                "tile_info": tile_info,
                "original_path": tiff_path
            }
            
            self.image_info[image_id] = metadata
            
            # Save metadata to file
            metadata_path = os.path.join(image_tiles_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Successfully processed TIFF: {image_id}")
            return tile_info
            
        except Exception as e:
            logger.error(f"Error processing TIFF {tiff_path}: {e}")
            raise
    
    def _array_to_pil_image(self, data) -> Image.Image:
        """Convert rasterio array data to PIL Image"""
        try:
            # Transpose from (bands, height, width) to (height, width, bands)
            if len(data) == 3:
                # RGB
                rgb_array = [data[i] for i in range(3)]
                image_array = []
                for i in range(len(rgb_array[0])):
                    row = []
                    for j in range(len(rgb_array[0][0])):
                        row.append((rgb_array[0][i][j], rgb_array[1][i][j], rgb_array[2][i][j]))
                    image_array.append(row)
                
                # Create PIL image
                height = len(image_array)
                width = len(image_array[0])
                image = Image.new('RGB', (width, height))
                
                for i in range(height):
                    for j in range(width):
                        image.putpixel((j, i), image_array[i][j])
                        
            else:
                # Single band
                image = Image.fromarray(data[0], mode='L')
                image = image.convert('RGB')
            
            # Normalize the image if needed
            image = ImageOps.autocontrast(image)
            return image
            
        except Exception as e:
            logger.error(f"Error converting array to PIL image: {e}")
            # Fallback: create a simple PIL image
            return Image.new('RGB', (256, 256), color='black')
    
    def _generate_tiles(self, image: Image.Image, tiles_dir: str, image_id: str) -> dict:
        """Generate tile pyramid for web display"""
        try:
            original_width, original_height = image.size
            
            # Calculate zoom levels needed
            max_dimension = max(original_width, original_height)
            zoom_levels = min(self.max_zoom, math.ceil(math.log2(max_dimension / self.tile_size)))
            
            tile_info = {
                "zoom_levels": zoom_levels,
                "tile_size": self.tile_size,
                "original_size": [original_width, original_height],
                "tiles": {}
            }
            
            # Generate tiles for each zoom level
            for zoom in range(zoom_levels + 1):
                zoom_dir = os.path.join(tiles_dir, str(zoom))
                os.makedirs(zoom_dir, exist_ok=True)
                
                # Calculate scale for this zoom level
                scale = 2 ** (zoom_levels - zoom)
                scaled_width = max(1, original_width // scale)
                scaled_height = max(1, original_height // scale)
                
                # Resize image for this zoom level
                scaled_image = image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                
                # Calculate number of tiles needed
                tiles_x = math.ceil(scaled_width / self.tile_size)
                tiles_y = math.ceil(scaled_height / self.tile_size)
                
                tile_info["tiles"][zoom] = {
                    "tiles_x": tiles_x,
                    "tiles_y": tiles_y,
                    "scaled_size": [scaled_width, scaled_height]
                }
                
                # Generate tiles for this zoom level
                for x in range(tiles_x):
                    x_dir = os.path.join(zoom_dir, str(x))
                    os.makedirs(x_dir, exist_ok=True)
                    
                    for y in range(tiles_y):
                        # Calculate tile bounds
                        left = x * self.tile_size
                        upper = y * self.tile_size
                        right = min(left + self.tile_size, scaled_width)
                        lower = min(upper + self.tile_size, scaled_height)
                        
                        # Extract tile
                        tile = scaled_image.crop((left, upper, right, lower))
                        
                        # Pad tile to tile_size if needed
                        if tile.size != (self.tile_size, self.tile_size):
                            padded_tile = Image.new('RGB', (self.tile_size, self.tile_size), color='white')
                            padded_tile.paste(tile, (0, 0))
                            tile = padded_tile
                        
                        # Save tile
                        tile_path = os.path.join(x_dir, f"{y}.jpg")
                        tile.save(tile_path, "JPEG", quality=85)
            
            logger.info(f"Generated {zoom_levels + 1} zoom levels for image {image_id}")
            return tile_info
            
        except Exception as e:
            logger.error(f"Error generating tiles: {e}")
            raise
    
    def get_image_info(self, image_id: str) -> Optional[dict]:
        """Get information about a processed image"""
        if image_id in self.image_info:
            return self.image_info[image_id]
        
        # Try to load from metadata file
        metadata_path = os.path.join(self.tiles_dir, image_id, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                self.image_info[image_id] = metadata
                return metadata
            except Exception as e:
                logger.error(f"Error loading metadata for {image_id}: {e}")
        
        return None
    
    def get_image_path(self, image_id: str) -> Optional[str]:
        """Get the original path of an image"""
        info = self.get_image_info(image_id)
        if info:
            return info.get("original_path")
        return None
    
    def get_tile_url(self, image_id: str, zoom: int, x: int, y: int) -> str:
        """Generate URL for a specific tile"""
        return f"/tiles/{image_id}/{zoom}/{x}/{y}.jpg"
    
    def pixel_to_coordinates(self, image_id: str, pixel_x: float, pixel_y: float) -> Optional[Tuple[float, float]]:
        """Convert pixel coordinates to geographic coordinates if possible"""
        try:
            info = self.get_image_info(image_id)
            if not info or not info.get("transform") or not info.get("crs"):
                return None
            
            # Use rasterio transform
            transform = info["transform"]
            x, y = rasterio.transform.xy(
                rasterio.transform.from_gdal(*transform), 
                pixel_y, pixel_x
            )
            
            return (x, y)
            
        except Exception as e:
            logger.error(f"Error converting coordinates: {e}")
            return None
    
    def cleanup_tiles(self, image_id: str):
        """Remove tiles for a specific image"""
        try:
            tiles_path = os.path.join(self.tiles_dir, image_id)
            if os.path.exists(tiles_path):
                import shutil
                shutil.rmtree(tiles_path)
                logger.info(f"Cleaned up tiles for image {image_id}")
            
            if image_id in self.image_info:
                del self.image_info[image_id]
                
        except Exception as e:
            logger.error(f"Error cleaning up tiles for {image_id}: {e}")