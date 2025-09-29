import os
import shutil
from flask import Flask, request, render_template, jsonify, send_from_directory, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import uuid
import time
import cv2
import gc
import torch
import logging
from scipy.interpolate import splprep, splev
import json
from scipy import ndimage
import zipfile
import trimesh
# from skimage import filters, morphology, measure  # Temporarily disabled due to numpy compatibility issues
# from skimage.transform import resize  # Temporarily disabled due to numpy compatibility issues
import base64
import time as _time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['SEGMENTATION_FOLDER'] = 'static/segmentations'
app.config['EXTRACTED_FOLDER'] = 'static/extracted'
app.config['RESULTS_FOLDER'] = 'static/results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max upload
app.config['MAX_IMAGE_SIZE'] = (2560, 2560)
app.config['USER_MODELS_FOLDER'] = 'user_models'

# Create all necessary folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SEGMENTATION_FOLDER'], exist_ok=True)
os.makedirs(app.config['EXTRACTED_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs(app.config['USER_MODELS_FOLDER'], exist_ok=True)

_last_inference_ts = 0.0
_inference_min_interval_sec = 0.3

# Load YOLOv11 model with segmentation capability - load when needed
model = None

def get_available_base_models():
    """Return list of available base nail models from static/3D_nails."""
    base_dir = os.path.join('static', '3D_nails')
    supported_exts = {'.obj', '.stl', '.glb'}
    try:
        if not os.path.isdir(base_dir):
            return []
        files = [f for f in os.listdir(base_dir) if os.path.splitext(f)[1].lower() in supported_exts]
        files.sort()
        return files
    except Exception as e:
        logger.error(f"Error listing base models: {e}")
        return []

def generate_consistent_planar_uv(mesh: trimesh.Trimesh) -> np.ndarray:
    """Generate stable UVs by projecting vertices onto the best-fit plane
    and aligning axes by principal components. Returns Nx2 array in [0,1]."""
    try:
        v = mesh.vertices.astype(np.float64)
        # Center
        centroid = v.mean(axis=0)
        vc = v - centroid
        # PCA for orientation
        cov = np.cov(vc.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort by largest variances descending
        order = np.argsort(-eigvals)
        axes = eigvecs[:, order]
        # Normal is smallest variance direction
        normal = axes[:, -1]
        # Build tangent and bitangent from the other two axes
        tangent = axes[:, 0]
        bitangent = axes[:, 1]
        # Project to plane coords
        u = vc.dot(tangent)
        w = vc.dot(bitangent)
        u_min, u_max = float(u.min()), float(u.max())
        w_min, w_max = float(w.min()), float(w.max())
        u_span = max(u_max - u_min, 1e-6)
        w_span = max(w_max - w_min, 1e-6)
        u_norm = (u - u_min) / u_span
        w_norm = (w - w_min) / w_span  # No flip since image is rotated 180 degrees
        uv = np.column_stack([u_norm, w_norm]).astype(np.float32)
        return uv
    except Exception:
        # Fallback to bbox projection
        v = mesh.vertices
        minv = v.min(axis=0); maxv = v.max(axis=0)
        span = np.maximum(maxv - minv, 1e-6)
        u = (v[:, 0] - minv[0]) / span[0]
        w = (v[:, 1] - minv[1]) / span[1]  # No flip since image is rotated 180 degrees
        return np.column_stack([u, w]).astype(np.float32)

def get_model():
    global model
    if model is None:
        from ultralytics import YOLO
        import os
        
        # Check if model path is specified via environment variable
        model_path = os.environ.get('MODEL_PATH', 'model/best_10k.pt')
        
        # Check if model is in S3
        if model_path.startswith('s3://'):
            try:
                import boto3
                
                # Parse S3 path
                bucket_name = model_path.split('/')[2]
                object_key = '/'.join(model_path.split('/')[3:])
                
                # Download from S3
                local_model_path = 'model/best.pt'
                os.makedirs('model', exist_ok=True)
                logger.info(f"Downloading model from S3: {bucket_name}/{object_key}")
                
                s3 = boto3.client('s3')
                s3.download_file(bucket_name, object_key, local_model_path)
                logger.info(f"Model downloaded successfully to {local_model_path}")
                
                model = YOLO(local_model_path)
            except Exception as e:
                logger.error(f"Error loading model from S3: {e}")
                # Fallback to local model if exists
                if os.path.exists('model/best.pt'):
                    logger.info("Falling back to local model")
                    model = YOLO('model/best.pt')
                else:
                    raise RuntimeError("Failed to load model from S3 and no local model found")
        else:
            # Load local model
            logger.info(f"Loading model from local path: {model_path}")
            model = YOLO(model_path)
            
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def resize_image_if_needed(image_path):
    """Resize large images while preserving quality"""
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            max_w, max_h = app.config['MAX_IMAGE_SIZE']
            
            if width > max_w or height > max_h:
                # Calculate new dimensions while preserving aspect ratio
                ratio = min(max_w/width, max_h/height)
                new_size = (int(width * ratio), int(height * ratio))
                
                # Use high quality resampling for better details
                img = img.resize(new_size, Image.LANCZOS)
                
                # Save the resized image with high quality
                img.save(image_path, quality=95)
                logger.info(f"Resized image from {width}x{height} to {new_size[0]}x{new_size[1]} with high quality")
    except Exception as e:
        logger.error(f"Error resizing image: {e}")

def process_nail_mask(mask_bool, orig_img):
    """Maximum smoothness nail edge processing with resolution preservation"""
    # Convert boolean mask to uint8 for OpenCV operations
    mask_uint8 = mask_bool.astype(np.uint8) * 255
    
    # Find contours from the mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # If no contours found, return original mask
    if not contours:
        return mask_bool
    
    # Get the largest contour (should be the nail)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a blank canvas
    refined_mask = np.zeros_like(mask_uint8)
    h, w = refined_mask.shape[:2]
    
    # Use 4x super-resolution canvas for maximum smoothness
    hires_canvas = np.zeros((h*4, w*4), dtype=np.uint8)
    
    # Apply advanced curve smoothing with extreme smoothness priority
    contour_points = largest_contour.reshape(-1, 2)
    
    if len(contour_points) > 5:
        try:
            # Make the contour cyclic with even more wraparound points
            contour_points = np.append(contour_points, contour_points[:10], axis=0)
            
            # Apply maximum smoothing for ultra-smooth curves
            tck, u = splprep([contour_points[:, 0], contour_points[:, 1]], s=len(contour_points)*10, per=1)
            
            # Generate extremely high density of points
            u_new = np.linspace(0, 1, num=4000)
            smoothed_points = np.column_stack(splev(u_new, tck))
            
            # Scale up points for 4x resolution canvas
            hires_points = smoothed_points * 4
            
            # Convert back to format for drawing
            smoothed_contour = hires_points.astype(np.int32).reshape(-1, 1, 2)
            
            # Draw the smoothed contour on high-res canvas
            cv2.drawContours(hires_canvas, [smoothed_contour], 0, 255, -1)
            
            # Multi-stage edge smoothing at high resolution
            
            # 1. First round of Gaussian blur (very strong)
            hires_smooth = cv2.GaussianBlur(hires_canvas, (51, 51), 12.0)
            
            # 2. Use large elliptical kernel for organic shapes
            ellip_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            hires_smooth = cv2.morphologyEx(hires_smooth, cv2.MORPH_CLOSE, ellip_kernel, iterations=3)
            
            # 3. Second round of blur for perfect smoothness
            hires_smooth = cv2.GaussianBlur(hires_smooth, (31, 31), 8.0)
            
            # Resize back to original size with high-quality interpolation
            refined_mask = cv2.resize(hires_smooth, (w, h), interpolation=cv2.INTER_LANCZOS4)
            
        except Exception as e:
            # Fallback to extreme Gaussian smoothing
            logger.warning(f"Ultra-smooth spline failed: {e}")
            cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
    
            # Extreme multi-stage blurring for fallback
            refined_mask = cv2.GaussianBlur(refined_mask, (51, 51), 15.0)
            refined_mask = cv2.GaussianBlur(refined_mask, (31, 31), 8.0)
    else:
        # For small contours, apply extreme smoothing
        cv2.drawContours(refined_mask, [largest_contour], 0, 255, -1)
        refined_mask = cv2.GaussianBlur(refined_mask, (51, 51), 12.0)
    
    # Convert back to binary with a threshold
    final_mask = cv2.threshold(refined_mask, 120, 255, cv2.THRESH_BINARY)[1]
    
    # One final light blur to ensure perfect edges
    final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0.8)
    final_mask = cv2.threshold(final_mask, 200, 255, cv2.THRESH_BINARY)[1]
    
    # Convert back to boolean
    return final_mask > 0

def cleanup_memory():
    """Free up GPU/CUDA memory and perform garbage collection"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Memory cleanup performed")


def _b64_to_image(b64_data: str):
    """Decode a dataURL or raw base64 image (PNG/JPEG) to numpy BGR image."""
    try:
        if ',' in b64_data:
            b64_data = b64_data.split(',', 1)[1]
        data = base64.b64decode(b64_data)
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {e}")
        return None

def generate_pbr_maps(segmented_image, base_filename):
    """
    Generate all 7 PBR material maps based on the specified parameters:
    - Micro Details: 1
    - Medium Details: 1  
    - Large Details: 0.5
    - AO Strength: 1
    - Material Type: Shiny
    - Roughness Base Value: 0.1
    - Variations: 0.5
    - Albedo Importance: 0
    - Delighting Intensity: 1
    """
    
    # Extract RGB and alpha channels
    rgb = segmented_image[:, :, :3]
    alpha = segmented_image[:, :, 3]
    
    # Create mask for non-transparent pixels
    mask = alpha > 128
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Get image dimensions
    h, w = gray.shape
    
    # Calculate center coordinates for curvature effects
    center_y, center_x = h // 2, w // 2
    
    # Create distance from center matrix
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
    max_dist = np.sqrt(center_y**2 + center_x**2)
    
    # Normalize distance
    center_weight = 1 - np.clip(dist_from_center / max_dist, 0, 1)
    
    # 1. DIFFUSE MAP (Base Color)
    diffuse_map = np.zeros((h, w, 4), dtype=np.uint8)
    # Use original colors but clear outside the mask to avoid background
    diffuse_map[..., :3] = 0
    if np.any(mask):
        diffuse_map[mask, :3] = rgb[mask]
    diffuse_map[..., 3] = alpha  # Preserve alpha
    
    # Preserve original color (disable delighting by default)
    # Set delighting_intensity > 0 if you want to attenuate lighting
    if np.any(mask):
        delighting_intensity = 0.0
        if delighting_intensity > 0:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0].astype(np.float32)
            l_smooth = cv2.GaussianBlur(l_channel, (21, 21), 5.0)
            l_delighted = l_channel * (1 - delighting_intensity) + l_smooth * delighting_intensity
            lab[:, :, 0] = np.clip(l_delighted, 0, 255).astype(np.uint8)
            rgb_delighted = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            diffuse_map[mask, :3] = rgb_delighted[mask]
    
    # 2. ROUGHNESS MAP (Inverted grayscale with low base value for shiny effect)
    roughness_map = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Start with low base roughness (0.1) for shiny effect
    roughness_base = 0.1
    
    # Create roughness variations based on image features
    roughness = np.ones_like(gray, dtype=np.float32) * roughness_base
    
    if np.any(mask):
        # Add variations based on brightness (darker areas = rougher)
        gray_norm = gray.astype(np.float32) / 255.0
        roughness[mask] += (1.0 - gray_norm[mask]) * 0.3
        
        # Add center-based variation (center is smoother)
        roughness[mask] += center_weight[mask] * -0.2
        
        # Add micro-detail variations
        micro_details = cv2.GaussianBlur(gray_norm, (3, 3), 0.5)
        roughness[mask] += micro_details[mask] * 0.1
        
        # Add medium-detail variations
        medium_details = cv2.GaussianBlur(gray_norm, (9, 9), 1.8)
        roughness[mask] += medium_details[mask] * 0.1
        
        # Add large-detail variations (reduced by 0.5 factor)
        large_details = cv2.GaussianBlur(gray_norm, (21, 21), 4.0)
        roughness[mask] += large_details[mask] * 0.05
        
        # Add edge roughness
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges > 0
        roughness[edge_mask] += 0.1
        
        # Clamp to reasonable range
        roughness = np.clip(roughness, 0.05, 0.8)
    
    # Convert to 8-bit and invert for shiny effect (lower values = shinier)
    roughness_inverted = 1.0 - roughness
    roughness_map[..., 0] = (roughness_inverted * 255).astype(np.uint8)
    roughness_map[..., 1] = roughness_map[..., 0]
    roughness_map[..., 2] = roughness_map[..., 0]
    roughness_map[..., 3] = alpha
    
    # 3. HEIGHT MAP (Enhanced grayscale with multi-scale detail enhancement)
    height_map = np.zeros((h, w, 4), dtype=np.uint8)
    
    if np.any(mask):
        # Start with grayscale as base height
        height_data = gray.astype(np.float32) / 255.0
        
        # Apply multi-scale detail enhancement
        # Micro details (factor 1.0)
        micro_enhanced = cv2.GaussianBlur(height_data, (3, 3), 0.5)
        height_data = height_data * 0.7 + micro_enhanced * 0.3
        
        # Medium details (factor 1.0)
        medium_enhanced = cv2.GaussianBlur(height_data, (9, 9), 1.8)
        height_data = height_data * 0.7 + medium_enhanced * 0.3
        
        # Large details (factor 0.5)
        large_enhanced = cv2.GaussianBlur(height_data, (21, 21), 4.0)
        height_data = height_data * 0.8 + large_enhanced * 0.2
        
        # Add center curvature for nail-like shape
        height_data[mask] += center_weight[mask] * 0.2
        
        # Add subtle noise for texture
        noise = np.random.normal(0, 0.02, height_data.shape).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (3, 3), 0.5)
        height_data[mask] += noise[mask]
        
        # Normalize and clamp
        height_data = np.clip(height_data, 0, 1)
        
        # INVERT if "pushed in" to make "pushed out"
        height_data = 1.0 - height_data
    
    # Convert to 8-bit
    height_map[..., 0] = (height_data * 255).astype(np.uint8)
    height_map[..., 1] = height_map[..., 0]
    height_map[..., 2] = height_map[..., 0]
    height_map[..., 3] = alpha
    
    # 4. NORMAL MAP (Generated from height map using Sobel gradients)
    normal_map = np.zeros((h, w, 4), dtype=np.uint8)
    
    if np.any(mask):
        # Calculate gradients using Sobel operators
        dx = cv2.Sobel(height_data, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(height_data, cv2.CV_32F, 0, 1, ksize=3)
        
        # Normalize gradients
        dx = dx / (np.max(np.abs(dx)) + 1e-8)
        dy = dy / (np.max(np.abs(dy)) + 1e-8)
        
        # Convert to normal map format (R=X, G=Y, B=Z)
        # Normal maps use specific encoding: R=128+dx*127, G=128+dy*127, B=128
        normal_map[..., 0] = np.clip(128 + dx * 127, 0, 255).astype(np.uint8)  # X component
        normal_map[..., 1] = np.clip(128 + dy * 127, 0, 255).astype(np.uint8)  # Y component
        normal_map[..., 2] = 128  # Z component (constant)
        normal_map[..., 3] = alpha
    
    # 5. METALLIC MAP (Pure black for nails - not metallic)
    metallic_map = np.zeros((h, w, 4), dtype=np.uint8)
    metallic_map[..., 3] = alpha  # Preserve alpha
    
    # 6. SPECULAR MAP (Mid-gray for some shine)
    specular_map = np.zeros((h, w, 4), dtype=np.uint8)
    
    if np.any(mask):
        # Base specular value (mid-gray)
        specular_value = 128
        
        # Add variations based on brightness
        specular_data = np.ones_like(gray, dtype=np.uint8) * specular_value
        
        # Brighter areas get higher specular
        bright_mask = gray > 128
        specular_data[bright_mask] = 180
        
        # Darker areas get lower specular
        dark_mask = gray < 64
        specular_data[dark_mask] = 80
        
        specular_map[..., 0] = specular_data
        specular_map[..., 1] = specular_data
        specular_map[..., 2] = specular_data
        specular_map[..., 3] = alpha
    
    # 7. AO MAP (Ambient occlusion simulation using thresholding and edge detection)
    ao_map = np.zeros((h, w, 4), dtype=np.uint8)
    
    if np.any(mask):
        # Start with bright base (nails don't have deep crevices)
        ao_data = np.ones_like(gray, dtype=np.uint8) * 220
        
        # Find edges for occlusion
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges > 0
        
        # Darken edges for AO effect
        ao_data[edge_mask] = 150
        
        # Add distance-based AO from edges
        edge_distance = ndimage.distance_transform_edt(~edge_mask)
        edge_distance = np.clip(edge_distance / 10, 0, 1)
        
        # Apply AO falloff
        ao_falloff = 220 - (edge_distance * 50).astype(np.uint8)
        ao_data = np.minimum(ao_data, ao_falloff)
        
        # Apply AO strength parameter
        ao_strength = 1.0
        ao_data = np.clip(ao_data * ao_strength, 100, 255)
        
        ao_map[..., 0] = ao_data
        ao_map[..., 1] = ao_data
        ao_map[..., 2] = ao_data
        ao_map[..., 3] = alpha
    
    # Save all maps
    maps = {}
    
    # Save diffuse map
    diffuse_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_diffuse.png")
    Image.fromarray(diffuse_map).save(diffuse_path)
    maps['diffuse'] = diffuse_path
    
    # Save roughness map
    roughness_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_roughness.png")
    Image.fromarray(roughness_map).save(roughness_path)
    maps['roughness'] = roughness_path
    
    # Save height map
    height_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_height.png")
    Image.fromarray(height_map).save(height_path)
    maps['height'] = height_path
    
    # Save normal map
    normal_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_normal.png")
    Image.fromarray(normal_map).save(normal_path)
    maps['normal'] = normal_path
    
    # Save metallic map
    metallic_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_metallic.png")
    Image.fromarray(metallic_map).save(metallic_path)
    maps['metallic'] = metallic_path
    
    # Save specular map
    specular_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_specular.png")
    Image.fromarray(specular_map).save(specular_path)
    maps['specular'] = specular_path
    
    # Save AO map
    ao_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_ao.png")
    Image.fromarray(ao_map).save(ao_path)
    maps['ao'] = ao_path
    
    return maps

def create_heightfield_model(height_map_path, base_filename):
    """
    Create a 3D nail directly from the segmented image height map.
    - Uses height intensity as displacement in +Z
    - Respects alpha: faces are only generated where the mask exists
    - Produces OBJ, STL and GLB
    """
    try:
        # Load height map and extract single channel + alpha
        height_img = Image.open(height_map_path).convert('RGBA')
        height_np = np.array(height_img)
        gray = height_np[:, :, 0].astype(np.float32) / 255.0
        if height_np.shape[2] > 3:
            alpha_mask = height_np[:, :, 3] > 127
        else:
            alpha_mask = np.ones_like(gray, dtype=bool)

        h, w = gray.shape

        # Optional smoothing for cleaner surface
        gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0.8)

        # Thickness scale relative to image size
        thickness = 0.08

        # Build vertex grid in [0,1]
        yy, xx = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w), indexing='ij')
        zz = gray_smooth * thickness
        vertices = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

        # Build faces only where alpha present for the whole quad
        faces = []
        for y in range(h - 1):
            for x in range(w - 1):
                if alpha_mask[y, x] and alpha_mask[y, x + 1] and alpha_mask[y + 1, x] and alpha_mask[y + 1, x + 1]:
                    v1 = y * w + x
                    v2 = y * w + (x + 1)
                    v3 = (y + 1) * w + x
                    v4 = (y + 1) * w + (x + 1)
                    faces.append([v1, v2, v3])
                    faces.append([v2, v4, v3])

        mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces, dtype=np.int64))

        # Export files
        obj_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_3d_model.obj")
        mesh.export(obj_path)
        stl_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_3d_model.stl")
        mesh.export(stl_path)
        glb_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_3d_model.glb")
        try:
            mesh.export(glb_path)
        except Exception:
            glb_path = None

        return {
            'obj': obj_path,
            'stl': stl_path,
            'glb': glb_path,
            'mtl': None
        }
    except Exception as e:
        logger.error(f"Error creating heightfield 3D model: {e}")
        return None

def create_3d_model(height_map_path, base_filename, selected_base_model=None, diffuse_map_path=None):
    """
    Create a 3D model by applying height map displacement to a base nail model
    """
    try:
        # Load the height map and alpha (mask)
        height_img = Image.open(height_map_path).convert('RGBA')
        height_np = np.array(height_img)
        height_array = height_np[:, :, 0]
        alpha_array = height_np[:, :, 3] if height_np.shape[2] > 3 else np.full_like(height_array, 255)
        h, w = height_array.shape
        
        # Attempt to load selected base mesh from static/3D_nails
        mesh = None
        if selected_base_model:
            try:
                base_path = os.path.join('static', '3D_nails', selected_base_model)
                if os.path.exists(base_path):
                    loaded = trimesh.load(base_path, force='mesh')
                    if isinstance(loaded, trimesh.Scene):
                        geoms = [g for g in loaded.geometry.values()]
                        if len(geoms) > 0:
                            mesh = trimesh.util.concatenate(geoms)
                    elif isinstance(loaded, trimesh.Trimesh):
                        mesh = loaded
            except Exception as e:
                logger.warning(f"Base model load failed ({selected_base_model}): {e}")

        # Fallback: generate procedural nail base
        if mesh is None:
            vertices = []
            faces = []
            for y in range(h):
                for x in range(w):
                    center_x, center_y = w // 2, h // 2
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_dist = np.sqrt(center_x**2 + center_y**2)
                    curvature = 1.0 - np.clip(dist_from_center / max_dist, 0, 1)
                    curvature = curvature ** 2
                    base_z = curvature * 0.1
                    height_displacement = (height_array[y, x] / 255.0) * 0.05
                    z = base_z + height_displacement
                    vertices.append([x / w, y / h, z])
            for y in range(h - 1):
                for x in range(w - 1):
                    v1 = y * w + x
                    v2 = y * w + (x + 1)
                    v3 = (y + 1) * w + x
                    v4 = (y + 1) * w + (x + 1)
                    faces.append([v1, v2, v3])
                    faces.append([v2, v4, v3])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Displace vertices along normals using height map via UVs if present
        try:
            vertices = mesh.vertices.copy()
            normals = mesh.vertex_normals.copy()
            uv = None
            if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
                uv = mesh.visual.uv

            height_f = height_array.astype(np.float32) / 255.0
            alpha_f = cv2.GaussianBlur((alpha_array.astype(np.float32) / 255.0), (5, 5), 0.8)

            def bilinear_sample(img, u, v):
                x = np.clip(u, 0.0, 1.0) * (w - 1)
                y = np.clip(v, 0.0, 1.0) * (h - 1)  # No flip since image is rotated 180 degrees
                x0 = np.floor(x).astype(np.int32); x1 = np.clip(x0 + 1, 0, w - 1)
                y0 = np.floor(y).astype(np.int32); y1 = np.clip(y0 + 1, 0, h - 1)
                wx = (x - x0).astype(np.float32); wy = (y - y0).astype(np.float32)
                Ia = img[y0, x0]; Ib = img[y0, x1]
                Ic = img[y1, x0]; Id = img[y1, x1]
                top = Ia * (1 - wx) + Ib * wx
                bottom = Ic * (1 - wx) + Id * wx
                return top * (1 - wy) + bottom * wy

            if uv is not None and len(uv) == len(vertices):
                # Use UV coordinates directly since image is rotated 180 degrees
                u = uv[:, 0]
                v = uv[:, 1]  # No flip since image is rotated 180 degrees
                disp_raw = bilinear_sample(height_f, u, v)
                mask_w = bilinear_sample(alpha_f, u, v)
            else:
                minv = vertices.min(axis=0); maxv = vertices.max(axis=0)
                span = np.maximum(maxv - minv, 1e-6)
                u_proj = (vertices[:, 0] - minv[0]) / span[0]
                v_proj = (vertices[:, 1] - minv[1]) / span[1]  # No flip since image is rotated 180 degrees
                disp_raw = bilinear_sample(height_f, u_proj, v_proj)
                mask_w = bilinear_sample(alpha_f, u_proj, v_proj)

            disp = disp_raw * np.clip(mask_w, 0.0, 1.0)

            # Scale displacement relative to mesh diagonal, and keep subtle
            bbox = mesh.bounds
            diag = float(np.linalg.norm(bbox[1] - bbox[0]))
            displacement_scale = max(diag, 1e-6) * 0.01
            mesh.vertices = vertices + normals * disp.reshape(-1, 1) * displacement_scale
        except Exception as e:
            logger.warning(f"UV displacement failed: {e}")

        # Apply diffuse texture using only the segmented nail, scaled to cover the whole nail surface
        try:
            if diffuse_map_path and os.path.exists(diffuse_map_path):
                from trimesh.visual.texture import TextureVisuals
                # Load segmented RGBA
                tex_rgba = Image.open(diffuse_map_path).convert('RGBA')
                tex_np = np.array(tex_rgba)
                H, W = tex_np.shape[:2]
                a = tex_np[:, :, 3]
                # Find tight bounding box around the nail
                ys, xs = np.where(a > 127)
                if len(xs) > 0 and len(ys) > 0:
                    x1, x2 = int(xs.min()), int(xs.max())
                    y1, y2 = int(ys.min()), int(ys.max())
                    crop_rgb = tex_np[y1:y2+1, x1:x2+1, :3]
                    # Resize crop to fill the whole texture canvas
                    crop_resized = cv2.resize(crop_rgb, (W, H), interpolation=cv2.INTER_CUBIC)
                    full_rgb = crop_resized
                else:
                    # Fallback to original RGB
                    full_rgb = tex_np[:, :, :3]
                # Use full coverage (opaque) so the entire nail surface is textured
                tex_img = Image.fromarray(full_rgb, mode='RGB')

                # Ensure we have UVs; if not, create planar UVs aligned to dominant axes
                uv = None
                if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None and len(mesh.visual.uv) == len(mesh.vertices):
                    uv = mesh.visual.uv
                if uv is None:
                    uv = generate_consistent_planar_uv(mesh)

                mesh.visual = TextureVisuals(uv=uv, image=tex_img)
        except Exception as e:
            logger.warning(f"Failed to apply diffuse texture: {e}")
        
        # Save as OBJ (with possible MTL+texture)
        obj_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_3d_model.obj")
        mesh.export(obj_path)
        
        # Track potential MTL generated by exporter
        mtl_path = obj_path.replace('.obj', '.mtl')
        if not os.path.exists(mtl_path):
            mtl_path = None

        # Save as STL (no color)
        stl_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_3d_model.stl")
        mesh.export(stl_path)

        # Save as GLB (embeds texture reliably)
        glb_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_3d_model.glb")
        try:
            mesh.export(glb_path)
        except Exception as e:
            logger.warning(f"Failed to export GLB: {e}")
            glb_path = None

        # Export as GLTF (JSON + BIN)
        gltf_path = os.path.join(app.config['RESULTS_FOLDER'], f"{base_filename}_3d_model.gltf")
        try:
            mesh.export(gltf_path)
        except Exception as e:
            logger.warning(f"Failed to export GLTF: {e}")
            gltf_path = None
        
        return {
            'obj': obj_path,
            'stl': stl_path,
            'mtl': mtl_path,
            'glb': glb_path,
            'gltf': gltf_path
        }
        
    except Exception as e:
        logger.error(f"Error creating 3D model: {e}")
        return None

def _pack_metallic_roughness_texture(roughness_path: str, metallic_path: str, output_path: str) -> str:
    """Create a combined MetallicRoughness texture where G=roughness, B=metallic, R=0, A=1.

    Returns the output path.
    """
    try:
        rough = Image.open(roughness_path).convert('L')
        metal = Image.open(metallic_path).convert('L')

        if rough.size != metal.size:
            metal = metal.resize(rough.size, Image.LANCZOS)

        zero = Image.new('L', rough.size, 0)
        one = Image.new('L', rough.size, 255)
        # RGBA: R=0, G=roughness, B=metallic, A=255
        mr = Image.merge('RGBA', (zero, rough, metal, one))
        mr.save(output_path)
        return output_path
    except Exception as e:
        logger.warning(f"Failed to create metallicRoughness texture: {e}")
        # Fallback: copy roughness as metallicRoughness
        shutil.copyfile(roughness_path, output_path)
        return output_path


@app.route('/')
def index():
    base_models = get_available_base_models()
    return render_template('index.html', base_models=base_models)

@app.route('/3d_converter.html')
def three_d_converter():
    return render_template('3d_converter.html')

@app.route('/detect', methods=['POST'])
def detect():
    """Handle nail detection and segmentation"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # Generate a unique filename to prevent collisions
            original_filename = secure_filename(file.filename)
            filename_parts = os.path.splitext(original_filename)
            unique_filename = f"{filename_parts[0]}_{uuid.uuid4().hex[:8]}{filename_parts[1]}"
            
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Resize if needed to prevent memory issues
            resize_image_if_needed(file_path)
            
            # Load the model
            yolo_model = get_model()
            
            # Process the image
            results = yolo_model(file_path, task='segment')
            result = results[0]
            
            # Call memory cleanup after processing
            cleanup_memory()
            
            # Save detection image with bounding boxes
            output_filename = f"detected_{unique_filename}"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            result_plotted = result.plot()
            # Convert BGR (OpenCV) to RGB before saving with PIL
            result_plotted_rgb = cv2.cvtColor(result_plotted, cv2.COLOR_BGR2RGB)
            Image.fromarray(result_plotted_rgb).save(output_path)
            
            # Check for masks
            has_masks = hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0
            
            if not has_masks:
                return jsonify({'error': 'No nail detected in the image'}), 400
            
            # Get the first mask and box
            mask = result.masks.data[0].cpu().numpy()
            box = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # Safety check for box coordinates
            original_height, original_width = result.orig_shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_width, x2)
            y2 = min(original_height, y2)
            
            # Add padding for the nail (20% on each side)
            box_width = x2 - x1
            box_height = y2 - y1
            padding_x = int(box_width * 0.20)
            padding_y = int(box_height * 0.20)

            # Apply padding with safety checks
            x1_pad = max(0, x1 - padding_x)
            y1_pad = max(0, y1 - padding_y)
            x2_pad = min(original_width, x2 + padding_x)
            y2_pad = min(original_height, y2 + padding_y)
                        
            # Convert mask to boolean and resize to match original image if needed
            mask_height, mask_width = mask.shape
            if mask_height != original_height or mask_width != original_width:
                mask_8bit = (mask * 255).astype(np.uint8)
                resized_mask = cv2.resize(mask_8bit, (original_width, original_height), 
                                         interpolation=cv2.INTER_CUBIC)
                mask_bool = resized_mask > 127
            else:
                mask_bool = mask > 0.5
            
            # Process the mask for smooth edges
            processed_mask = process_nail_mask(mask_bool, result.orig_img)
                
            # Create a cropped version of the original image
            cropped_orig = result.orig_img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            
            # Create a cropped version of the mask
            cropped_mask = processed_mask[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Create RGBA image for the cropped region
            rgba_image = np.zeros((y2_pad-y1_pad, x2_pad-x1_pad, 4), dtype=np.uint8)
            # Convert BGR to RGB for correct colors in output
            cropped_rgb = cv2.cvtColor(cropped_orig, cv2.COLOR_BGR2RGB)
            rgba_image[..., :3] = cropped_rgb
            rgba_image[..., 3] = (cropped_mask * 255).astype(np.uint8)
            
            # Rotate 90 degrees counter-clockwise to correct nail orientation
            rgba_image = np.rot90(rgba_image, -1)
            
            # Save the cropped segmented image
            segmented_filename = f"segmented_{unique_filename}.png"
            segmented_path = os.path.join(app.config['SEGMENTATION_FOLDER'], segmented_filename)
            Image.fromarray(rgba_image).save(segmented_path, format='PNG', quality=100, optimize=False)
            
            # Generate PBR maps
            base_filename = f"nail_{uuid.uuid4().hex[:8]}"
            pbr_maps = generate_pbr_maps(rgba_image, base_filename)
            
            # Read selected base model name
            selected_base_model = request.form.get('base_model')

            # Create 3D model by deforming the selected base mesh using the height map
            height_map_path = pbr_maps['height']
            diffuse_map_path = pbr_maps.get('diffuse')
            model_3d = create_3d_model(height_map_path, base_filename, selected_base_model, diffuse_map_path)
            
            # Create zip file with all outputs
            zip_filename = f"{base_filename}_complete_package.zip"
            zip_path = os.path.join(app.config['RESULTS_FOLDER'], zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add segmented image
                zipf.write(segmented_path, "segmented_nail.png")
                
                # Add all PBR maps
                for map_name, map_path in pbr_maps.items():
                    zipf.write(map_path, f"{map_name}_map.png")
                
                # Add 3D models
                if model_3d:
                    if model_3d.get('obj') and os.path.exists(model_3d['obj']):
                        zipf.write(model_3d['obj'], "nail_3d_model.obj")
                    if model_3d.get('mtl') and model_3d['mtl'] and os.path.exists(model_3d['mtl']):
                        zipf.write(model_3d['mtl'], "nail_3d_model.mtl")
                    # include diffuse for reference
                    if diffuse_map_path and os.path.exists(diffuse_map_path):
                        zipf.write(diffuse_map_path, os.path.basename(diffuse_map_path))
                    if model_3d.get('stl') and os.path.exists(model_3d['stl']):
                        zipf.write(model_3d['stl'], "nail_3d_model.stl")
                    if model_3d.get('glb') and model_3d['glb'] and os.path.exists(model_3d['glb']):
                        zipf.write(model_3d['glb'], "nail_3d_model.glb")

            
            # Store results in session or pass as URL parameters
            # For now, we'll redirect to the results page with the base filename
            return redirect(url_for('show_results',
                                  segmented_image=f"segmentations/{segmented_filename}",
                                  detected_image=f"uploads/{output_filename}",
                                  zip_file=f"results/{zip_filename}",
                                  base_filename=base_filename))
            
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({'error': f'Detection failed: {str(e)}'}), 500


## Socket.IO realtime processing removed

@app.route('/3d_converter', methods=['POST'])
def process_3d_conversion():
    """Handle the complete workflow: upload → YOLO segmentation → PBR maps → 3D model"""
    try:
        if 'file' not in request.files:
            return render_template('error.html', error='No file uploaded'), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return render_template('error.html', error='No file selected'), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            original_filename = secure_filename(file.filename)
            filename_parts = os.path.splitext(original_filename)
            unique_filename = f"{filename_parts[0]}_{uuid.uuid4().hex[:8]}{filename_parts[1]}"
            
            # Save uploaded file
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(upload_path)
            
            # Resize if needed
            resize_image_if_needed(upload_path)
            
            # Load YOLO model
            yolo_model = get_model()
            
            # Perform YOLO segmentation
            results = yolo_model(upload_path, task='segment')
            result = results[0]
            
            # Clean up memory
            cleanup_memory()
            
            # Check for masks
            has_masks = hasattr(result, 'masks') and result.masks is not None and len(result.masks) > 0
            
            if not has_masks:
                return render_template('error.html', error='No nail detected in the image'), 400
            
            # Get the first mask and box
            mask = result.masks.data[0].cpu().numpy()
            box = result.boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # Safety check for box coordinates
            original_height, original_width = result.orig_shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(original_width, x2)
            y2 = min(original_height, y2)
            
            # Add padding
            box_width = x2 - x1
            box_height = y2 - y1
            padding_x = int(box_width * 0.20)
            padding_y = int(box_height * 0.20)
            
            x1_pad = max(0, x1 - padding_x)
            y1_pad = max(0, y1 - padding_y)
            x2_pad = min(original_width, x2 + padding_x)
            y2_pad = min(original_height, y2 + padding_y)
            
            # Convert mask to boolean and resize if needed
            mask_height, mask_width = mask.shape
            if mask_height != original_height or mask_width != original_width:
                mask_8bit = (mask * 255).astype(np.uint8)
                resized_mask = cv2.resize(mask_8bit, (original_width, original_height), 
                                         interpolation=cv2.INTER_CUBIC)
                mask_bool = resized_mask > 127
            else:
                mask_bool = mask > 0.5
            
            # Process mask for smooth edges
            processed_mask = process_nail_mask(mask_bool, result.orig_img)
            
            # Create cropped segmented image
            cropped_orig = result.orig_img[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            cropped_mask = processed_mask[y1_pad:y2_pad, x1_pad:x2_pad]
            
            # Create RGBA image
            rgba_image = np.zeros((y2_pad-y1_pad, x2_pad-x1_pad, 4), dtype=np.uint8)
            # Convert to RGB to preserve accurate colors
            cropped_rgb = cv2.cvtColor(cropped_orig, cv2.COLOR_BGR2RGB)
            rgba_image[..., :3] = cropped_rgb
            rgba_image[..., 3] = (cropped_mask * 255).astype(np.uint8)
            
            # Rotate 90 degrees counter-clockwise to correct nail orientation
            rgba_image = np.rot90(rgba_image, -1)
            
            # Save segmented image
            segmented_filename = f"segmented_{unique_filename}.png"
            segmented_path = os.path.join(app.config['SEGMENTATION_FOLDER'], segmented_filename)
            Image.fromarray(rgba_image).save(segmented_path)
            
            # Generate PBR maps
            base_filename = f"nail_{uuid.uuid4().hex[:8]}"
            pbr_maps = generate_pbr_maps(rgba_image, base_filename)
            
            # Read selected base model name
            selected_base_model = request.form.get('base_model')
            
            # Create 3D model
            height_map_path = pbr_maps['height']
            diffuse_map_path = pbr_maps.get('diffuse')
            model_3d = create_3d_model(height_map_path, base_filename, selected_base_model, diffuse_map_path)
            
            # Create zip file with all outputs
            zip_filename = f"{base_filename}_complete_package.zip"
            zip_path = os.path.join(app.config['RESULTS_FOLDER'], zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                # Add segmented image
                zipf.write(segmented_path, "segmented_nail.png")
                
                # Add all PBR maps
                for map_name, map_path in pbr_maps.items():
                    zipf.write(map_path, f"{map_name}_map.png")
                
                # Add 3D models
                if model_3d:
                    if model_3d.get('obj') and os.path.exists(model_3d['obj']):
                        zipf.write(model_3d['obj'], "nail_3d_model.obj")
                    if model_3d.get('mtl') and model_3d['mtl'] and os.path.exists(model_3d['mtl']):
                        zipf.write(model_3d['mtl'], "nail_3d_model.mtl")
                    if diffuse_map_path and os.path.exists(diffuse_map_path):
                        zipf.write(diffuse_map_path, os.path.basename(diffuse_map_path))
                    if model_3d.get('stl') and os.path.exists(model_3d['stl']):
                        zipf.write(model_3d['stl'], "nail_3d_model.stl")
                    if model_3d.get('glb') and model_3d['glb'] and os.path.exists(model_3d['glb']):
                        zipf.write(model_3d['glb'], "nail_3d_model.glb")
            
            # Redirect to results page with all parameters
            return redirect(url_for('show_results',
                                  segmented_image=f"segmentations/{segmented_filename}",
                                  detected_image=f"uploads/{upload_path.split('/')[-1]}",
                                  zip_file=f"results/{zip_filename}",
                                  base_filename=base_filename))
            
        else:
            return render_template('error.html', error='File type not allowed'), 400
            
    except Exception as e:
        logger.error(f"3D conversion error: {str(e)}")
        return render_template('error.html', error=f'Conversion failed: {str(e)}'), 500

@app.route('/3d_result.html')
def three_d_result_page():
    return render_template('3d_result.html')

@app.route('/3d_result/<base_filename>')
def three_d_result(base_filename):
    """Display the 3D conversion results"""
    try:
        # Check if all files exist
        results_folder = app.config['RESULTS_FOLDER']
        
        # Get file paths
        pbr_maps = {}
        for map_type in ['diffuse', 'roughness', 'height', 'normal', 'metallic', 'specular', 'ao']:
            map_path = os.path.join(results_folder, f"{base_filename}_{map_type}.png")
            if os.path.exists(map_path):
                pbr_maps[map_type] = url_for('static', filename=f"results/{os.path.basename(map_path)}")
        
        # Check for 3D models
        model_3d = {}
        obj_path = os.path.join(results_folder, f"{base_filename}_3d_model.obj")
        stl_path = os.path.join(results_folder, f"{base_filename}_3d_model.stl")
        mtl_path = os.path.join(results_folder, f"{base_filename}_3d_model.mtl")
        glb_path = os.path.join(results_folder, f"{base_filename}_3d_model.glb")
        
        if os.path.exists(obj_path):
            model_3d['obj'] = url_for('static', filename=f"results/{os.path.basename(obj_path)}")
        if os.path.exists(mtl_path):
            model_3d['mtl'] = url_for('static', filename=f"results/{os.path.basename(mtl_path)}")
        if os.path.exists(stl_path):
            model_3d['stl'] = url_for('static', filename=f"results/{os.path.basename(stl_path)}")
        if os.path.exists(glb_path):
            model_3d['glb'] = url_for('static', filename=f"results/{os.path.basename(glb_path)}")
        
        # Check for zip file
        zip_path = os.path.join(results_folder, f"{base_filename}_complete_package.zip")
        zip_url = None
        if os.path.exists(zip_path):
            zip_url = url_for('static', filename=f"results/{os.path.basename(zip_path)}")

        
        # Prepare results data
        results_data = {
            'base_filename': base_filename,
            'pbr_maps': pbr_maps,
            'model_3d': model_3d,
            'zip_file': zip_url
        }
        
        return render_template('3d_result.html', results=results_data)
        
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        return render_template('error.html', error=f'Error displaying results: {str(e)}'), 500

@app.route('/result.html')
def result():
    return render_template('result.html')


@app.route('/results')
def show_results():
    """Display the detection results"""
    try:
        # Get parameters from URL
        segmented_image = request.args.get('segmented_image')
        detected_image = request.args.get('detected_image')
        zip_file = request.args.get('zip_file')
        base_filename = request.args.get('base_filename')

        if not all([segmented_image, detected_image, zip_file]):
            return render_template('error.html', error='Invalid results parameters'), 400

        # Prepare additional asset URLs if they exist
        results_folder = app.config['RESULTS_FOLDER']
        model_glb_path = os.path.join(results_folder, f"{base_filename}_3d_model.glb") if base_filename else None
        diffuse_path = os.path.join(results_folder, f"{base_filename}_diffuse.png") if base_filename else None

        model_glb_url = url_for('static', filename=f"results/{os.path.basename(model_glb_path)}") if model_glb_path and os.path.exists(model_glb_path) else None
        diffuse_url = url_for('static', filename=f"results/{os.path.basename(diffuse_path)}") if diffuse_path and os.path.exists(diffuse_path) else None

        # Prepare results data for template
        results_data = {
            'segmented_image': url_for('static', filename=segmented_image),
            'detected_image': url_for('static', filename=detected_image),
            'zip_file': url_for('static', filename=zip_file),
            'base_filename': base_filename,
            'model_glb': model_glb_url,
            'diffuse_map': diffuse_url,
            'message': 'Nail detected and processed successfully'
        }
        
        return render_template('result.html', results=results_data)
        
    except Exception as e:
        logger.error(f"Error displaying results: {str(e)}")
        return render_template('error.html', error=f'Error displaying results: {str(e)}'), 500

@app.route('/3d_viewer.html')
def three_d_viewer():
    return render_template('3d_viewer.html')


@app.route('/ar_tryon.html')
def ar_tryon():
    return render_template('ar_tryon.html')


@app.route('/error.html')
def error():
    return render_template('error.html')


# Removed live try-on route per request

@app.route('/static/results/<filename>')
def serve_results(filename):
    """Serve generated result files with CORS headers"""
    response = send_from_directory(app.config['RESULTS_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files with CORS headers"""
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/static/uploads/<filename>')
def serve_uploads(filename):
    """Serve uploaded files with CORS headers"""
    response = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/static/segmentations/<filename>')
def serve_segmentations(filename):
    """Serve segmented files with CORS headers"""
    response = send_from_directory(app.config['SEGMENTATION_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/static/extracted/<filename>')
def serve_extracted(filename):
    """Serve extracted files with CORS headers"""
    response = send_from_directory(app.config['EXTRACTED_FOLDER'], filename)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/api/designs')
def list_designs():
    return jsonify([])

@app.route('/.env_config')
def serve_config():
    """Serve the configuration file"""
    return send_from_directory('.', '.env_config', mimetype='text/plain')




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)
