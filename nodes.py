import comfy.model_management as mm
import os
import folder_paths
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torch
import numpy as np
import cv2
from PIL import Image

import os

from .distillanydepth.modeling.archs.dam.dam import DepthAnything
from .distillanydepth.utils.image_util import chw2hwc, colorize_depth_maps
from .distillanydepth.midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose

# Try to import DepthAnythingV2, use DepthAnything as fallback if failed
try:
    from .distillanydepth.depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_ANYTHING_V2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DepthAnythingV2 import failed: {e}")
    print("Will use DepthAnything as fallback for Base and Small models")
    DepthAnythingV2 = DepthAnything  # Use DepthAnything as fallback
    DEPTH_ANYTHING_V2_AVAILABLE = False


import logging
log = logging.getLogger(__name__)

class DownloadDistillAnyDepthModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":(["Distill-Any-Depth-Large", "Distill-Any-Depth-Base", "Distill-Any-Depth-Small", "Distill-Any-Depth-Teacher-Large-2w-iter"],
                        {"default": "Distill-Any-Depth-Large"}),
            }
        }
        
    RETURN_TYPES = ("DISTILLPIPE",)
    RETURN_NAMES = ("pipeline",)
    
    FUNCTION = "loadmodel"
    CATEGORY = "DistillAnyDepth"

    def loadmodel(self, model):
        # Device management
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        
        model_configs = {
            "Distill-Any-Depth-Large": {
                "repo_id": "xingyang1/Distill-Any-Depth", 
                "filename": "large/model.safetensors",
                "model_class": DepthAnything,
                "model_config": {
                    "encoder": "vitl", 
                    "features": 256, 
                    "out_channels": [256, 512, 1024, 1024], 
                    "use_bn": False, 
                    "use_clstoken": False, 
                    "max_depth": 150.0, 
                    "mode": 'disparity',
                    "pretrain_type": 'dinov2',
                    "del_mask_token": False 
                },
                "needs_key_fix": True  
            },
            "Distill-Any-Depth-Base": {
                "repo_id": "xingyang1/Distill-Any-Depth", 
                "filename": "base/model.safetensors",
                "model_class": DepthAnythingV2 if DEPTH_ANYTHING_V2_AVAILABLE else DepthAnything,
                "model_config": {
                    "encoder": 'vitb',
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                } if DEPTH_ANYTHING_V2_AVAILABLE else {
                    "encoder": 'vitb',
                    "features": 128,
                    "out_channels": [96, 192, 384, 768],
                    "use_bn": False,
                    "use_clstoken": False,
                    "max_depth": 150.0,
                    "mode": 'disparity',
                    "pretrain_type": 'dinov2',
                    "del_mask_token": True
                },
                "needs_key_fix": False  
            },
            "Distill-Any-Depth-Small": {
                "repo_id": "xingyang1/Distill-Any-Depth", 
                "filename": "small/model.safetensors",
                "model_class": DepthAnythingV2 if DEPTH_ANYTHING_V2_AVAILABLE else DepthAnything,
                "model_config": {
                    "encoder": 'vits',
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                } if DEPTH_ANYTHING_V2_AVAILABLE else {
                    "encoder": 'vits',
                    "features": 64,
                    "out_channels": [48, 96, 192, 384],
                    "use_bn": False,
                    "use_clstoken": False,
                    "max_depth": 150.0,
                    "mode": 'disparity',
                    "pretrain_type": 'dinov2',
                    "del_mask_token": True
                },
                "needs_key_fix": False  
            },
            "Distill-Any-Depth-Teacher-Large-2w-iter": {
                "repo_id": "xingyang1/Distill-Any-Depth", 
                "filename": "Distill-Any-Depth-Dav2-Teacher-Large-2w-iter/model.safetensors",
                "model_class": DepthAnything,
                "model_config": {
                    "encoder": "vitl", 
                    "features": 256, 
                    "out_channels": [256, 512, 1024, 1024], 
                    "use_bn": False, 
                    "use_clstoken": False, 
                    "max_depth": 150.0, 
                    "mode": 'disparity',
                    "pretrain_type": 'dinov2',
                    "del_mask_token": False  
                },
                "needs_key_fix": True  
            },
        }
        
        if model not in model_configs:
            raise ValueError(f"Unsupported model: {model}")
        
        config = model_configs[model]
        
        # Get model path
        models_dir = folder_paths.models_dir
        download_path = os.path.join(models_dir, "distill_any_depth")
        os.makedirs(download_path, exist_ok=True)
        
        model_path = os.path.join(download_path, f"{model}.safetensors")
        
        # Download model weight file
        if not os.path.exists(model_path):
            print(f"Downloading model from HuggingFace: {model}")
            try:
                checkpoint_path = hf_hub_download(
                    repo_id=config["repo_id"],
                    filename=config["filename"],
                    repo_type="model",
                    local_dir=download_path,
                    local_dir_use_symlinks=False
                )
                
                # Rename to unified format
                if os.path.exists(checkpoint_path) and checkpoint_path != model_path:
                    import shutil
                    shutil.move(checkpoint_path, model_path)
                    
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {str(e)}")
        
        # Load model
        print(f"Loading model: {model}")
        print(f"Using model class: {config['model_class'].__name__}")
        print(f"DepthAnythingV2 available: {DEPTH_ANYTHING_V2_AVAILABLE}")
        try:
            # Initialize model with appropriate class
            ModelClass = config["model_class"]
            depth_model = ModelClass(**config["model_config"])
            depth_model = depth_model.to(device)
            
            # Handle mask_token parameter (for Base/Small models using DepthAnything)
            if (not DEPTH_ANYTHING_V2_AVAILABLE and model != "Distill-Any-Depth-Large" and 
                config["model_config"].get("del_mask_token", False) and 
                hasattr(depth_model.backbone, 'mask_token')):
                delattr(depth_model.backbone, 'mask_token')
                print(f"Removed mask_token parameter to match pretrained weights")
            
            # Load weights
            model_weights = load_file(model_path)
            
            # Handle weight loading based on model type and availability
            if config["needs_key_fix"] or (not DEPTH_ANYTHING_V2_AVAILABLE and model != "Distill-Any-Depth-Large"):
                # Large model or Base/Small models using DepthAnything: Fix key names
                fixed_weights = {}
                for key, value in model_weights.items():
                    if key.startswith('pretrained.'):
                        new_key = key.replace('pretrained.', 'backbone.')
                        fixed_weights[new_key] = value
                    else:
                        fixed_weights[key] = value
                
                missing_keys, unexpected_keys = depth_model.load_state_dict(fixed_weights, strict=False)
            else:
                # Base/Small models using DepthAnythingV2: Load weights directly
                missing_keys, unexpected_keys = depth_model.load_state_dict(model_weights, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            # Set to evaluation mode
            depth_model.eval()
            
            # Create pipeline object
            pipeline = {
                "model": depth_model,
                "device": device,
                "offload_device": offload_device,
                "model_name": model,
                "model_type": "large" if model == "Distill-Any-Depth-Large" else "base_small"
            }
            
            print(f"Model {model} loaded successfully!")
            return (pipeline,)
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

class DistillAnyDepthProcessImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("DISTILLPIPE",),
                "image": ("IMAGE",),
                "processing_resolution": ("INT", {
                    "default": 756, 
                    "min": 64, 
                    "max": 2048, 
                    "step": 16,
                    "tooltip": "Processing resolution, must be multiple of 14"
                }),
                "output_type": (["colorized", "grayscale", "grayscale 16-bit"], {
                    "default": "colorized",
                    "tooltip": "Output type: colorized depth map or grayscale depth map"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth_image",)
    FUNCTION = "process_image"
    CATEGORY = "DistillAnyDepth"
    
    def process_image(self, pipeline, image, processing_resolution, output_type):
        # Get model and device information
        model = pipeline["model"]
        device = pipeline["device"]
        offload_device = pipeline["offload_device"]
        
        # Move model to main device
        model = model.to(device)
        
        # Convert ComfyUI image format
        if image.dim() == 4:
            image = image[0]  # Take the first batch
        
        # Convert to numpy array and adjust to [0,255] range
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        
        # Preprocess image
        image_np = np.array(pil_image)[..., ::-1] / 255.0  # Convert RGB to BGR and normalize
        
        # Create transform pipeline
        transform = Compose([
            Resize(
                processing_resolution, 
                processing_resolution, 
                resize_target=False, 
                keep_aspect_ratio=False,
                ensure_multiple_of=14, 
                resize_method='lower_bound', 
                image_interpolation_method=cv2.INTER_CUBIC
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet()
        ])
        
        # Apply transform
        image_tensor = transform({'image': image_np})['image']
        image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
        
        # Model inference
        with torch.no_grad():
            pred_disp, _ = model(image_tensor)
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Convert depth map to numpy
        pred_disp_np = pred_disp.cpu().detach().numpy()[0, 0, :, :]
        
        # Normalize depth map
        pred_disp_normalized = (pred_disp_np - pred_disp_np.min()) / (pred_disp_np.max() - pred_disp_np.min())
        
        # Get original image dimensions
        h, w = image_np.shape[:2]
        
        if output_type == "colorized":
            # Colorized depth map
            depth_colored = colorize_depth_maps(
                pred_disp_normalized[None, ..., None], 0, 1, cmap="Spectral_r"
            ).squeeze()
            depth_colored = (depth_colored * 255).astype(np.uint8)
            depth_colored_hwc = chw2hwc(depth_colored)
            depth_colored_hwc = cv2.resize(depth_colored_hwc, (w, h), cv2.INTER_LINEAR)
            output_image = depth_colored_hwc
        
        elif output_type == "grayscale":
            # Grayscale depth map
            depth_gray = (pred_disp_normalized * 255).astype(np.uint8)
            depth_gray_hwc = np.stack([depth_gray] * 3, axis=-1)
            depth_gray_hwc = cv2.resize(depth_gray_hwc, (w, h), cv2.INTER_LINEAR)
            output_image = depth_gray_hwc

        elif output_type == "grayscale 16-bit":
            # 16-bit Grayscale depth map
            depth_gray = (pred_disp_normalized * 65536).astype(np.uint16)
            depth_gray_hwc = np.stack([depth_gray] * 3, axis=-1)
            depth_gray_hwc = cv2.resize(depth_gray_hwc, (w, h), cv2.INTER_LINEAR)
            output_image = depth_gray_hwc
        
        # Move model back to offload device
        model = model.to(offload_device)
        
        # Convert back to ComfyUI format
        if output_type == "grayscale 16-bit":
            output_image = output_image.astype(np.float32) / 65536.0
        else:
            output_image = output_image.astype(np.float32) / 255.0
        output_tensor = torch.from_numpy(output_image).unsqueeze(0)
        
        return (output_tensor,)


NODE_CLASS_MAPPINGS = {
    "DownloadDistillAnyDepthModel": DownloadDistillAnyDepthModel,
    "DistillAnyDepthProcessImage": DistillAnyDepthProcessImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadDistillAnyDepthModel": "Download Distill Any Depth Model",
    "DistillAnyDepthProcessImage": "Distill Any Depth Process Image",
}
