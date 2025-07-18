# Distill-Any-Depth package initialization
__version__ = "1.0.0"
__author__ = "Distill-Any-Depth"

# 导入核心模块
try:
    from .modeling.archs.dam.dam import DepthAnything
    from .utils.image_util import chw2hwc, colorize_depth_maps
    from .midas.transforms import Resize, NormalizeImage, PrepareForNet
    
    __all__ = ['DepthAnything', 'chw2hwc', 'colorize_depth_maps', 'Resize', 'NormalizeImage', 'PrepareForNet']
except ImportError as e:
    print(f"warning: some modules import failed: {e}")
    __all__ = [] 