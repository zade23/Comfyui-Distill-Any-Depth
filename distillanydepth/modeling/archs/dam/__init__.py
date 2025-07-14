# DAM module initialization
try:
    from .dam import DepthAnything
    __all__ = ['DepthAnything']
except ImportError as e:
    print(f"warning: dam module {e}")
    __all__ = [] 