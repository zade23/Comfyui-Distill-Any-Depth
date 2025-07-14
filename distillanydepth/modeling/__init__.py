# Modeling module initialization
try:
    from .archs.dam.dam import DepthAnything
    __all__ = ['DepthAnything']
except ImportError as e:
    print(f"warning: modeling module {e}")
    __all__ = [] 