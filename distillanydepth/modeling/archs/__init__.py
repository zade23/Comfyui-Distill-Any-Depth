# Archs module initialization
try:
    from .dam.dam import DepthAnything
    __all__ = ['DepthAnything']
except ImportError as e:
    print(f"warning: archs module {e}")
    __all__ = [] 