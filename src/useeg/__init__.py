from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("useeg")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = ["__version__"]
