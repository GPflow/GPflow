try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:  # Python < 3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("gpflow")
except PackageNotFoundError:
    __version__ = "develop"
