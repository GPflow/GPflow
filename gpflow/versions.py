import sys

if sys.version_info >= (3, 8):
    from importlib.metadata import PackageNotFoundError, version
else:
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("gpflow")
except PackageNotFoundError:
    __version__ = "develop"
