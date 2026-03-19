from importlib.metadata import version as get_version, PackageNotFoundError

try:
    __version__ = get_version("gpflow")
except PackageNotFoundError:
    __version__ = "develop"
