import pkg_resources

try:
    __version__ = str(pkg_resources.get_distribution("gpflow").parsed_version)
except pkg_resources.DistributionNotFound:
    __version__ = "develop"
