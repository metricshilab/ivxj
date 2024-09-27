# read version from installed package
from importlib.metadata import version
__version__ = version("ivxj")

# populate package namespace
from ivxj.ivxj import ivxj