# read version from installed package
from importlib.metadata import version
__version__ = version("tdads")

# only include public functions here
from .diagram_utils import check_diagram, preprocess_diagram

__all__ = [
    'check_diagram',
    'preprocess_diagram'
]