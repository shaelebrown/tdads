# read version from installed package
from importlib.metadata import version
__version__ = version("tdads")

# only include public functions here
from .diagram_utils import check_diagram, preprocess_diagram
from .PH_utils import enclosing_radius
from .distance import distance
from .kernel import kernel
from .machine_learning import *
from .inference import *

__all__ = [
    'check_diagram',
    'preprocess_diagram',
    'enclosing_radius',
    'distance',
    'kernel',
    'diagram_mds',
    'diagram_kpca',
    'perm_test',
    'diagram_bootstrap'
]