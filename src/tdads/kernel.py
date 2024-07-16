
# kernel methods for persistence diagrams
# currently just the persistence Fisher kernel
from tdads.diagram_utils import *
from tdads.distance import *
from multiprocessing import cpu_count
from numpy import exp

class kernel:
    def __init__(self, dim:int = 0, sigma:float = 1, t:float = 1, inf_replace_val:float = None, n_cores:int = cpu_count() - 1):
        '''Kernel functions for persistence diagrams.
        
        Current available option is the persistence Fisher kernel.

        Parameters
        ----------
        `dim` : int, default 0
            The non-negative homological dimension in which kernels will be computed.
        `sigma` : float, default 1
            The scale parameter for the Fisher information metric.
        `t` : float, default 1
            The positive dispersion parameter for the persistence Fisher kernel.
        `inf_replace_val` : float or int, default `None`
            The value with which `inf` values should be replaced, if desired. If `None`, topological features with `inf` values will 
            be ignored, otherwise original diagrams will be modified.
        `n_cores` : int, default is the number of available cores minus 1
            The number of CPU cores to use for parallel computation of distance matrices.
        
        Attributes
        ----------
        `distance` : `tdads.distance.distance`
            The Fisher information metric distance object.
        `t` : float
            The input `t` parameter.
        '''
        self.dist = distance(dim = dim, metric = 'FIM', sigma = sigma, inf_replace_val = inf_replace_val, n_cores = n_cores)
        if not isinstance(t,type(2)) and not isinstance(t,type(2.0)):
            raise Exception('For the persistence Fisher kernel, t must be a number.')
        if t <= 0:
            raise Exception('For the persistence Fisher kernel, t must be positive.')
        self.t = t
    def __str__(self):
        '''Describe a persistence Fisher kernel by its `sigma` and `t` parameters.'''
        return 'Persistence Fisher kernel with sigma = ' + str(self.dist.sigma) + ', t = ' + str(self.t) + '.'
    def compute(self, D1, D2):
        '''Compute the kernel value between two persistence diagrams.

        Parameters
        ----------
        `D1` : any
            The first persistence diagram (computed from either the ripser, gph, flagser, gudhi or cechmate packages).
        `D2` : any
            The second persistence diagram.
        
        Returns
        -------
        float
            The numeric kernel calculation value.

        Examples
        --------
        >>> from tdads import kernel
        >>> from ripser import ripser
        >>> import numpy as np
        >>> # create 2 datasets
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> # compute persistence diagrams with ripser
        >>> diagram1 = ripser(data1)
        >>> diagram2 = ripser(data2)
        >>> # create kernel object
        >>> k = kernel()
        >>> # compute kernel value
        >>> k.compute(diagram1, diagrams2)

        Citations
        ---------        
        Le T, Yamada M (2018). "Persistence fisher kernel: a riemannian manifold kernel for persistence diagrams." https://proceedings.neurips.cc/paper/2018/file/959ab9a0695c467e7caf75431a872e5c-Paper.pdf.
        '''
        d_FIM = self.dist.compute(D1, D2)
        return exp(-1*self.t*d_FIM)
    def compute_matrix(self, diagrams, other_diagrams = None):
        '''Compute a Gram (kernel) matrix between one or two lists of persistence diagrams.
        Parameters
        ----------
        `diagrams` : list
            The first first of persistence diagram (computed from either the ripser, gph, flagser, gudhi or cechmate packages).
        `other_diagrams` : any
            The optional second list of persistence diagram for computing a cross-Gram matrix. Default `None`.
        
        Returns
        -------
        numpy.ndarray
            The (cross) Gram matrix.

        Examples
        --------
        >>> from tdads import kernel
        >>> from ripser import ripser
        >>> import numpy as np
        >>> # create 2 datasets
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> # compute persistence diagrams with ripser
        >>> diagram1 = ripser(data1)
        >>> diagram2 = ripser(data2)
        >>> # create kernel object
        >>> k = kernel()
        >>> # compute Gram matrix
        >>> k.compute_matrix([diagram1, diagram2])
        >>> # this is the same as:
        >>> k.compute_matrix([diagram1, diagram2], [diagram1, diagram2])
        '''
        dm_FIM = self.dist.compute_matrix(diagrams, other_diagrams)
        return exp(-1*self.t*dm_FIM)