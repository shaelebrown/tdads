
# functions to calculate distances between diagrams
from tdads.diagram_utils import *
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from numpy import concatenate, array, logical_or, exp, sqrt, arccos
from math import pi

# add in extra parameter for n_cores in distance constructor
class distance:
    def __init__(self, dim:int = 0,metric='W',p:float=2, sigma:float=None, n_cores:int=1):
        '''Create a distance object.
        
        Available distance metrics are the wasserstein, bottleneck and Fisher information metric distances.

        Parameters
        ----------
        `dim` : int
            The non-negative homological dimension in which distances will be computed (default 0).
        `metric` : str
            One of \"W\" (default) or \"FIM\" for the wasserstein/bottleneck and Fisher information metric
            functions respectively.
        `p` : float
            The power parameter for the wasserstein metric, must be at least 1 (default 2).
        `sigma` : float
            The scale parameter for the Fisher information metric, default None but must be supplied when
            `metric` is \"FIM\".
        `n_cores` : int
            The number of CPU cores to use for parallel computation of distance matrices.
        '''
        if not isinstance(dim,type(2)):
            raise Exception('dim must be an integer.')
        if dim < 0:
            raise Exception('dim must be non-negative.')
        self.dim = dim
        if not isinstance(metric,type('')):
            raise Exception('The distance metric must be a string.')
        if metric != 'W' and metric != 'FIM':
            raise Exception('The distance metric must be either \"W\" for the wasserstein/bottleneck distances or \"FIM\" for the Fisher information metric.')
        self.metric = metric
        if metric == 'W':
            if not isinstance(p,type(2)) and not isinstance(p,type(2.0)):
                raise Exception('For the wasserstein/bottleneck distance p must be a number.')
            if p < 1:
                raise Exception('For the wasserstein/bottleneck distance p must be at least 1.')
            self.p = p
            self.sigma = None
        else:
            if not isinstance(sigma,type(2)) and not isinstance(sigma,type(2.0)):
                raise Exception('For the Fisher information metric sigma must be a number.')
            if sigma <= 0:
                raise Exception('For the Fisher information metric sigma must be positive.')
            self.sigma = sigma
            self.p = None
        if not isinstance(n_cores,type(2)):
            raise Exception('n_cores must be an integer.')
        if n_cores < 0:
            raise Exception('n_cores must be non-negative.')
        # then check if n_cores is more than the number of available cores
        self.n_cores = n_cores
    def __str__(self):
        if self.sigma != None:
            M = 'Fisher information metric with parameter sigma = ' + str(self.sigma) + '.'
        else:
            if self.p == float('inf'):
                M = 'Bottleneck distance.'
            else:
                M = str(self.p) + '-wasserstein distance.'
        return M
    def compute(self,D1,D2) -> float:
        '''Compute the distance between two persistence diagrams.

        Parameters
        ----------
        `D1` : any
            The first persistence diagram (computed from either the ripser, gph, flagser, gudhi or cechmate packages).
        `D2` : any
            The second persistence diagram (\"\").
        
        Returns
        -------
        float
            The numeric distance calculation value.

        Examples
        --------
        >>> from tdads import distance
        >>> from ripser import ripser
        >>> import numpy as np
        >>> # create 2 datasets
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> compute persistence diagrams with ripser
        >>> diagram1 = ripser(data1)
        >>> diagram2 = ripser(data2)
        >>> # create distance object
        >>> d_wass = distance() # 2-wasserstein distance
        >>> # compute distance
        >>> d_wass.compute(diagram1, diagrams2)

        Citations
        ---------
        Kerber M, Morozov D and Nigmetov A (2017). "Geometry Helps to Compare Persistence Diagrams." https://dl.acm.org/doi/10.1145/3064175.
        
        Le T, Yamada M (2018). "Persistence fisher kernel: a riemannian manifold kernel for persistence diagrams." https://proceedings.neurips.cc/paper/2018/file/959ab9a0695c467e7caf75431a872e5c-Paper.pdf.
        
        Vlad I. Morariu, Balaji Vasan Srinivasan, Vikas C. Raykar, Ramani Duraiswami, and Larry S. Davis. Automatic online tuning for fast Gaussian summation. Advances in Neural Information Processing Systems (NIPS), 2008.
        '''
        # preprocess diagrams
        D1 = preprocess_diagram(D1, ret=True)
        D2 = preprocess_diagram(D2, ret=True)
        # subset diagrams to correct dimensions
        D1_sub = D1[self.dim]
        D2_sub = D2[self.dim]
        # avoid infinity's in diagrams
        if float('inf') in D1_sub[:,1] or float('inf') in D2_sub[:,1]:
            raise Exception('Infinity value found in a diagram - infinities should be removed prior to distance calculations.')
        # remove diagonal points from both diagrams
        D1_sub = D1_sub[(D1_sub[:,0] < D1_sub[:,1]),:]
        D2_sub = D2_sub[(D2_sub[:,0] < D2_sub[:,1]),:]
        # check for empty diagrams, return distance 0
        if D1_sub.shape[0] == 0 and D2_sub.shape[0] == 0:
            return 0
        # compute diagonal projections and concatenate to opposite diagram
        if D1_sub.shape[0] > 0:
            diag1 = concatenate([0.5*array([x[0]+x[1], x[0]+x[1]]).reshape((1,2)) for x in D1_sub])
            n_diag1 = diag1.shape[0]
        else:
            diag1 = empty((0,2))
            n_diag1 = 0
        if D2_sub.shape[0] > 0:
            diag2 = concatenate([0.5*array([x[0]+x[1], x[0]+x[1]]).reshape((1,2)) for x in D2_sub])
            n_diag2 = diag2.shape[0]
        else:
            diag2 = empty((0,2))
            n_diag2 = 0
        D2_sub = concatenate([D2_sub, diag1])
        D1_sub = concatenate([D1_sub, diag2])
        # compute distance
        if self.metric == 'W':
            # wasserstein/bottleneck distance
            # calculate max (chebyshev) distance between each pair of rows in D1_sub and D2_sub
            dist_mat = cdist(D1_sub, D2_sub, metric='chebyshev')
            # set distances between points and their projections to 0
            v1 = D1_sub.shape[0] - n_diag2
            v2 = D2_sub.shape[0] - n_diag1
            if v1 < D1_sub.shape[0] and v2 < D2_sub.shape[0]:
                dist_mat[range(D1_sub.shape[0] - n_diag2, D1_sub.shape[0]),range(D2_sub.shape[0] - n_diag1, D2_sub.shape[0])] = 0
            # if wasserstein distance then exponentiate
            if self.p < float('inf'):
                dist_mat = dist_mat**self.p
            # solve linear sum assignment problem
            row_ind, col_ind = linear_sum_assignment(cost_matrix = dist_mat, maximize = False)
            # subset to remove matches between projection points
            good_inds = logical_or(row_ind < D1_sub.shape[0] - n_diag2, col_ind < D2_sub.shape[0] - n_diag1)
            row_ind = row_ind[good_inds]
            col_ind = col_ind[good_inds]
            # return distance value
            if self.p < float('inf'):
                return dist_mat[row_ind, col_ind].sum()**(1/self.p)
            else:
                return dist_mat[row_ind, col_ind].max()
        else:
            # Fisher information metric distance
            theta = concatenate([D1_sub, D2_sub])
            # exact calculation for now...
            rho1 = [exp(-1*cdist(x.reshape((1,2)), D1_sub, metric='euclidean')**2/(2*self.sigma**2)).sum()/(sqrt(2*pi)*self.sigma) for x in theta]
            rho2 = [exp(-1*cdist(x.reshape((1,2)), D2_sub, metric='euclidean')**2/(2*self.sigma**2)).sum()/(sqrt(2*pi)*self.sigma) for x in theta]
            # check for same vectors
            if rho1 == rho2:
                return 0
            # normalize
            rho1 = rho1/sum(rho1)
            rho2 = rho2/sum(rho2)
            # dot product
            norm = sum([sqrt(x)*sqrt(y) for x,y in zip(rho1, rho2)])
            # check bounds and return arc cos
            if norm > 1:
                norm = 1
            if norm < -1:
                norm = -1
            return arccos(norm)

        
        
