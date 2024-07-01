
# machine learning of persistence diagrams
from tdads.distance import *
from tdads.kernel import *
from multiprocessing import cpu_count
from sklearn.manifold import MDS

# multidimensional scaling
class diagram_mds:
    '''Multidimensional scaling with persistence diagrams.'''
    def __init__(self, n_components:int = 2, random_state:int = None, precomputed:bool = False, dim:int = 0, metric:str = 'W', p:float = 2, sigma:float = None, n_cores:int = cpu_count() - 1):
        '''Multidimensional scaling with persistence diagrams.
        
        Parameters
        ----------
        `n_components` : int
            The number of dimensions in which to disperse the distance values, default 2.
        `random_state` : int
            Determines the random number generator used for reproducibility, default None.
        `precomputed` : bool
            Determines whether a precomputed distance matrix of persistence diagrams (`True`) or a list of persistence diagrams (`False`, the default) will be passed to the fit method.
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
            The number of CPU cores to use for parallel computation of distance matrices. Default is the
            number of available cores minus 1.
        
        '''
        self.distance = distance(dim = dim, metric = metric, p = p, sigma = sigma, n_cores = n_cores)
        self.MDS = MDS(n_components = n_components, metric = False, n_jobs = n_cores, random_state = random_state, dissimilarity = 'precomputed')
        if isinstance(precomputed, type(True)) == False:
            raise Exception('precomputed must be True or False.')
        self.precomputed = precomputed
    def __str__(self):
        s = 'Non-metric multidimensional scaling of persistence diagrams. Distance metric used: ' + self.distance.__str__
        return s
    def fit_transform(self, X, y:any = None):
        '''Fit the data in X and compute the position of the persistence diagrams in the embedding space.
        
        Parameters
        ----------
        X : ndarray or list
            Either a precomputed distance matrix of persistence diagrams (if `precomputed` was set to `True`) or a list of persistence diagrams (otherwise).
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : ndarray
            X transformed in the new space.
        '''
        if self.precomputed == False:
            if isinstance(X, type(array([0,1]))):
                raise Exception('When precomputed is False, X must be a list of persistence diagrams.')
            X = self.distance.compute_matrix(X)
        else:
            if isinstance(X, type([0,1])):
                raise Exception('When precomputed is True, X must be a ndarray distance matrix of persistence diagrams.')
        X_new = self.MDS.fit_transform(X, y)
        return X_new

