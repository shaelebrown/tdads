
# machine learning of persistence diagrams
from tdads.distance import *
from tdads.kernel import *
from multiprocessing import cpu_count
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA

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
        `X` : {array-like of shape `(n_diagrams, n_diagrams)`} or {list of length `n_diagrams`}
            Either a precomputed distance matrix of `n_diagrams` many persistence diagrams (if `precomputed` was set to `True`) or a list of `n_diagrams` many persistence diagrams (otherwise).
        `y` : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        `X_new` : ndarray of shape `(n_diagrams, n_components)`
            `X` transformed in the new space.
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
    
# kernel PCA
class diagram_kpca:
    '''Kernel PCA with persistence diagrams.'''
    def __init__(self, n_components:int = 2, random_state:int = None, precomputed:bool = False, diagrams:list = None, dim:int = 0, sigma:float = 1.0, t:float = 1.0, n_cores:int = cpu_count() - 1):
        '''Multidimensional scaling with persistence diagrams.
        
        Parameters
        ----------
        `n_components` : int
            The number of dimensions in which to disperse the kernel values, default 2.
        `random_state` : int
            Determines the random number generator used for reproducibility, default None.
        `precomputed` : bool
            Determines whether a precomputed Gram matrix of persistence diagrams (`True`) or a list of persistence diagrams (`False`, the default) will be passed to the fit method.
        `diagrams` : list
            An optional list of persistence diagrams. If `precomputed` is `False` then `diagrams` will take the value of the
            list of diagrams passed to the `fit` method. Otherwise, `diagrams` must be supplied in order to perform inference (i.e. prediction).
        `dim` : int
            The non-negative homological dimension in which distances will be computed (default 0).
        `metric` : str
            One of \"W\" (default) or \"FIM\" for the wasserstein/bottleneck and Fisher information metric
            functions respectively.
        `sigma` : float
            The scale parameter for the Fisher information metric, default 1.
        `t` : float
            The positive dispersion parameter for the persistence Fisher kernel, default 1.
        `n_cores` : int
            The number of CPU cores to use for parallel computation of distance matrices. Default is the
            number of available cores minus 1.
        '''
        self.kernel = kernel(dim = dim, sigma = sigma, t = t, n_cores = n_cores)
        self.kPCA = KernelPCA(n_components = n_components, n_jobs = n_cores, random_state = random_state, kernel = 'precomputed')
        if isinstance(precomputed, type(True)) == False:
            raise Exception('precomputed must be True or False.')
        self.precomputed = precomputed
        if isinstance(diagrams, type([1,2])) == False:
            raise Exception('diagrams must be a list of diagrams.')
        self.diagrams = diagrams
    def __str__(self):
        s = 'Kernel PCA of persistence diagrams. Kernel used: ' + self.kernel.__str__
        return s
    def fit(self, X, y:any = None):
        '''Fit the model from data in X.
        
        Parameters
        ----------
        `X` : {array-like of shape `(n_diagrams, n_diagrams)`} or {list of length `n_diagrams`}
            Either a precomputed Gram matrix of `n_diagrams` many persistence diagrams (if `precomputed` was set to `True`) or a list of `n_diagrams` many persistence diagrams (otherwise).
        `y` : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        `self` : object
            Returns the instance itself.
        '''
        if self.precomputed == False:
            if isinstance(X, type(array([0,1]))):
                raise Exception('When precomputed is False, X must be a list of persistence diagrams.')
            self.diagrams = X
            X = self.kernel.compute_matrix(X)
        else:
            if isinstance(X, type([0,1])):
                raise Exception('When precomputed is True, X must be a ndarray distance matrix of persistence diagrams.')
        self.kPCA = self.kPCA.fit(X, y)
        return self
    def transform(self, X):
        '''Project new persistence diagrams into the embedding space.
        
        Parameters
        ----------
        `X` : {array-like of shape `(n_diagrams, n_diagrams)`} or {list of length `n_diagrams`}
            Either a precomputed (cross) Gram matrix of shape `(n_new_diagrams, n_diagrams)` (between the new persistence diagrams and the 
            training set diagrams, if `precomputed` was set to `True`) or a list of `n_new_diagrams` many persistence diagrams (otherwise).

        Returns
        -------
        `X_new` : ndarray
            The embedding of the new persistence diagrams.
        
        '''
        if self.precomputed == False:
            X = self.kernel.compute_matrix(X, self.diagrams)
        return self.kPCA.transform(X)
    def fit_transform(self, X, y:any = None):
        '''Fit the data in X and compute the position of the persistence diagrams in the embedding space.
        
        Parameters
        ----------
        `X` : {array-like of shape `(n_diagrams, n_diagrams)`} or {list of length `n_diagrams`}
            Either a precomputed Gram matrix of `n_diagrams` many persistence diagrams (if `precomputed` was set to `True`) or a list of `n_diagrams` many persistence diagrams (otherwise).
        `y` : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        `X_new` : ndarray
            `X` transformed in the new space.
        '''
        self = self.fit(X, y)
        X_new = self.transform(X)
        return X_new

