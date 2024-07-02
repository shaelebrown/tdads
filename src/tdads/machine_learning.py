
# machine learning of persistence diagrams
from tdads.distance import *
from tdads.kernel import *
from multiprocessing import cpu_count
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC, SVR

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

        Examples
        --------
        >>> from tdads.machine_learning import diagram_mds
        >>> from tdads.distance import distance
        >>> from ripser import ripser
        >>> import numpy as np
        >>> # create 2 datasets
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> # compute persistence diagrams with ripser
        >>> diagram1 = ripser(data1)
        >>> diagram2 = ripser(data2)
        >>> # project into 2D with the 2-wasserstein distance
        >>> mds = diagram_mds()
        >>> mds.fit_transform([D1, D2])
        >>> # can also fit with a precomputed distance matrix
        >>> d_wass = distance()
        >>> dist_mat = d_wass.compute_matrix([D1, D2])
        >>> mds_precomp = diagram_mds(precomputed = True)
        >>> mds_precomp.fit_transform(dist_mat)
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

        Examples
        --------
        >>> from tdads.machine_learning import diagram_mds
        >>> from tdads.kernel import kernel
        >>> from ripser import ripser
        >>> import numpy as np
        >>> # create 2 datasets
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> # compute persistence diagrams with ripser
        >>> diagram1 = ripser(data1)
        >>> diagram2 = ripser(data2)
        >>> # fit model with the persistence Fisher kernel (sigma = t = 1)
        >>> kpca = diagram_kpca()
        >>> kpca_fitted = kpca.fit([D1, D2])
        >>> # can also fit with a precomputed distance matrix
        >>> pfk = kernel()
        >>> gram_mat = pfk.compute_matrix([D1, D2])
        >>> kpca_precomp = diagram_kpca(precomputed = True)
        >>> kpca_precomp_fitted = kpca_precomp.fit(gram_mat)
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
        
        Examples
        --------
        >>> from tdads.machine_learning import diagram_mds
        >>> from tdads.kernel import kernel
        >>> from ripser import ripser
        >>> import numpy as np
        >>> # create 2 datasets
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> # compute persistence diagrams with ripser
        >>> diagram1 = ripser(data1)
        >>> diagram2 = ripser(data2)
        >>> # fit models (regular and precomputed) with the 
        >>> # persistence Fisher kernel (sigma = t = 1)
        >>> kpca = diagram_kpca()
        >>> kpca_fitted = kpca.fit([D1, D2]) # or
        >>> pfk = kernel()
        >>> gram_mat = pfk.compute_matrix([D1, D2])
        >>> kpca_precomp = diagram_kpca(precomputed = True)
        >>> kpca_precomp_fitted = kpca_precomp.fit(gram_mat)
        >>> # create 2 new datasets
        >>> data3 = np.random((100,2))
        >>> data4 = np.random((100,2))
        >>> # project new data into 2D space
        >>> kpca_fitted.transform([D3, D4]) # or
        >>> cross_gram = pfk.compute_matrix([D1, D2], [D3, D4])
        >>> kpca_precomputed_fitted.transform([D3, D4])
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
        
        Examples
        --------
        >>> from tdads.machine_learning import diagram_mds
        >>> from tdads.kernel import kernel
        >>> from ripser import ripser
        >>> import numpy as np
        >>> # create 2 datasets
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> # compute persistence diagrams with ripser
        >>> diagram1 = ripser(data1)
        >>> diagram2 = ripser(data2)
        >>> # fit models (regular and precomputed) with the 
        >>> # persistence Fisher kernel (sigma = t = 1) and
        >>> # project into 2D space
        >>> kpca = diagram_kpca()
        >>> kpca.fit_transform([D1, D2]) # or
        >>> pfk = kernel()
        >>> gram_mat = pfk.compute_matrix([D1, D2])
        >>> kpca_precomp = diagram_kpca(precomputed = True)
        >>> kpca_precomp.fit_transform(gram_mat)
        '''
        self = self.fit(X, y)
        X_new = self.transform(X)
        return X_new

def diagram_svm():
    def __init__(self, cv:int = 1, dims:list = [0], sigmas:list = [1.0], ts:list = [1.0], precomputed:bool = False, C:float = 1.0, epsilon:float = 1.0, random_state:int = None):
        if epsilon != None:
            self.SVM = SVC(C = C, random_state = random_state)
        else:
            self.SVM = SVR(C = C, epsilon = epsilon)
        if isinstance(precomputed, type(True)) == False:
            raise Exception('precomputed must be True or False.')
        self.precomputed = precomputed
        if isinstance(cv, 1) == False:
            raise Exception('cv must be an integer.')
        if cv < 1:
            raise Exception('cv must be at least 1.')
        if not isinstance(dim,type(2)):
            raise Exception('dim must be an integer.')
        self.cv = cv
        if set([type(d) for d in dims]) != set(type(1)):
            raise Exception('Each dimension in dims must be an integer.')
        if min(dims) < 0:
            raise Exception('Each dimension in dims must be non-negative.')
        self.dims = dims
        if set([type(d) for d in dims]) != set(type(1)):
            raise Exception('Each dimension in dims must be an integer.')
        if min(dims) < 0:
            raise Exception('Each dimension in dims must be non-negative.')
        self.dims = dims
        if set([type(s) for s in sigmas]) != set([type(1), type(1.0)]):
                raise Exception('Each sigma value must be a number.')
        if min(sigmas) <= 0:
                raise Exception('Each sigma value must be positive.')
        self.sigmas = sigmas
        if set([type(t) for t in ts]) != set([type(1), type(1.0)]):
                raise Exception('Each t value must be a number.')
        if min(ts) <= 0:
                raise Exception('Each t value must be positive.')
        self.ts = ts
        

