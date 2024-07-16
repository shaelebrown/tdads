
# machine learning of persistence diagrams
from tdads.distance import *
from tdads.kernel import *
from multiprocessing import cpu_count
from sklearn.manifold import MDS
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC, SVR
from numpy import concatenate, array

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

        Attributes
        ----------
        `distance` : tdads.distance.distance
            The object used to compute the distance matrix of persistence diagrams. 
        `MDS` : sklearn.manifold._mds.MDS
            The sklearn.manifold.MDS object used for embedding the distance matrix.
        `precomputed` : bool
            The input `precomputed` parameter.
        '''
        self.distance = distance(dim = dim, metric = metric, p = p, sigma = sigma, n_cores = n_cores)
        self.MDS = MDS(n_components = n_components, metric = False, n_jobs = n_cores, random_state = random_state, dissimilarity = 'precomputed')
        if isinstance(precomputed, type(True)) == False:
            raise Exception('precomputed must be True or False.')
        self.precomputed = precomputed
    def __str__(self):
        '''Describe a persistence diagram multidimensional scaling object via its distance metric.'''
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
        
        Attributes
        ----------
        `kernel` : tdads.kernel.kernel
            The object used to compute the (cross) Gram matrices of persistence diagrams. 
        `kPCA` : sklearn.decomposition._kernel_pca.KernelPCA
            The kernel PCA object used for embedding the persistence diagrams.
        `precomputed` : bool
            The input `precomputed` parameter.
        `diagrams` : list of length `n_diagrams`
            The input `diagrams` parameter for inference.
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
        '''Describe a persistence diagram kernel principle components analysis object via its kernel function.'''
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

# class diagram_svm():
#     '''Support vector machine for persistence diagrams.'''
#     def __init__(self, diagrams:list = None, cv:int = 1, dims:list = [0], sigmas:list = [1.0], ts:list = [1.0], precomputed:bool = False, Cs:list = [1.0], epsilons:list = [0.1], n_cores:int = cpu_count() - 1):
#         '''Support vector machines for persistence diagrams.
        
#         Parameters
#         ----------
#         `diagrams` : list of persistence diagrams, default None
#             When `precomputed = True`, `diagrams` must be supplied in order to call the predict method.
#         `cv` : int, default 1
#             The number of folds for cross validattion. The default is no cross-validation.
#         `dims` : list of int, default [0]
#             The homological dimensions in which to fit SVM models.
#         `sigmas` : list of float, default [1.0]
#             The values of `sigma` for the persistence Fisher kernel.
#         `ts` : list of float, default [1.0]
#             The values of `t` for the persistence Fisher kernel.
#         `precomputed` : bool, default False
#             If `True` then the `fit` method will expect precomputed Gram matrices for training, otherwise
#             a list of persistence diagrams.
#         `Cs` : list of float, default [1.0]
#             A list of regularization parameters. The strength of the regularization is inversely proportional to C. 
#             Must be strictly positive. The penalty is a squared l2.
#         `epsilons` : list of float, default [0.1]
#             A list of epsilons in the epsilon-SVR model. If performing classification set `epsilon = None`. 
#             `epsilons` specifies the epsilon-tubes within which no penalty is associated 
#             in the training loss function with points predicted within a distance epsilon from the actual value. 
#             Must be non-negative.
#         `n_cores` : int
#             The number of CPU cores to use for parallel computation of distance matrices. Default is the
#             number of available cores minus 1.

#         Attributes
#         ----------
#         `precomputed` : bool
#             The input `precomputed` parameter.
#         `cv` : int
#             The input `cv` parameter. 
#         `diagrams` : list of length `n_diagrams`
#             The input `diagrams` parameter for inference when `precomputed` is `False`.
#         `n_cores` : int
#             The input `n_cores` parameter.
#         `parameter_grid` : ndarray either of shape `(num_param_combos, 5)` or `(num_param_combos, 4)`
#             The cartesian product of all possible model parameter combinations (the number of which is `num_param_combos`).
#             The columns give the values of `dims`, `sigmas`, `ts`, `Cs` and `epsilons` (for regression) in that order, resulting
#             in five columns for classification and four for regression.
#         `models` : list of sklearn.svm._classes.SVR or sklearn.svm._classes.SVC of length `num_param_combos`
#             One model for each row of `parameter_grid` (to be fit with those parameters).
#         `final_model` : None or {sklearn.svm._classes.SVR or sklearn.svm._classes.SVC}
#             Initially None but becomes the optimal model object based on cross-validation results
#             once the `fit` method has been called.
#         `final_model_kernel` : None or tdads.kernel.kernel
#             Initially None but becomes the kernel object with parameters determined by `final_model`
#             once the `fit` method has been called.
#         '''
#         if isinstance(precomputed, type(True)) == False:
#             raise Exception('precomputed must be True or False.')
#         self.precomputed = precomputed
#         if isinstance(cv, type(1)) == False:
#             raise Exception('cv must be an integer.')
#         if cv < 1:
#             raise Exception('cv must be at least 1.')
#         self.cv = cv
#         if isinstance(n_cores, type(1)) == False:
#             raise Exception('n_cores must be an integer.')
#         if n_cores < 1:
#             raise Exception('n_cores must be at least 1.')
#         self.n_cores = n_cores
#         if set([type(d) for d in dims]) != set([type(1)]):
#             raise Exception('Each dimension in dims must be an integer.')
#         if min(dims) < 0:
#             raise Exception('Each dimension in dims must be non-negative.')
#         if set([x in set([type(1), type(1.0)]) for x in set([type(s) for s in sigmas])]) != set([True]):
#             raise Exception('Each sigma value must be a number.')
#         if min(sigmas) <= 0:
#             raise Exception('Each sigma value must be positive.')
#         if set([x in set([type(1), type(1.0)]) for x in set([type(t) for t in ts])]) != set([True]):
#             raise Exception('Each t value must be a number.')
#         if min(ts) <= 0:
#             raise Exception('Each t value must be positive.')
#         if isinstance(Cs, type([0,1])) == False:
#             raise Exception('Cs must be a list.')
#         if isinstance(epsilons, type([0,1])) == False:
#             raise Exception('epsilons must be a list.')
#         if epsilons != None:
#             parameter_grid = product(dims, sigmas, ts, Cs, epsilons)
#         else:
#             parameter_grid = product(dims, sigmas, ts, Cs)
#         parameter_grid = concatenate([[array(x)] for x in parameter_grid])
#         if epsilons == None:
#             self.models = [SVC(C=parameter_grid[i,3]) for i in range(len(parameter_grid))]
#         else:
#             self.models = [SVR(C=parameter_grid[i,3],epsilon=parameter_grid[i,4]) for i in range(len(parameter_grid))]
#         self.final_model = None
#         self.parameter_grid = parameter_grid
#         self.final_model_kernel = None
#     def __str__(self):
#         if self.parameter_grid.shape[1] == 4:
#             task = 'classification'
#         else:
#             task = 'regression'
#         if self.final_model == None:
#             fit_str = 'Model has not yet been fit.'
#         else:
#             fit_str = 'Model has been fit.'
#         return 'Support vector ' + task + ' object. ' + fit_str
#     def fit(self, X, y):
#         '''Fit the SVM model according to the training data.
        
#         Parameters
#         ----------
#         `X` : {array-like of shape `(n_diagrams, n_diagrams)`} or {list of length `n_diagrams`}
#             Either a precomputed Gram matrix of `n_diagrams` many persistence diagrams (if `precomputed` was set to `True`) or a list of `n_diagrams` many persistence diagrams (otherwise).
#         `y` : array-like of shape `(n_diagrams,)`
#             Target values (class labels in classification, real numbers in regression).

#         Returns
#         -------
#         `self` : object
#             The fitted estimator.

#         Examples
#         --------
#         # DO!
#         '''
#         # make row memberships for cv
#         if self.parameter_grid.shape[1] == 5:
#             1
#         else:
#             1

