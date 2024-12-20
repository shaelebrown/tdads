
# inference with persistence diagrams
from tdads.distance import *
from tdads.PH_utils import enclosing_radius
from multiprocessing import cpu_count, Pool
from numpy import ones, ndarray, percentile, array, float64, log, exp, Inf, where, concatenate, reshape, delete, inf, intersect1d
from numpy.testing import assert_almost_equal
from itertools import repeat, combinations
from random import sample, choices
from inspect import getfullargspec
from scipy.special import digamma
import warnings

# permutation test
class perm_test:
    def __init__(self, iterations:int = 20, dims:list = [0], p:float = 2.0, q:float = 2.0, paired:bool = False, n_cores:int = cpu_count() - 1):
        '''Group difference testing for persistence diagrams. Based on the paper Robinson and Turner 2017, with
        extra functionality detailed in Abdallah 2021.
        
        Parameters
        ----------
        `iterations` : int, default 20
            The number of permutation iterations to carry out. The smallest p-value that can
            be returned by this test is 1/(`iterations` + 1).
        `dims` : list of int, default [0]
            The list of homological dimensions in which to carry out the test.
        `p` : float, default 2.0
            The power parameter for the Wasserstein metric.
        `q` : float, default 2.0
            The loss function exponential parameter, must be at least 1. See Robinson and Turner 2017
            for details.
        `paired` : bool, default `False`
            Whether or not there is a second-order pairing between each group, i.e. if only
            permutations which shuffle the groups first, second, third, etc elements should
            be used. See Abdallah 2021 for details.

        Attributes
        ----------
        `iterations` : int
            The input `iterations` parameter.
        `dims` : list of int
            The input `dims` parameter.
        `p` : float
            The input `p` parameter.
        `q` : float
            The input `q` parameter.
        `paired` : bool
            The input `paired` parameter.
        `distances` : list of `tdads.distance.distance` objects
            The distance functions in each desired dimension.
        `group_sizes` : `None` initially, then list of int once `test` is called
            The integer sizes of the diagram groups provided for testing.
        `dist_mats` : `None` initially, then a list of ndarray's
            A list, with one element for each dimension in `dim`, of partially computed distance matrices between all diagrams in the groups.
            Once two diagrams have there distances calculated in each dimension, the corresponding entries in all elements of `dist_mats` will be updated from -1 to the
            actual distance value in that dimension.
        
        Citations
        ---------
        Robinson T, Turner K (2017). "Hypothesis testing for topological data analysis." https://link.springer.com/article/10.1007/s41468-017-0008-7.

        Abdallah H et al. (2021). "Statistical Inference for Persistent Homology applied to fMRI." https://github.com/hassan-abdallah/Statistical_Inference_PH_fMRI/blob/main/Abdallah_et_al_Statistical_Inference_PH_fMRI.pdf.
        '''
        if isinstance(iterations, type(1)) == False:
            raise Exception('iterations must be an integer.')
        if iterations < 1:
            raise Exception('iterations must be at least 1.')
        self.iterations = iterations
        if not isinstance(dims, type([0,1])):
            raise Exception('dims must be a list.')
        if set([type(d) for d in dims]) != set([type(1)]):
            raise Exception('Each dimension in dims must be an integer.')
        if min(dims) < 0:
            raise Exception('Each dimension in dims must be non-negative.')
        self.dims = dims
        distances = [distance(dim = d, p = p) for d in dims]
        self.distances = distances
        if not isinstance(q,type(2)) and not isinstance(q,type(2.0)):
            raise Exception('q must be a number.')
        if q < 1:
            raise Exception('q must be at least 1.')
        self.q = q
        if isinstance(n_cores, type(1)) == False:
            raise Exception('n_cores must be an integer.')
        if n_cores < 1:
            raise Exception('n_cores must be at least 1.')
        self.group_sizes = None
        self.dist_mats = None
        if isinstance(paired, type(True)) == False:
            raise Exception('paired must be True or False.')
        self.paired = paired
    def __str__(self):
        '''Describe a permutation test procedure based on the number of permutation iterations
        and whether the groups are paired or unpaired.'''
        if self.paired == True:
            start_str = 'Paired permutation test with '
        else:
            start_str = 'Non-paired permutation test with '
        s = start_str + str(self.iterations) + ' iterations.'
        return s
    def compute_loss(self, diagram_groups):
        '''Internal method to compute the loss function from Robinson and Turner 2017.
        This function should not be called directly.'''
        combs = concatenate([concatenate([g*ones((int(len(diagram_groups[g])*(len(diagram_groups[g]) - 1)/2),1)), concatenate([array(x).reshape(1,2) for x in combinations(range(len(diagram_groups[g])), 2)])], axis = 1) for g in range(len(self.group_sizes))]).astype(int)
        def get_distance(dim_ind, combination): # make sure this updates!!
            v = self.dist_mats[dim_ind][diagram_groups[combination[0]][combination[1]]['ind'], diagram_groups[combination[0]][combination[2]]['ind']]
            if v == -1:
                v = self.distances[dim_ind].compute(diagram_groups[combination[0]][combination[1]]['diagram'], diagram_groups[combination[0]][combination[2]]['diagram'])
                self.dist_mats[dim_ind][diagram_groups[combination[0]][combination[1]]['ind'], diagram_groups[combination[0]][combination[2]]['ind']] = v
            return v
        statistics = []
        for d_i in range(len(self.dims)):
            d_tots = [get_distance(d_i, comb) for comb in combs]
            # with Pool(processes=self.n_cores) as pool:
            #     d_tots = pool.starmap(get_distance, zip(repeat(dim), combs)) # each row, need to do this per dim
            for i in range(len(combs)):
                self.dist_mats[d_i][diagram_groups[combs[i, 0]][combs[i, 1]]['ind'], diagram_groups[combs[i, 0]][combs[i, 2]]['ind']] = d_tots[i]
            statistics.append(sum([sum([d_tots[x]**self.q for x in range(len(combs)) if combs[x,0] == g])/(len(diagram_groups[g])*(len(diagram_groups[g]) - 1)) for g in range(len(diagram_groups))]))
        return statistics
    def test(self, diagram_groups):
        '''Run the permutation test.
        
        Parameters
        ----------
        `diagram_groups` : list of lists
            The groups of persistence diagrams to be analyzed.

        Returns
        -------
        Dict
            Keys are 'test_statistics' for the test statistic in each dimension, 
            'permvals' for the null distribution in each dimension and 'p_values' for the
            p-values in each dimension. For example, `output['p_values']['1']` would give the
            p-value for the second homological dimension in `self.dims`.
        
        Examples
        --------
        >>> # create two groups of persistence diagrams
        >>> from ripser import ripser
        >>> import numpy as np
        >>> data1 = np.random((100,2))
        >>> data2 = np.random((100,2))
        >>> D1 = ripser(data1)
        >>> D2 = ripser(data2)
        >>> group1 = [D1, D2]
        >>> group2 = [D1, D2]
        >>> # create perm test object in dimensions 0 and 1
        >>> from tdads.inference import permutation_test
        >>> pt = permutation_test(dims = [0, 1], n_cores = 2)
        >>> # run test
        >>> res = pt.test([g1, g2])
        >>> # get p-values
        >>> res['p_values']

        Citations
        ---------
        Robinson T, Turner K (2017). "Hypothesis testing for topological data analysis." https://link.springer.com/article/10.1007/s41468-017-0008-7.

        Abdallah H et al. (2021). "Statistical Inference for Persistent Homology applied to fMRI." https://github.com/hassan-abdallah/Statistical_Inference_PH_fMRI/blob/main/Abdallah_et_al_Statistical_Inference_PH_fMRI.pdf.
        '''
        # test the diagram_groups and preprocess
        diagram_groups, csum_group_sizes = preprocess_diagram_groups_for_inference(diagram_groups)
        # update group_sizes
        self.group_sizes = [len(g) for g in diagram_groups]
        # more checks
        if min(self.group_sizes) < 2:
            raise Exception('Each group of diagrams must have at least 2 diagrams.')
        if self.paired == True and len(set(self.group_sizes)) > 1:
            raise Exception('When paired is True each group of diagrams must have the same number of elements.')
        # set up to store distance calculations
        n = sum(self.group_sizes)
        self.dist_mats = [-1*ones((n, n)) for d in self.dims]
        # compute test statistics
        test_statistics = self.compute_loss(diagram_groups)
        # generate permutations
        perm_values = [[] for d in self.dims]
        # iterate
        for iteration in range(self.iterations):
            if not self.paired:
                permuted_groups = []
                samples = [i for i in range(n)]
                for i in range(len(diagram_groups)):
                    ss = sample(samples, len(diagram_groups[i]))
                    gs = [max([i for i in range(len(csum_group_sizes)) if csum_group_sizes[i] <= X]) if len([i for i in range(len(csum_group_sizes)) if csum_group_sizes[i] <= X]) > 0 else 0 for X in ss]
                    permuted_groups.append([diagram_groups[gs[j]][ss[j] - csum_group_sizes[gs[j]]] for j in range(len(ss))])
                    samples = [x for x in samples if x not in ss]
            else:
                perm = [sample(range(len(diagram_groups)), len(diagram_groups)) for x in range(self.group_sizes[0])]
                permuted_groups = [[diagram_groups[perm[p][i]][p] for p in range(len(perm))] for i in range(len(diagram_groups))]
            permuted_statistics = self.compute_loss(permuted_groups)
            for i in range(len(self.dims)):
                perm_values[i].append(permuted_statistics[i])
        # gather all data in results
        perm_values_ret = {}
        p_values_ret = {}
        test_statistics_ret = {}
        for i in range(len(self.dims)):
            perm_values_ret[str(self.dims[i])] = perm_values[i]
            test_statistics_ret[str(self.dims[i])] = test_statistics[i]
            p_values_ret[str(self.dims[i])] = (sum(perm_values[i] <= test_statistics[i]) + 1)/(self.iterations + 1)
        return {'permvals':perm_values_ret, 'test_statistics':test_statistics_ret, 'p_values':p_values_ret}
    
class diagram_bootstrap:
    def __init__(self, diag_fun, dims:list = [0], num_samples:int = 20, distance_mat:bool = False, alpha:float = 0.05):
        '''Compute confidence sets for (the topological features within) persistence diagrams.
        
        Parameters
        ----------
        `diag_fun` : function of two variables `X` and `thresh`
            The persistent homology algorithm (Vietoris Rips) for the dataset `X` up to 
            radius `thresh`. The maximum homological dimension should be the same as the
            maximum value of `dims`.
        `dims` : list of int, default [0]
            The list of homological dimensions in which to compute confidence sets.
        `num_samples` : int, default 20
            The number of bootstrap resamplings to carry out.
        `distance_mat` : bool, default False
            Whether the input dataset will be a distance matrix or not.
        `alpha` : float, default 0.05
            The type 1 error for determining significant topological features.

        Attributes
        ----------
        `diag_fun` : function
            The input `diag_fun` parameter.
        `dims` : list
            The input `dims` parameter.
        `num_samples` : int
            The input `num_samples` parameter.
        `distance_mat` : bool, default False
            The input `distance_mat` parameter.
        `alpha` : float
            The input `alpha` parameter.

        Examples
        --------
        >>> from tdads.inference import diagram_bootstrap
        >>> from ripser import ripser
        >>> from numpy.random import uniform
        >>> from numpy import array, cos, sin
        >>> # build circle dataset
        >>> theta = uniform(low = 0, high = 2*pi, size = 100)
        >>> data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
        >>> # define persistent homology function
        >>> def diag_fun(X, thresh):
        >>>     return ripser(X = X, thresh = thresh)
        >>> # create bootstrap object
        >>> boot = diagram_bootstrap(diag_fun = diag_fun)
        
        Citations
        ---------
        Chazal F et al (2017). "Robust Topological Inference: Distance to a Measure and Kernel Distance." https://www.jmlr.org/papers/volume18/15-484/15-484.pdf.
        '''
        def check_fun(a, b):
            return 1
        if not isinstance(diag_fun, type(check_fun)):
            raise Exception('diag_fun must be a function.')
        if getfullargspec(diag_fun)[0] != ['X', 'thresh']:
            raise Exception('diag_fun must be a function of two parameters, X and thresh.')
        if not isinstance(dims, type([0,1])):
            raise Exception('dims must be a list.')
        if set([type(d) for d in dims]) != set([type(1)]):
            raise Exception('Each dimension in dims must be an integer.')
        if min(dims) < 0:
            raise Exception('Each dimension in dims must be non-negative.')
        self.dims = dims

        if isinstance(distance_mat, type(True)) == False:
            raise Exception('distance_mat must be True or False.')
        self.distance_mat = distance_mat

        diamond = array([[0,0],[1,1],[-1,1],[0,2]])
        dist_diamond = array([[0, sqrt(2), sqrt(2), 2], [sqrt(2), 0, 2, sqrt(2)], [sqrt(2), 2, 0, sqrt(2)], [2, sqrt(2), sqrt(2), 0]])
        diamond_diag = [array([[0, sqrt(2)],[0, sqrt(2)],[0, sqrt(2)],[0, float('inf')]]),array([[sqrt(2), 2]])]
        diamond_diag = [diamond_diag[x] if x < len(diamond_diag) else array([]).reshape(0,2).astype(float64) for x in self.dims]
        if not distance_mat:
            X = diamond
        else:
            X = dist_diamond
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample_res = diag_fun(X, float('inf'))
            sample_res = preprocess_diagram(sample_res, ret = True)
        except Exception as ex:
            raise Exception('diag_fun doesn\'t seem to be computing persistence diagrams correctly.')
        for i in self.dims:
            try:
                assert_almost_equal(sample_res[i], diamond_diag[self.dims.index(i)], 7)
            except:
                raise Exception('diag_fun doesn\'t seem to be computing persistence diagrams correctly.')
        self.diag_fun = diag_fun

        if isinstance(num_samples, type(1)) == False:
            raise Exception('num_samples must be an integer.')
        if num_samples < 1:
            raise Exception('num_samples must be at least 1.')
        self.num_samples = num_samples

        if not isinstance(alpha,type(0.05)):
            raise Exception('alpha must be a float.')
        if alpha <= 0 or alpha >= 1:
            raise Exception('alpha must be between 0 and 1 (non-inclusive).')
        self.alpha = alpha
    def __str__(self):
        '''Describe a bootstrap procedure based on the number of bootstrap samples,
        whether or not the input will be a distance matrix and the Type 1 error rate (alpha).'''
        dms = ''
        if not self.distance_mat:
            dms = 'non-'
        s = 'Bootstrap confidence intervals with ' + str(self.num_samples) + ' many samples, ' + '' +  f'{dms}distance-matrix input, and a Type 1 error of ' + str(self.alpha) + '.'
        return s
    def compute(self, X:ndarray, thresh:float):
        '''Carry out the bootstrap procedure.
        
        Parameters
        ----------
        `X` : numpy.ndarray (2D)
            The input dataset - either raw tabular data or a distance matrix of samples.
        `thresh` : float
            The maximum filtration radius for Vietoris-Rips persistent homology.

        Returns
        -------
        Dict
            Entries are 'diagram' (the computed persistence diagram), 'thresholds' (a Dict of the computed
            persistence thresholds for each desired dimension) and 'subsetted_diagram' (the persistence diagram
            thresholded by the threshold values in each dimension).

        Examples
        --------
        >>> from tdads.inference import diagram_bootstrap
        >>> from ripser import ripser
        >>> from numpy.random import uniform
        >>> # build circle dataset
        >>> theta = uniform(low = 0, high = 2*pi, size = 100)
        >>> data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
        >>> # define persistent homology function
        >>> def diag_fun(X, thresh):
        >>>     return ripser(X = X, thresh = thresh)
        >>> # create bootstrap object and compute significant features
        >>> boot = diagram_bootstrap(diag_fun = diag_fun)
        >>> res = boot.compute(data, 2)
        >>> # print subsetted diagram
        >>> res['subsetted_diagram']

        Citations
        ---------
        Chazal F et al (2017). "Robust Topological Inference: Distance to a Measure and Kernel Distance." https://www.jmlr.org/papers/volume18/15-484/15-484.pdf.
        '''
        if not isinstance(X, type(array([0,1]))):
            raise Exception('X must be a numpy array.')
        if len(X.shape) != 2 or X.shape[0] < 2 or X.shape[1] < 1:
            raise Exception('X must be a 2-dimensional array with at least two rows and one column.')
        if self.distance_mat and X.shape[0] != X.shape[1]:
            raise Exception('When distance_mat is True X must have the same number of rows and columns (as a distance matrix).')
        if not (isinstance(thresh, type(1)) or isinstance(thresh, type(0.1))):
            raise Exception('thresh must be a number.')
        if thresh <= 0:
            raise Exception('thresh must be positive.')
        # try to calculate the full persistence diagram
        try:
            diagram = self.diag_fun(X, thresh)
        except Exception as ex:
            raise Exception('An error occured when diag_fun tried to compute the persistence diagram of X. The error was: ' + str(ex))
        # error check the persistence diagram
        try:
            diagram = preprocess_diagram(D = diagram, ret = True)
        except Exception as ex:
            raise Exception('The output of diagam_fun(X, thresh) was not in the correct format for a persistence diagram.')
        # create bottleneck distance objects in each dimension
        distances = [distance(dim = d, p = float('inf'), n_cores = 2) for d in self.dims]
        # function to bootstrap
        def get_distances():
            # generate sample (unique row indices)
            s = choices(population = range(len(X)), k = len(X))
            s = list(set(s))
            # subset X
            if self.distance_mat:
                X_sample = X[s,:][:,s]
            else:
                X_sample = X[s, :]
            # compute new persistence diagram
            try:
                diagram_sample = self.diag_fun(X_sample, thresh)
            except Exception as ex:
                raise Exception('An output of diag_fun was not in the correct format for a persistence diagram.')
            # compute distances in each desired dimension
            res = [dist.compute(diagram, diagram_sample) for dist in distances]
            return res
        # perform bootstrap
        bootstrap_values = [get_distances() for i in range(self.num_samples)]
        # filter by dimension
        bootstrap_values = [array([bv[i] for bv in bootstrap_values]) for i in range(len(self.dims))]
        # compute thresholds
        thresholds = [2*percentile(bv, 1 - self.alpha) for bv in bootstrap_values]
        # subset diagram by thresholds
        subsetted_diagram = [diagram[i][diagram[i][:,1] - diagram[i][:,0] > thresholds[self.dims.index(i)]] if i in self.dims else empty((0, 2)) for i in range(max(self.dims) + 1)]
        # set up return dict
        ret = {'diagram':diagram, 'thresholds':thresholds, 'subsetted_diagram':subsetted_diagram}
        return ret

class universal_null:
    def __init__(self, diag_fun, dims:list = [1], distance_mat:bool = False, alpha:float = 0.05, infinite_cycle_inference:bool = False):
        '''A universal null distribution for (the topological features in) persistence diagrams.
        
        Parameters
        ----------
        `diag_fun` : function of two variables, `X` and `thresh`
            The persistent homology algorithm (Vietoris Rips) for the dataset `X` up to 
            radius `thresh`. The maximum homological dimension should be the same as the
            maximum value of `dims`.
        `dims` : list of int, default [1]
            The list of homological dimensions in which to compute confidence sets.
        `distance_mat` : bool, default False
            Whether the input dataset will be a distance matrix or not.
        `alpha` : float, default 0.05
            The type 1 error for determining significant topological features with the
            Bonferroni correction.
        `infinite_cycle_inference` : bool, default `FALSE`
            Whether or not to perform inference for features with infinite death values. 

        Attributes
        ----------
        `diag_fun` : function
            The input `diag_fun` parameter.
        `dims` : list
            The input `dims` parameter.
        `distance_mat` : bool, default False
            The input `distance_mat` parameter.
        `alpha` : float
            The input `alpha` parameter.
        `infinite_cycle_inference` : bool
            The input `infinite_cycle_inference` parameter.

        Examples
        --------
        >>> from tdads.inference import universal_null
        >>> from tdads.PH_utils import enclosing_radius
        >>> from ripser import ripser
        >>> # define the persistent homology function
        >>> def diag_fun(X):
        >>>     return ripser(X = X, thresh = enclosing_radius(X))
        >>> # create universal null object
        >>> univ_null = universal_null(diag_fun = diag_fun)
        
        Citations
        ---------
        Bobrowski O, Skraba P (2023). "A universal null-distribution for topological data analysis." https://www.nature.com/articles/s41598-023-37842-2.
        '''
        def check_fun(a):
            return 1
        if not isinstance(diag_fun, type(check_fun)):
            raise Exception('diag_fun must be a function.')
        if getfullargspec(diag_fun)[0] != ['X', 'thresh']:
            raise Exception('diag_fun must be a function of two parameters, X and thresh.')

        if not isinstance(dims, type([0,1])):
            raise Exception('dims must be a list.')
        if set([type(d) for d in dims]) != set([type(1)]):
            raise Exception('Each dimension in dims must be an integer.')
        if min(dims) < 1:
            raise Exception('Each dimension in dims must be at least 1.')
        self.dims = dims

        if isinstance(distance_mat, type(True)) == False:
            raise Exception('distance_mat must be True or False.')
        self.distance_mat = distance_mat

        if not isinstance(alpha,type(0.05)):
            raise Exception('alpha must be a float.')
        if alpha <= 0 or alpha >= 1:
            raise Exception('alpha must be between 0 and 1 (non-inclusive).')
        self.alpha = alpha

        diamond = array([[0,0],[1,1],[-1,1],[0,2]])
        dist_diamond = array([[0, sqrt(2), sqrt(2), 2], [sqrt(2), 0, 2, sqrt(2)], [sqrt(2), 2, 0, sqrt(2)], [2, sqrt(2), sqrt(2), 0]])
        diamond_diag = [array([[0, sqrt(2)],[0, sqrt(2)],[0, sqrt(2)],[0, float('inf')]]),array([[sqrt(2), 2]])]
        diamond_diag = [diamond_diag[x] if x < len(diamond_diag) else array([]).reshape(0,2).astype(float64) for x in self.dims]
        if not distance_mat:
            X = diamond
        else:
            X = dist_diamond
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sample_res = diag_fun(X, float('inf'))
            sample_res = preprocess_diagram(sample_res, ret = True)
        except Exception as ex:
            raise Exception('diag_fun doesn\'t seem to be computing persistence diagrams correctly.')
        for i in self.dims:
            try:
                assert_almost_equal(sample_res[i], diamond_diag[self.dims.index(i)], 7)
            except:
                raise Exception('diag_fun doesn\'t seem to be computing persistence diagrams correctly.')
        self.diag_fun = diag_fun

        if isinstance(infinite_cycle_inference, type(True)) == False:
            raise Exception('infinite_cycle_inference must be True or False.')
        self.infinite_cycle_inference = infinite_cycle_inference
    def __str__(self):
        '''Describe a universal null procedure based on the dimensions being analyzed,
        whether or not the input will be a distance matrix, the Type 1 error rate (alpha) and
        whether or not infinite cycle inference will be carried out.'''
        dms = ''
        if not self.distance_mat:
            dms = 'non-'
        if len(self.dims) == 1:
            dim_str = 'dimension ' + str(self.dims[0])
        else:
            dim_str = ', '.join([str(d) for d in self.dims[::-1]]) + ' and ' + str(self.dims[-1])
        if self.distance_mat:
            distmat_str = 'distance-matrix'
        else:
            distmat_str = 'point-cloud'
        if self.infinite_cycle_inference:
            ici_str = ''
        else:
            ici_str = 'no '
        s = f'Universal null procedure for {dim_str}, {distmat_str} input, a Type 1 error rate of {str(self.alpha)} and {ici_str}infinite cycle inference.'
        return s
    def compute(self, X:ndarray, thresh):
        '''Carry out the universal null inference procedure.
        
        Parameters
        ----------
        `X` : numpy.ndarray
            The input dataset - either raw tabular data or a distance matrix of samples.
        `thresh` : float or 'enclosing'
            The maximum filtration radius for persistent homology. If 'enclosing' then the
            enclosing radius of `X` will be computed and used, otherwise `thresh` must be a
            set number.

        Returns
        -------
        Dict
            The entries are 'subsetted_diagram' - the list of subsetted persistence diagrams in each dimension (numpy ndarrays),
            and 'p_values' - a list of lists for the p-values of each remaining topological feature.

        Examples
        --------
        >>> from tdads.inference import universal_null
        >>> from ripser import ripser
        >>> from numpy.random import uniform, normal
        >>> # build circle dataset and add noise
        >>> theta = uniform(low = 0, high = 2*pi, size = 100)
        >>> data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
        >>> data = data + normal(scale = 0.2, size = (100, 2))
        >>> # define the persistent homology function
        >>> def diag_fun(X, thresh):
        >>>     return ripser(X = X, thresh = thresh)
        >>> # create universal null object
        >>> univ_null = universal_null(diag_fun = diag_fun)
        >>> # carry out the inference procedure
        >>> res = univ_null.compute(data)
        
        Citations
        ---------
        Bobrowski O, Skraba P (2023). "A universal null-distribution for topological data analysis." https://www.nature.com/articles/s41598-023-37842-2.
        '''
        # check X and thresh parameters
        if not isinstance(X, type(array([0,1]))):
            raise Exception('X must be a numpy array.')
        if len(X.shape) != 2 or X.shape[0] < 2 or X.shape[1] < 1:
            raise Exception('X must be a 2-dimensional array with at least two rows and one column.')
        if self.distance_mat and X.shape[0] != X.shape[1]:
            raise Exception('When distance_mat is True X must have the same number of rows and columns (as a distance matrix).')
        if not (isinstance(thresh, type(1)) or isinstance(thresh, type(0.1)) or thresh == 'enclosing'):
            raise Exception('thresh must be a number or the string \"enclosing\".')
        if thresh != 'enclosing':
            if thresh <= 0:
                raise Exception('thresh must be positive.')
        else:
            # compute the enclosing radius
            thresh = enclosing_radius(X)
        # try to calculate the full persistence diagram
        try:
            diagram = self.diag_fun(X, thresh)
        except Exception as ex:
            raise Exception('An error occured when diag_fun tried to compute the persistence diagram of X. The error was: ' + str(ex))
        # error check the persistence diagram
        try:
            diagram = preprocess_diagram(D = diagram, ret = True)
        except Exception as ex:
            raise Exception('The output of diagam_fun(X, thresh) was not in the correct format for a persistence diagram.')
        # set up return dict
        ret = {}
        # check if there is any work to be done
        if len(diagram) > 1:
            # subset for diagrams above dimension 0
            diag_highdim = diagram[1::]
            # replace inf values with thresh
            diag_highdim = [where(d == inf,concatenate([reshape(d[:,0], (len(d), 1)), reshape(array([thresh for _ in range(len(d))]), (len(d), 1))], axis = 1),d) for d in diag_highdim]
            # compute test statistics and p-values
            A = 1 # for VR persistent homology
            lambd = -1*digamma(1)
            pi = [diag_sub[:,1]/diag_sub[:,0] for diag_sub in diag_highdim] # ratio of death to birth
            loglog_pi = [log(log(pi_sub)) for pi_sub in pi]
            Lbar = [loglog_pi_sub.mean() for loglog_pi_sub in loglog_pi]
            B = -1*lambd - A*Lbar
            test_statistics = [A*loglog_pi[i] + B[i] for i in range(len(loglog_pi))]
            p_values = [exp(-1*exp(test_statistics_sub)) for test_statistics_sub in test_statistics]
            # determine Bonferroni thresholds in each dimension
            alpha_thresh = [self.alpha/len(diag_sub) if len(diag_sub) > 0 else Inf for diag_sub in diag_highdim]
            # subset the diagram
            subsetted_diagram = [concatenate([reshape(diag_highdim[i][j,:], (1,2)) if p_values[i][j] < alpha_thresh[i] else empty(shape=(0, 2)) for j in range(len(diag_highdim[i]))], axis=0) for i in range(len(diag_highdim))]
            # switch thresh values back to inf
            subsetted_diagram = [where(d == thresh,concatenate([reshape(d[:,0], (len(d), 1)), reshape(array([inf for _ in range(len(d))]), (len(d), 1))], axis = 1),d) for d in subsetted_diagram]
            # perform infinite cycle inference if desired
            if self.infinite_cycle_inference:
                infinite_inds = [where((diag_highdim[i][:,1] == thresh)*(p_values[i] >= alpha_thresh[i]))[0] for i in range(len(diag_highdim))]
                # subset p-values
                p_values = [array([p_values[i][x] for x in where(p_values[i] < alpha_thresh[i])]) for i in range(len(diag_highdim))]
                if sum([len(x) for x in infinite_inds]) > 0:
                    infinite_cycle_inference = [empty(shape = (0,2)) for _ in range(len(infinite_inds))]
                    diag_infinite = [diag_highdim[i][infinite_inds[i],:] for i in range(len(infinite_inds))]
                    alpha_cutoffs = [exp(exp((log(log(1/alpha_thresh[i])) - B[i])/A)) for i in range(len(alpha_thresh))]
                    while sum([len(x) for x in diag_infinite]) > 0:
                        # get minimum birth value of any point
                        min_birth_vals = [min(d[:,0]) for d in diag_infinite]
                        min_birth_val = min(min_birth_vals)
                        min_birth_dim = min_birth_vals.index(min_birth_val)
                        min_birth_ind = where(diag_infinite[min_birth_dim][:,0] == min_birth_val)[0].item()
                        pt = diag_infinite[min_birth_dim][min_birth_ind]
                        diag_infinite[min_birth_dim] = delete(diag_infinite[min_birth_dim],min_birth_ind, axis = 0)
                        # compute new death value
                        new_death = alpha_cutoffs[min_birth_dim]*pt[0]
                        # calculate new diagram
                        new_diag = self.diag_fun(X = X, thresh = new_death)
                        new_diag = preprocess_diagram(new_diag, ret = True)
                        new_diag = new_diag[1::]
                        # check if the point is still there
                        if len(intersect1d(where(new_diag[min_birth_dim][:,0] == pt[0]), where(new_diag[min_birth_dim][:,1] == inf))) > 0:
                            pt[1] = inf
                            pt = reshape(pt, (1,2))
                            infinite_cycle_inference[min_birth_dim] = concatenate([infinite_cycle_inference[min_birth_dim], pt])
                    # update results
                    subsetted_diagram = [concatenate([subsetted_diagram[i], infinite_cycle_inference[i]]) for i in range(len(infinite_cycle_inference))]
                    p_values = [p_values[i] + [None for _ in range(len(infinite_cycle_inference[i]))] for i in range(len(infinite_cycle_inference))]
                else:
                    # subset p-values
                    p_values = [array([p_values[i][x] for x in where(p_values[i] < alpha_thresh[i])[0]]) for i in range(len(diag_highdim))]
            else:
                p_values = [array([p_values[i][x] for x in where(p_values[i] < alpha_thresh[i])[0]]) for i in range(len(diag_highdim))]
        else:
            subsetted_diagram = [empty(shape = (0,2)) for _ in range(len(diagram))]
            p_values = [[] for _ in range(len(diagram))]
        ret['subsetted_diagram'] = subsetted_diagram
        ret['p_values'] = p_values
        return ret
    