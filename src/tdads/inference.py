
# inference with persistence diagrams
from tdads.distance import *
from multiprocessing import cpu_count, Pool
from numpy import ones
from itertools import repeat
from random import sample

# permutation test
class perm_test:
    def __init__(self, iterations:int = 20, dims:list = [0], p:float = 2.0, q:float = 2.0, n_cores:int = cpu_count() - 1):
        if isinstance(iterations, type(1)) == False:
            raise Exception('iterations must be an integer.')
        if iterations < 1:
            raise Exception('iterations must be at least 1.')
        self.iterations = iterations
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
        self.n_cores = n_cores
        self.group_sizes = None
        self.dist_mats = None
        self.paired = False
    def __str__(self):
        if self.paired == True:
            start_str = 'Paired permutation test with '
        else:
            start_str = 'Non-paired permutation test with '
        s = start_str + str(self.iterations) + ' iterations.'
        return s
    def compute_loss(self, diagram_groups):
        # check if 
        if isinstance(self.dist_mats[0,0], type(1)) or isinstance(self.dist_mats[0,0], type(1.0)):
            statistics = [sum([(self.dist_mats[d][product(diagram_groups[g], diagram_groups[g])]**self.q)/(len(diagram_groups[g])*(len(diagram_groups[g] - 1))) for g in range(self.group_sizes)]) for d in self.dims]
        else:
            combinations = concatenate([concatenate([g*ones((len(diagram_groups[g]),1)), concatenate([x for x in product(diagram_groups[g], diagram_groups[g])])], axis = 1) for g in range(self.group_sizes)])
            def get_distance(dim, combination): # make sure this updates!!
                v = self.dist_mats[dim][diagram_groups[combination[0]][combination[1]]['ind'], diagram_groups[combination[0]][combination[2]]['ind']]
                if v == -1:
                    v = self.distances[dim].compute(diagram_groups[combination[0]][combination[1]]['diag'], diagram_groups[combination[0]][combination[2]]['diag'])
                    self.dist_mats[dim][diagram_groups[combination[0]][combination[1]]['ind'], diagram_groups[combination[0]][combination[2]]['ind']] = v
                return v
            statistics = []
            for dim in self.dims:
                with Pool(processes=self.n_cores) as pool:
                    d_tots = pool.starmap(get_distance, zip(repeat(dim), combinations)) # each row, need to do this per dim
                for i in range(len(combinations)):
                    self.dist_mats[dim][diagram_groups[combinations[i, 0]][combinations[i, 1]]['ind'], diagram_groups[combinations[i, 0]][combinations[i, 2]]['ind']] = d_tots[i]
                statistics.append = sum([(d_tots[combinations[:,0] == g]**self.q)/(len(diagram_groups[g])*(len(diagram_groups[g] - 1))) for g in range(self.group_sizes)])
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
        if not isinstance(diagram_groups, type([0,1])):
            raise Exception('diagram_groups must be a list.')
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
        dist_mats = [-1*ones((n, n)) for d in self.dims]
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
                    gs = [min(csum_group_sizes >= X['ind']) for X in ss]
                    permuted_groups.append([diagram_groups[gs[j]][ss[j]['ind'] - csum_group_sizes[gs[j]]] for j in ss])
                    samples = [x for x in samples if x not in ss]
            else:
                perm = [sample(range(len(diagram_groups)), len(diagram_groups)) for x in range(set(self.group_sizes)[0])]
                permuted_groups = [[diagram_groups[i][p] for p in perm[i]] for i in range(len(diagram_groups))]
            permuted_statistics = self.compute_loss(permuted_groups)
            for i in range(len(self.dims)):
                perm_values[i].append(permuted_statistics[i])
        # gather all data in results
        perm_values_ret = {}
        p_values_ret = {}
        test_statistics_ret = {}
        for i in range(len(self.dims)):
            perm_values_ret[str(i)] = perm_values[i]
            test_statistics_ret[str(i)] = test_statistics[i]
            p_values_ret[str(i)] = (sum(perm_values[i] <= test_statistics[i]) + 1)/(self.iterations + 1)
        return {'permvals':perm_values_ret, 'test_statistics':test_statistics_ret, 'p_values':p_values_ret}
        

        
    