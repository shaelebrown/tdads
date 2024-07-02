
# inference with persistence diagrams
from tdads.distance import *
from multiprocessing import cpu_count, Pool

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
    def compute_loss(self):
        # check if 
        if isinstance(self.dist_mats[0,0], type(1)) or isinstance(self.dist_mats[0,0], type(1.0)):
            statistics = [sum([(self.dist_mats[d][product(self.diagram_groups[g], self.diagram_groups[g])]**self.q)/(len(self.diagram_groups[g])*(len(self.diagram_groups[g] - 1))) for g in range(self.group_sizes)]) for d in self.dims]
        else:
            precomputed = False
    