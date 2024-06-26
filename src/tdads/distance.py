
# functions to calculate distances between diagrams
from tdads.diagram_utils import *

# add in extra parameter for n_cores in distance constructor
class distance:
    def __init__(self, metric='W',p=2, sigma=None):
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
    def __str__(self):
        if self.sigma != None:
            M = 'Fisher information metric with parameter sigma = ' + str(self.sigma) + '.'
        else:
            if self.p == float('inf'):
                M = 'Bottleneck distance.'
            else:
                M = str(self.p) + '-wasserstein distance.'
        return M
    def compute(self,D1,D2):
        '''Compute a single distance value.'''
        D1 = preprocess_diagram(D1)
        D2 = preprocess_diagram(D2)
        return 1
