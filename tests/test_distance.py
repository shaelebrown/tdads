
# test file for diagram_utils
import pytest
from tdads.distance import *
from numpy import array, inf, equal
from numpy.random import random
from ripser import ripser
import gudhi

def test_distance_class():
    '''Test class constructor and other basic methods.'''
    with pytest.raises(Exception, match = 'The distance metric must be a string.'):
        dist = distance(metric=2)
    with pytest.raises(Exception, match = 'must be either'):
        dist = distance(metric='w')
    with pytest.raises(Exception, match = 'For the wasserstein/bottleneck distance p must be a number.'):
        dist = distance(p='2')
    with pytest.raises(Exception, match = 'For the wasserstein/bottleneck distance p must be at least 1.'):
        dist = distance(p=0.5)
    with pytest.raises(Exception, match = 'For the Fisher information metric sigma must be a number.'):
        dist = distance(metric = 'FIM',sigma='2')
    with pytest.raises(Exception, match = 'For the Fisher information metric sigma must be positive.'):
        dist = distance(metric = 'FIM',sigma=-0.5)
    dist = distance()
    assert dist.__str__() == '2-wasserstein distance.'
    dist = distance(metric = 'FIM', sigma = 2)
    assert dist.__str__() == 'Fisher information metric with parameter sigma = 2.'
    dist = distance(p = float('inf'))
    assert dist.__str__() == 'Bottleneck distance.'
    rips = gudhi.RipsComplex(points=[[0, 0], [1, 0], [0, 1], [1, 1]], max_edge_length=42)
    simplex_tree = rips.create_simplex_tree(max_dimension=1)
    diag1 = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    diag2 = [array([[0,1],[1,2]]),array([[1,2],[3,4]])]
    assert distance().compute(diag1,diag2) == 1 # for now
    with pytest.raises(Exception, match = 'Diagrams must be computed from either the ripser, gph, flagser, gudhi or cechmat libraries.'):
        dist.compute(diag1, 1)