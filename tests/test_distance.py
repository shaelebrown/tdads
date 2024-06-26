
# test file for diagram_utils
import pytest
from tdads.distance import *
from numpy import array, inf, equal
from numpy.random import random
from ripser import ripser
import gudhi
from math import sqrt

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
    with pytest.raises(Exception, match = 'dim'):
        dist = distance(dim='1')
    with pytest.raises(Exception, match = 'dim'):
        dist = distance(dim=1.1)
    with pytest.raises(Exception, match = 'dim'):
        dist = distance(dim=-1)
    with pytest.raises(Exception, match = 'n_cores'):
        dist = distance(n_cores='1')
    with pytest.raises(Exception, match = 'n_cores'):
        dist = distance(n_cores=1.1)
    with pytest.raises(Exception, match = 'n_cores'):
        dist = distance(n_cores=-1)
    dist = distance()
    assert dist.__str__() == '2-wasserstein distance.'
    dist = distance(metric = 'FIM', sigma = 2)
    assert dist.__str__() == 'Fisher information metric with parameter sigma = 2.'
    dist = distance(p = float('inf'))
    assert dist.__str__() == 'Bottleneck distance.'
    diag = [array([[0,1],[1,2]]),array([[1,2],[3,4]])]
    with pytest.raises(Exception, match = 'Diagrams must be computed from either the ripser, gph, flagser, gudhi or cechmat libraries.'):
        dist.compute(diag, 1)
    
def test_distance_calculations():
    '''For wasserstein, bottleneck, FIM and parallelized distance mats'''
    dist_w2 = distance()
    dist_w3 = distance(p = 3)
    dist_b = distance(p = float('inf'))
    dist_FIM1 = distance(metric = 'FIM', sigma = 1)
    D1 = [array([2,3]).reshape(1,2)]
    D2 = [array([2,3.1,5,6]).reshape(2,2)]
    assert dist_w2.compute(D1,D2) == sqrt(0.1**2 + 0.5**2)
    assert dist_w3.compute(D1,D2) == (0.1**3 + 0.5**3)**(1/3)
    assert dist_b.compute(D1,D2) == 0.5
    assert dist_w2.compute(D2,D1) == sqrt(0.1**2 + 0.5**2)
    assert dist_w3.compute(D2,D1) == (0.1**3 + 0.5**3)**(1/3)
    assert dist_b.compute(D2,D1) == 0.5
    assert dist_w2.compute(D1,D1) == 0
    assert dist_w3.compute(D1,D1) == 0
    assert dist_b.compute(D1,D1) == 0
    assert dist_w2.compute(D2,D2) == 0
    assert dist_w3.compute(D2,D2) == 0
    assert dist_b.compute(D2,D2) == 0
    D2 = [empty((0,2)),array([2,3.1,5,6]).reshape(2,2)]
    assert dist_w2.compute(D1,D2) == 0.5
    assert dist_w3.compute(D1,D2) == 0.5
    assert dist_b.compute(D1,D2) == 0.5
    assert dist_w2.compute(D2,D1) == 0.5
    assert dist_w3.compute(D2,D1) == 0.5
    assert dist_b.compute(D2,D1) == 0.5
    D2 = [array([2,3.1,5,6]).reshape(2,2)]
    assert dist_FIM1.compute(D1,D2) == pytest.approx(dist_FIM1.compute(D2,D1))
    assert dist_FIM1.compute(D1,D1) == 0
    assert dist_FIM1.compute(D2,D2) == 0
    D1 = [array([2,3]).reshape((1,2))]
    D2 = [array([[2,3.3],[0,0.5]])]
    D3 = [array([0,0.5]).reshape((1,2))]
    assert dist_w2.compute(D1,D2) == pytest.approx(sqrt(0.1525))
    assert dist_b.compute(D1,D2) == pytest.approx(0.3)
    assert dist_w2.compute(D3,D1) == pytest.approx(sqrt(0.3125))
    assert dist_b.compute(D3,D1) == pytest.approx(0.5)
    assert dist_w2.compute(D2,D3) == pytest.approx(0.65)
    assert dist_b.compute(D2,D3) == pytest.approx(0.65)
    assert dist_FIM1.compute(D1,D2) == pytest.approx(0.02354624, abs=1e-4)
    assert dist_FIM1.compute(D1,D3) == pytest.approx(0.08821907, abs=1e-4)
    assert dist_FIM1.compute(D3,D2) == pytest.approx(0.08741134, abs=1e-4)

