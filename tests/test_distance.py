
# test file for distance
import pytest
from tdads.distance import *
from numpy import array, inf, equal, empty
from numpy.random import random
from math import sqrt
from ripser import ripser

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
    with pytest.raises(Exception, match = 'n_cores'):
        dist = distance(n_cores=10000)
    dist = distance()
    assert dist.__str__() == '2.0-wasserstein distance.'
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
    D4 = [D1[0],D2[0]]
    d1 = distance(dim = 1)
    assert d1.compute(D1, D2) == 0
    assert d1.compute(D1, D4) > 0
    assert dist_w2.compute(D1,D2) == pytest.approx(sqrt(0.1525))
    assert dist_b.compute(D1,D2) == pytest.approx(0.3)
    assert dist_w2.compute(D3,D1) == pytest.approx(sqrt(0.3125))
    assert dist_b.compute(D3,D1) == pytest.approx(0.5)
    assert dist_w2.compute(D2,D3) == pytest.approx(0.65)
    assert dist_b.compute(D2,D3) == pytest.approx(0.65)
    assert dist_FIM1.compute(D1,D2) == pytest.approx(0.02354624, abs=1e-4)
    assert dist_FIM1.compute(D1,D3) == pytest.approx(0.08821907, abs=1e-4)
    assert dist_FIM1.compute(D3,D2) == pytest.approx(0.08741134, abs=1e-4)
    data1 = random((100,2))
    diagrams1 = ripser(data1)
    data2 = random((100,2))
    diagrams2 = ripser(data2)
    assert dist_b.compute(diagrams1, diagrams2) > 0
    D1 = [array([0, float('inf')]).reshape(1,2)]
    D2 = [array([0, 2]).reshape(1,2)]
    assert dist_b.compute(D1, D2) == 1
    dist_b_2 = distance(p = float('inf'), inf_replace_val = 2)
    assert dist_b_2.compute(D1, D2) == 0
    D1 = [array([0, float('inf')]).reshape(1,2)]
    D2 = [array([0, 2]).reshape(1,2)]
    dist_b_21 = distance(p = float('inf'), inf_replace_val = 2.1)
    assert abs(dist_b_21.compute(D1, D2) - 0.1) < 1e-7
    with pytest.raises(Exception, match = 'inf_replace_val'):
        dist_b_1 = distance(p = float('inf'), inf_replace_val = 1)
        dist_b_1.compute(D1, D2)

def test_distance_matrices():
    '''For distance matrix calculations of all three types.
    Including cross-distance matrix calculations.'''
    D1 = [array([2,3]).reshape((1,2))]
    D2 = [array([[2,3.1],[5,6]])]
    D3 = [array([2,3.1]).reshape((1,2))]
    dist_w2 = distance(n_cores=2)
    dist_w3 = distance(n_cores=3)
    dist_FIM1 = distance(metric = 'FIM', sigma = 1, n_cores = 2)
    m1 = array([[0,dist_w2.compute(D1 = D1, D2 = D2)],[dist_w2.compute(D1 = D2, D2 = D1),0]])
    m2 = array([[0,dist_w3.compute(D1 = D1, D2 = D2),dist_w3.compute(D1 = D1, D2 = D3)],[dist_w3.compute(D1 = D2, D2 = D1),0,dist_w3.compute(D1 = D2, D2 = D3)],[dist_w3.compute(D1 = D3, D2 = D1),dist_w3.compute(D1 = D3, D2 = D2),0]])
    m3 = array([[0,dist_FIM1.compute(D1 = D1, D2 = D3)],[dist_FIM1.compute(D1 = D2, D2 = D1),dist_FIM1.compute(D1 = D2, D2 = D3)]])
    assert (dist_w2.compute_matrix([D1, D2]) == m1).all()
    assert (dist_w3.compute_matrix([D1, D2, D3]) == m2).all()
    assert (dist_FIM1.compute_matrix([D1, D2], [D1, D3]) == m3).all()
    with pytest.raises(Exception, match = 'Diagrams must be computed from either the ripser, gph, flagser, gudhi or cechmat libraries.'):
        dist_w2.compute_matrix([D1, 1])
