# test file for distance
import pytest
from tdads.kernel import *
from numpy import array, inf, equal, exp
from numpy.random import random
from numpy.testing import assert_allclose

def test_kernel_class():
    '''Test constructors and str method'''
    with pytest.raises(Exception, match = 'persistence Fisher kernel'):
        k = kernel(t='2')
    with pytest.raises(Exception, match = 'persistence Fisher kernel'):
        k = kernel(t=-1)
    with pytest.raises(Exception, match = 'For the Fisher information metric sigma must be a number.'):
        k = kernel(sigma='2')
    with pytest.raises(Exception, match = 'For the Fisher information metric sigma must be positive.'):
        k = kernel(sigma=-0.5)
    with pytest.raises(Exception, match = 'dim'):
        k = kernel(dim='1')
    with pytest.raises(Exception, match = 'dim'):
        k = kernel(dim=1.1)
    with pytest.raises(Exception, match = 'dim'):
        k = kernel(dim=-1)
    with pytest.raises(Exception, match = 'n_cores'):
        k = kernel(n_cores='1')
    with pytest.raises(Exception, match = 'n_cores'):
        k = kernel(n_cores=1.1)
    with pytest.raises(Exception, match = 'n_cores'):
        k = kernel(n_cores=-1)
    with pytest.raises(Exception, match = 'n_cores'):
        k = kernel(n_cores=10000)
    k = kernel()
    assert k.__str__() == 'Persistence Fisher kernel with sigma = 1, t = 1.'
    dist = distance(metric = 'FIM', sigma = 2)
    
def test_kernel_calculations():
    k = kernel(n_cores = 2)
    D1 = [array([2,3]).reshape((1,2))]
    D2 = [array([[2,3.3],[0,0.5]])]
    D3 = [array([0,0.5]).reshape((1,2))]
    assert k.compute(D1,D2) == pytest.approx(exp(-1*0.02354624), abs=1e-4)
    assert k.compute(D1,D3) == pytest.approx(exp(-1*0.08821907), abs=1e-4)
    assert k.compute(D3,D2) == pytest.approx(exp(-1*0.08741134), abs=1e-4)
    D4 = [array([0, float('inf')]).reshape(1,2)]
    D5 = [array([0, 3.3]).reshape(1,2)]
    assert k.compute(D1,D4) == pytest.approx(0.8827831030254935, abs=1e-4)
    k2 = kernel(n_cores = 2, inf_replace_val=3.3)
    assert k2.compute(D1,D4) == pytest.approx(k.compute(D1, D5), abs=1e-4)

def test_gram_matrices():
    D1 = [array([2,3]).reshape((1,2))]
    D2 = [array([[2,3.1],[5,6]])]
    D3 = [array([2,3.1]).reshape((1,2))]
    k1 = kernel(n_cores = 2)
    k2 = kernel(n_cores = 2,t = 2)
    m1 = array([[1,k1.compute(D1 = D1, D2 = D2)],[k1.compute(D1 = D2, D2 = D1),1]])
    m2 = array([[1,k1.compute(D1 = D1, D2 = D2),k1.compute(D1 = D1, D2 = D3)],[k1.compute(D1 = D2, D2 = D1),1,k1.compute(D1 = D2, D2 = D3)],[k1.compute(D1 = D3, D2 = D1),k1.compute(D1 = D3, D2 = D2),1]])
    m3 = array([[1,k1.compute(D1 = D1, D2 = D3)],[k1.compute(D1 = D2, D2 = D1),k1.compute(D1 = D2, D2 = D3)]])
    m4 = array([[1,k2.compute(D1 = D1, D2 = D3)],[k2.compute(D1 = D2, D2 = D1),k2.compute(D1 = D2, D2 = D3)]])
    assert_allclose(k1.compute_matrix([D1, D2]), m1, atol = 1e-4)
    assert_allclose(k1.compute_matrix([D1, D2, D3]), m2, atol = 1e-4)
    assert_allclose(k1.compute_matrix([D1, D2], [D1, D3]), m3, atol = 1e-4)
    assert_allclose(m3, sqrt(m4), atol = 1e-4)
    with pytest.raises(Exception, match = 'Diagrams must be computed from either the ripser, gph, flagser, gudhi or cechmat libraries.'):
        k1.compute_matrix([D1, 1])
