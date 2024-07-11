
from tdads.inference import *
from tdads.distance import distance
import pytest
from numpy import array
from numpy.random import random
from ripser import ripser

def test_perm_test_constructor():
    with pytest.raises(Exception, match = 'iterations'):
        pt = perm_test(iterations = None)
    with pytest.raises(Exception, match = 'list'):
        pt = perm_test(dims = 1)
    with pytest.raises(Exception, match = 'dims'):
        pt = perm_test(dims = ['1','2'])
    with pytest.raises(Exception, match = 'p'):
        pt = perm_test(p = -1)
    with pytest.raises(Exception, match = 'q'):
        pt = perm_test(q = 0)
    with pytest.raises(Exception, match = 'paired'):
        pt = perm_test(paired = 0)
    with pytest.raises(Exception, match = 'n_cores'):
        pt = perm_test(n_cores = 1.1)

def test_perm_test():
    D1 = [array([2,3]).reshape((1,2))]
    D2 = [array([[2,3.1],[5,6]])]
    pt = perm_test(iterations = 3)
    res = pt.test([[D1,D2,D1], [D2, D1, D2]])
    assert len(res['p_values']) == 1
    assert len(res['permvals']['0']) == 3
    res2 = pt.test([[D1,D1,D1], [D2, D2]])
    assert res2['test_statistics'][str(0)] == 0
    assert res2['p_values'][str(0)] >= 0.25
    pt3 = perm_test(iterations = 5)
    res3 = pt3.test([[D1,D2,D1], [D2, D1, D2]])
    assert len(res3['permvals']['0']) == 5
    with pytest.raises(Exception, match = 'paired'):
        pt = perm_test(paired = True)
        res = pt.test([[D1,D2,D1], [D2, D1]])
    data1 = random((100,2))
    diagrams1 = ripser(data1)
    data2 = random((100,2))
    diagrams2 = ripser(data2)
    dist = distance(n_cores = 2,dim = 1)
    d = dist.compute(diagrams1, diagrams2)
    pt = perm_test(dims = [1], n_cores = 2, iterations = 1)
    res = pt.test([[diagrams1, diagrams1, diagrams1], [diagrams2, diagrams2, diagrams2]])
    assert res['p_values']['1'] <= 0.5
    res = pt.test([[diagrams1, diagrams1, diagrams1], [diagrams1, diagrams1, diagrams2]])
    assert res['permvals']['1'][0] == (d**2)/3.0
    pt = perm_test(dims = [1], n_cores = 2, iterations = 1, paired = True)
    res = pt.test([[diagrams1, diagrams1, diagrams1], [diagrams1, diagrams1, diagrams2]])
    assert res['permvals']['1'][0] == (d**2)/3.0
    res = pt.test([[diagrams1, diagrams1], [diagrams1, diagrams2], [diagrams2, diagrams2]])
    assert res['permvals']['1'][0] > 0
    
