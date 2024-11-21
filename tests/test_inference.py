
from tdads.inference import *
from tdads.distance import distance
import pytest
from numpy import array
from numpy.random import random, uniform, normal
from ripser import ripser
from math import pi, cos, sin
from scipy.spatial.distance import cdist

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

def test_diagram_bootstrap_constructor():
    with pytest.raises(Exception, match = 'diag_fun'):
        bs = diagram_bootstrap(diag_fun = 1)
    def f():
        1
    with pytest.raises(Exception, match = 'X and thresh'):
        bs = diagram_bootstrap(diag_fun = f)
    def f(X, thresh):
        1
    with pytest.raises(Exception, match = 'computing'):
        bs = diagram_bootstrap(diag_fun = f)
    def f(X, thresh):
        return ripser(X = X, thresh = thresh, maxdim = 1)
    with pytest.raises(Exception, match = 'num_samples'):
        bs = diagram_bootstrap(diag_fun = f,num_samples = [0,1])
    with pytest.raises(Exception, match = 'alpha'):
        bs = diagram_bootstrap(diag_fun = f, alpha = '0.05')
    with pytest.raises(Exception, match = 'distance_mat'):
        bs = diagram_bootstrap(diag_fun = f, distance_mat = 'False')
    with pytest.raises(Exception, match = 'computing'):
        bs = diagram_bootstrap(diag_fun = f, distance_mat = True)
    
def test_diagram_bootstrap():
    theta = uniform(low = 0, high = 2*pi, size = 100)
    data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
    def f(X, thresh):
        return ripser(X = X, thresh = thresh, maxdim = 1)
    bs = diagram_bootstrap(diag_fun = f, num_samples = 3, dims = [0,1])
    res = bs.compute(X = data, thresh = 2)
    assert len(res['thresholds']) == 2
    assert res['thresholds'][0]*res['thresholds'][1] > 0
    assert len(res['subsetted_diagram']) == 2
    assert len(res['subsetted_diagram'][0]) == 1
    assert res['subsetted_diagram'][0][0,1] == float('inf')
    assert len(res['subsetted_diagram'][1]) == 1
    # now for 2-sphere
    phi = uniform(low = 0, high = 2*pi, size = 100)
    theta = uniform(low = 0, high = pi, size = 100)
    data = array([[sin(theta[i])*cos(phi[i]), sin(theta[i])*sin(phi[i]), cos(theta[i])] for i in range(100)])
    def f(X, thresh):
        return ripser(X = X, thresh = thresh, maxdim = 2)
    bs = diagram_bootstrap(diag_fun = f, num_samples = 3, dims = [0,1,2])
    res = bs.compute(X = data, thresh = 2)
    assert len(res['thresholds']) == 3
    assert res['thresholds'][0]*res['thresholds'][1]*res['thresholds'][2] > 0
    assert len(res['subsetted_diagram']) == 3
    assert len(res['subsetted_diagram'][2]) == 1
    bs = diagram_bootstrap(diag_fun = f, num_samples = 3, dims = [0,2]) # with a subset of the dimensions
    res = bs.compute(X = data, thresh = 2)
    assert len(res['thresholds']) == 2
    assert res['thresholds'][0]*res['thresholds'][1] > 0
    assert len(res['subsetted_diagram']) == 3
    assert len(res['subsetted_diagram'][2]) == 1
    # add test for distance matrix ripser!!
    def f(X, thresh):
        return ripser(X = X, thresh = thresh, maxdim = 2, distance_matrix=True)
    bs = diagram_bootstrap(diag_fun = f, num_samples = 3, dims = [0,2], distance_mat=True)
    with pytest.raises(Exception, match = 'number of rows'):
        res = bs.compute(X = data, thresh = 2)
    D = cdist(XA = data, XB = data, metric = 'euclidean')
    res = bs.compute(X = D, thresh = 2)
    assert len(res['thresholds']) == 2
    assert res['thresholds'][0]*res['thresholds'][1] > 0
    assert len(res['subsetted_diagram']) == 3
    assert len(res['subsetted_diagram'][2]) == 1
    # add test for distance matrix ripser!!
    
def test_universal_null_constructor():
    with pytest.raises(Exception, match = 'diag_fun'):
        un = universal_null(diag_fun = 1)
    def f():
        1
    with pytest.raises(Exception, match = 'X and thresh'):
        un = universal_null(diag_fun = f)
    def f(X, thresh):
        1
    with pytest.raises(Exception, match = 'computing'):
        un = universal_null(diag_fun = f)
    def f(X, thresh):
        return ripser(X = X, thresh = thresh, maxdim = 1)
    with pytest.raises(Exception, match = 'alpha'):
        un = universal_null(diag_fun = f, alpha = '0.05')
    with pytest.raises(Exception, match = 'distance_mat'):
        un = universal_null(diag_fun = f, distance_mat = 'False')
    with pytest.raises(Exception, match = 'computing'):
        un = universal_null(diag_fun = f, distance_mat = True)
    with pytest.raises(Exception, match = 'infinite_cycle_inference'):
        un = universal_null(diag_fun = f, infinite_cycle_inference = 1)
    with pytest.raises(Exception, match = 'dims'):
        un = universal_null(diag_fun = f, dims = '[0,1]')
    with pytest.raises(Exception, match = 'at least 1'):
        un = universal_null(diag_fun = f, dims = [0])

def test_universal_null():
    def f(X, thresh):
        return ripser(X = X, thresh = thresh, maxdim = 1)
    def f_dist(X, thresh):
        return ripser(X = X, thresh = thresh, maxdim = 1, distance_matrix=True)
    un = universal_null(diag_fun = f)
    un_dist = universal_null(diag_fun = f_dist, distance_mat=True)
    un_inf = universal_null(diag_fun = f, infinite_cycle_inference=True)
    with pytest.raises(Exception, match = 'numpy'):
        res = un.compute(X = 1,thresh = 0)
    with pytest.raises(Exception, match = '2-dimensional'):
        res = un.compute(X = array([0,1]),thresh = 0)
    with pytest.raises(Exception, match = '2-dimensional'):
        res = un_dist.compute(X = array([0,1]),thresh = 0)
    with pytest.raises(Exception, match = 'enclosing'):
        res = un_dist.compute(X = array([[0,1],[1,2]]),thresh = 'Enc')
    with pytest.raises(Exception, match = 'positive'):
        res = un_dist.compute(X = array([[0,1],[1,2]]),thresh = 0)
    theta = uniform(low = 0, high = 2*pi, size = 100)
    data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
    data_dist = cdist(data, data, metric = 'euclidean')
    res1 = un.compute(data, thresh = 2)
    res2 = un.compute(data, thresh = 'enclosing')
    res3 = un_dist.compute(data_dist, thresh = 'enclosing')
    assert res1['subsetted_diagram'][0].shape[0] == 0
    assert res2['subsetted_diagram'][0].shape[0] == 0
    assert res3['subsetted_diagram'][0].shape[0] == 0
    # adding noise
    data = data + normal(loc = 0, scale = 0.1, size = (100, 2))
    data_dist = cdist(data, data, metric = 'euclidean')
    res1 = un.compute(data, thresh = 2)
    res2 = un.compute(data, thresh = 'enclosing')
    res3 = un_dist.compute(data_dist, thresh = 'enclosing')
    assert res1['subsetted_diagram'][0].shape[0] == 1
    assert res2['subsetted_diagram'][0].shape[0] == 1
    assert res3['subsetted_diagram'][0].shape[0] == 1
    # now with and without infinite cycle inference
    res4 = un.compute(data, thresh = 1.1)
    assert res4['subsetted_diagram'][0].shape[0] == 0 # failing?
    res5 = un_inf.compute(data, thresh = 1.1)
    assert res5['subsetted_diagram'][0].shape[0] == 1
