
# test machine learning methods
import pytest
from tdads.machine_learning import *
from tdads.distance import distance
from numpy import array
from numpy.testing import assert_allclose
from numpy.random import uniform
from math import cos, sin, pi
from ripser import ripser

def test_mds():
    with pytest.raises(Exception, match = 'The distance metric must be a string.'):
        mds = diagram_mds(metric = 2)
    mds_diags = diagram_mds(random_state=1)
    mds_precomp = diagram_mds(precomputed=True, random_state=1)
    D1 = [array([2,3]).reshape(1,2)]
    D2 = [array([2,3.1,5,6]).reshape(2,2)]
    D3 = [array([2,3.1]).reshape(1,2)]
    mds1 = mds_diags.fit_transform([D1, D2, D3])
    dist = distance(n_cores = 2)
    DM = dist.compute_matrix([D1, D2, D3])
    mds2 = mds_precomp.fit_transform(DM)
    assert_allclose(mds1, mds2, atol=1e-6)
    with pytest.raises(Exception, match = 'False'):
        mds_diags.fit_transform(DM)
    with pytest.raises(Exception, match = 'True'):
        mds_precomp.fit_transform([D1, D2, D3])
    # make sure distance matrix is symmetric for document example
    def circle_diagram():
        theta = uniform(low = 0, high = 2*pi, size = 100)
        data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
        diag = ripser(data, maxdim = 2)
        return [data, diag]
    def sphere_diagram():
        phi = uniform(low = 0, high = 2*pi, size = 100)
        theta = uniform(low = 0, high = pi, size = 100)
        data = array([[sin(theta[i])*cos(phi[i]), sin(theta[i])*sin(phi[i]), cos(theta[i])] for i in range(100)])
        diag = ripser(data, maxdim = 2)
        return [data, diag]
    result = [circle_diagram(), circle_diagram(), circle_diagram(), circle_diagram(), circle_diagram(),
              sphere_diagram(), sphere_diagram(), sphere_diagram(), sphere_diagram(), sphere_diagram()]
    data = [r[0] for r in result]
    diagrams = [r[1] for r in result]
    mds = diagram_mds(p = float('inf'), dim = 2) # for 2-dimensional homology
    dist = distance(dim = 2, p = float('inf'))
    D = dist.compute_matrix(diagrams)
    assert_allclose(D, D.T, atol = 1e-7)       

def test_kpca():
    D1 = [array([2,3]).reshape(1,2)]
    D2 = [array([2,3.1,5,6]).reshape(2,2)]
    D3 = [array([2,3.1]).reshape(1,2)]
    kpca = diagram_kpca(random_state = 1, diagrams = [D1, D2, D3], n_cores = 2)
    kpca3 = diagram_kpca(n_components = 3, random_state = 1, diagrams = [D1, D2, D3], n_cores = 2)
    FT = kpca.fit_transform([D1, D2, D3])
    assert_allclose(FT, kpca.fit([D1, D2, D3]).transform([D1, D2, D3]), atol = 1e-4)
    FT3 = kpca3.fit_transform([D1, D2, D3])
    assert_allclose(FT3[:,0:2], FT, atol = 1e-4)
    assert FT3.shape[1] == 3


