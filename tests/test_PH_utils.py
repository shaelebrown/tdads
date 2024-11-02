
from tdads.PH_utils import *
import pytest
from numpy import array
from numpy.random import random, uniform
from math import pi, cos, sin
from scipy.spatial.distance import cdist

def test_enclosing_radius():
    with pytest.raises(Exception, match = 'distance_mat'):
        enc_rad = enclosing_radius(1, 1)
    with pytest.raises(Exception, match = 'numpy array'):
        enc_rad = enclosing_radius(1)
    with pytest.raises(Exception, match = 'two rows'):
        enc_rad = enclosing_radius(array([0,1]))
    with pytest.raises(Exception, match = 'distance matrix'):
        enc_rad = enclosing_radius(array([[0,1],[1,2],[2,3]]), True)
    X = array([[i,i] for i in range(1,11)])
    dist_X = cdist(X, X, 'euclidean')
    enc_rad1 = enclosing_radius(X)
    enc_rad2 = enclosing_radius(dist_X, True)
    assert enc_rad1 == enc_rad2
    assert enc_rad1 == dist_X[0,5]
    theta = uniform(low = 0, high = 2*pi, size = 100)
    data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
    dist_data = cdist(data, data, 'euclidean')
    enc_rad1 = enclosing_radius(data)
    enc_rad2 = enclosing_radius(dist_data, True)
    assert enc_rad1 == enc_rad2