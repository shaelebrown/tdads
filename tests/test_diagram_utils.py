
# test file for diagram_utils
import pytest
from tdads.diagram_utils import *
from numpy import array, inf, equal
from numpy.random import random
from ripser import ripser
import gudhi

def test_check_diagram():
    '''Test checking of a diagram'''
    with pytest.raises(Exception, match = 'Birth values have to be <= death values'):
        check_diagram(array([1,0]).reshape((1,2))[0])
    with pytest.raises(Exception, match = 'non-negative'):
        check_diagram(array([-1,1]).reshape((1,2))[0])
    check_diagram(array([0,1,2,3,4,5]).reshape((3,2))[0])
        
def test_preprocess_diagram():
    '''Test preprocessing of various diagram types'''
    # for gudhi
    rips = gudhi.RipsComplex(points=[[0, 0], [1, 0], [0, 1], [1, 1]], max_edge_length=42)
    simplex_tree = rips.create_simplex_tree(max_dimension=1)
    diag = simplex_tree.persistence(homology_coeff_field=2, min_persistence=0)
    assert equal([array([[ 0., inf],[ 0.,  1.],[ 0.,  1.],[ 0.,  1.]])],preprocess_diagram(diag, ret = True)).prod() == 1, 'Can\'t handle gudhi diags.'
    # for cechmate
    cech_diag = [array([[0,1],[1,2]]),array([[1,2],[3,4]])]
    res = preprocess_diagram(cech_diag, ret = True)
    for i in range(len(cech_diag)):
        assert equal(cech_diag[i],res[i]).prod() == 1, 'Can\'t handle cechmate diags.'
    # for ripser, gph or flagser
    data = random((100,2))
    diagrams = ripser(data)
    res = preprocess_diagram(diagrams, ret = True)
    for i in range(len(res)):
        assert equal(diagrams['dgms'][i],res[i]).prod() == 1, 'Can\'t handle ripser diags.'
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram(1)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram(1,ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram('1')
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram('1',ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'a':1})
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'a':1},ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'dgms':1})
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'dgms':1},ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'dgms':[1,2]})
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'dgms':[1,2]},ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'dgms':[array([1,2,3,4])]})
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram({'dgms':[array([1,2,3,4])]},ret=True)
    with pytest.raises(Exception, match = 'have to be'):
        preprocess_diagram({'dgms':[array([-1,2,3,4]).reshape((2,2))]})
    with pytest.raises(Exception, match = 'have to be'):
        preprocess_diagram({'dgms':[array([-1,2,3,4]).reshape((2,2))]},ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([1,2])
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([1,2],ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram(array([1,2]))
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram(array([1,2]),ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([array([1,2,3,4])])
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([array([1,2,3,4])],ret=True)
    with pytest.raises(Exception, match = 'have to'):
        preprocess_diagram([array([-1,2,3,4]).reshape((2,2))])
    with pytest.raises(Exception, match = 'have to'):
        preprocess_diagram([array([-1,2,3,4]).reshape((2,2))],ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,2),(1)])
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,2),(1)],ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,2),(1,2)])
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,2),(1,2)],ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,(0,1)),(-1,(0,1))])
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,(0,1)),(-1,(0,1))],ret=True)
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,(0,1)),(0.5,(0,1))])
    with pytest.raises(Exception, match = 'must be'):
        preprocess_diagram([(1,(0,1)),(0.5,(0,1))],ret=True)
    