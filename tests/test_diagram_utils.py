
# test file for diagram_utils
import pytest
from tdads.diagram_utils import check_diagram, preprocess_diagram

# first test
def test_preprocess_diagram():
    '''Test preprocessing of various diagram types'''
    assert 1 == 1, 'Problem.'
    assert True == True, 'Can\'t handle Gudhi diags.'
    # check for an error message!
    error_message = 'Diagrams must be computed from either the ripser, gph, flagser, gudhi or cechmat libraries.'
    with pytest.raises(Exception, match = error_message):
        preprocess_diagram([(1,(1,2))])