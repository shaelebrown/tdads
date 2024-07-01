
# test machine learning methods
import pytest
from tdads.machine_learning import *
from tdads.distance import distance
from numpy import array
from numpy.testing import assert_allclose

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
