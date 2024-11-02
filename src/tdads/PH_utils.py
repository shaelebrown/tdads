
from scipy.spatial.distance import cdist
from numpy import ndarray, array, apply_along_axis

# helper functions for persistent homology calculations

def enclosing_radius(X:ndarray, distance_mat:bool = False):
    '''Compute the enclosing radius of a dataset. Beyond this filtration radius no
    topological changes can occur.
    
    Parameters
    ----------
    `X` : numpy.ndarray (2D)
        The input dataset - either raw tabular data or a distance matrix of samples.
    `distance_mat` : bool, default `False`
        Whether or not `X` is a distance matrix. If `False` then a Euclidean distance
        matrix will be computed.

    Returns
    -------
    numpy.float64
        The enclosing radius value of `X`.

    Examples
    --------
    >>> from tdads.PH_utils import enclosing_radius
    >>> from ripser import ripser
    >>> from numpy.random import uniform
    >>> from numpy import array, cos, sin
    >>> from math import pi
    >>> from scipy.spatial.distance import cdist 
    >>> # build circle dataset
    >>> theta = uniform(low = 0, high = 2*pi, size = 100)
    >>> data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
    >>> # compute the enclosing radius
    >>> enc_rad = enclosing_radius(data)
    >>> # compute persistence diagram
    >>> diagram = ripser(data, enc_rad)
    >>> # now for a distance matrix
    >>> dist_data = cdist(data, data, 'euclidean')
    >>> enc_rad = enclosing_radius(dist_data, True)
    >>> diagram = ripser(dist_data, enc_rad, distance_matrix = True)
    '''
    # error check parameters
    if isinstance(distance_mat, type(True)) == False:
            raise Exception('distance_mat must be True or False.')
    if not isinstance(X, type(array([0,1]))):
        raise Exception('X must be a numpy array.')
    if len(X.shape) != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise Exception('X must be a 2-dimensional array with at least two rows and one column.')
    if distance_mat and X.shape[0] != X.shape[1]:
        raise Exception('When distance_mat is True X must have the same number of rows and columns (as a distance matrix).')
    # if X is not a distance matrix, convert it into one
    if not distance_mat:
        X = cdist(XA=X,XB = X,metric = 'euclidean')
    # compute enclosing radius
    enc_rad = apply_along_axis(max, 0, X).min()
    return enc_rad