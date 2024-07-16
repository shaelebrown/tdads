
from numpy import where, append, array, cumsum, concatenate

# functions to check persistence diagrams and convert diagrams between formats
def check_diagram(D):
    '''Checks for persistence diagrams.
    
    Internal method to verify that birth values are non-negative and less than death values.

    Parameters
    ----------
    `D` : numpy.ndarray
        The input diagram to be checked.
    
    Returns
    -------
    None
    '''
    # make sure that birth values are <= death values
    # if D has one entry
    if len(D.shape) != 1:
        pers = D[:,1] - D[:,0]
        birth = D[:,0]
    else:
        pers = D[1] - D[0]
        birth = D[0]
    if where(pers < 0)[0].shape[0] > 0:
        raise Exception('Birth values have to be <= death values.')
    # make sure that birth values are >= 0
    if where(birth < 0)[0].shape[0] > 0:
        raise Exception('Birth values have to be non-negative.')

def preprocess_diagram(D, inf_replace_val = None, ret = False):
    '''Verify the format of a persistence diagram and convert to a standard format.
    
    This function can verify a persistence diagram from the ripser, gph, flagser, gudhi or cechmate packages 
    and convert any such diagram into a list of numpy arrays if desired (largely an internal functionality but
    can be used in a standalone fashion).
    
    Parameters
    ----------
    `D` : any
        The persistence diagram to be verified. An exception will be raised if `D` is not a persistence
        diagram computed from one of the aforementioned packages.
    `inf_replace_val` : float or int, default `None`
        The value with which `inf` values should be replaced, if desired.
    `ret` : bool, default `False`
        Whether or not to return a processed diagram.
    
    Returns
    -------
    None or list of numpy.ndarray
        If `ret` is `True` and the diagram is verified then a list is returned. The i-th element of 
        the returned list is the array of i-dimensional topological features in the diagram.
    '''
    # error check inf_replace_val
    def check_val(D, inf_replace_val):
        if inf_replace_val != None:
            if not isinstance(inf_replace_val, type(2.0)) and not isinstance(inf_replace_val, type(2)):
                raise Exception('inf_replace_val must be a number.')
            if inf_replace_val <= 0:
                raise Exception('inf_replace_val must be positive.')
            max_death = max([d[d[:,1] != float('inf')].max() if len(d[d[:,1] != float('inf')]) > 0 else 0 for d in D])
            if inf_replace_val < max_death:
                raise Exception('inf_replace_val should be at least as large as any death value in the diagram.')
            # replace vals
            D[0][D[0][:,1] == float('inf'),1] = inf_replace_val
            return D
        else:
            return D
    error_message = 'Diagrams must be computed from either the ripser, gph, flagser, gudhi or cechmat libraries.'
    # first check if the diagram is from ripser, gph or flagser
    if isinstance(D, dict):
        if not 'dgms' in D.keys():
            raise Exception(error_message)
        if not isinstance(D['dgms'],list):
            raise Exception(error_message)
        if set([type(x) for x in D['dgms']]) != set([type(array([0,1]))]):
            raise Exception(error_message)
        if set([len(x.shape) for x in D['dgms']]) != set([2]):
            raise Exception(error_message)
        if set([x.shape[1] for x in D['dgms']]) != set([2]):
            raise Exception(error_message)
        # convert to list of numpy arrays
        D = D['dgms']
        # perform final numeric diagram checks
        lst = [check_diagram(d) for d in D]
        if ret == True:
            return check_val(D, inf_replace_val)
    # now check if diagram is from cechmate or gudhi
    elif isinstance(D, list):
        if set([type(x) for x in D]) == set([type((1,2))]):
            if set([len(x) for x in D]) != set([2]):
                raise Exception(error_message)
            if set([type(x[1]) for x in D]) != set([type((1,2))]):
                raise Exception(error_message)
            if set([len(x[1]) for x in D]) != set([2]):
                raise Exception(error_message)
            dims = [x[0] for x in D]
            if any(e < 0 for e in dims):
                raise Exception(error_message)
            if set([type(d) for d in dims]) != set([type(1)]):
                raise Exception(error_message)
            # convert to list of numpy arrays
            max_dim = max(dims)
            res = [array([0,0]).reshape((1,2)) for i in range(max_dim + 1)]
            for pt in D:
                res[pt[0]] = append(res[pt[0]], array([pt[1][0], pt[1][1]]).reshape((1,2)),axis = 0)
            res = [r[range(1,len(r)),:] for r in res]
            lst = [check_diagram(d) for d in res]
            if ret == True:
                return check_val(res, inf_replace_val)
        elif set([type(x) for x in D]) == set([type(array([0,1]))]):
            if set([len(x.shape) for x in D]) != set([2]):
                raise Exception(error_message)
            if set([x.shape[1] for x in D]) != set([2]):
                raise Exception(error_message)
            # no need to convert as this is the base format
            # final numeric diagram checks
            lst = [check_diagram(d) for d in D]
            if ret == True:
                return check_val(D, inf_replace_val)
        else:
            raise Exception(error_message)
    else:
        raise Exception(error_message)
    
def preprocess_diagram_groups_for_inference(diagram_groups):
    '''Additional internal preprocessing for permutation tests.
    
    Parameters
    ----------
    `diagram_groups` : list of lists
        Contains the persistence diagrams for the permutation test.
    
    Returns
    -------
    List of dicts and list of int
        Each element in the first list contains the diagram (key 'diagram') and its index in the total list (key 'ind').
        The second list contains the cumulative sum of group sizes, starting with 0.
    '''
    if not isinstance(diagram_groups, type([0,1])):
        raise Exception('diagram_groups must be a list.')
    if set([type(x) for x in diagram_groups]) != set([type([0,1])]):
        raise Exception('Each element of diagram_groups must be a list.')
    group_sizes = [len(g) for g in diagram_groups]
    csum_group_sizes = concatenate([array([0]), cumsum(group_sizes)])
    diagram_groups = [[{'diagram':preprocess_diagram(diagram_groups[g][i], ret = True), 'ind':csum_group_sizes[g] + i} for i in range(len(diagram_groups[g]))] for g in range(len(diagram_groups))]
    return diagram_groups, csum_group_sizes