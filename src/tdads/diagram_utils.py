
# activate venv/bin/activate
# pip install --editable .

# TDA modules
# cechmate # not working currently, list of numpy arrays with 2 columns
# gudhi # list of (dim, (birth, death))
# dionysus # not working currently, complicado
# ripser # dict with keys dgms, cocyles, num_edges, dperm2all, idx_perm, r_cover.
# dgms[i] gives the diagram of dimenion i is the numpy array of birth death pairs
# gph # same but with keys dgms and gens
# flagser # same with keys dgms, betti, cell_count, euler.
# phat # doesn't seem to save feature dimensions... let's avoid for now

from numpy import where, append, empty, array

# functions to check persistence diagrams and convert diagrams between formats

# NEED: functions to create empty/sample diagrams from each type, check each diagram type, convert types to DataFrame
def check_diagram(D):
    '''
    '''
    # make sure that birth values are <= death values
    pers = D[:,1] - D[:,0]
    if len(where(pers < 0)) > 0:
        raise Exception('Birth values have to be <= death values.')
    # make sure that birth values are >= 0
    if len(where(D[:,0] < 0)) > 0:
        raise Exception('Birth values have to be non-negative.')

def preprocess_diagram(D, ret = False):
    '''
    '''
    error_message = 'Diagrams must be computed from either the ripser, gph, flagser, gudhi or cechmat libraries.'
    # first check if the diagram is from ripser, gph or flagser
    if isinstance(D, dict):
        if 'dgms' in D.keys() == False:
            raise Exception(error_message)
        if isinstance(D['dgms'],list) == False:
            raise Exception(error_message)
        if set([type(x) for x in D['dgms']]) != set(['ndarray']):
            raise Exception(error_message)
        if set([x.shape[1] for x in D['dgms']]) != set([2]):
            raise Exception(error_message)
        if ret == True:
            # convert to list of numpy arrays
            D = D['dgms']
            lst = [check_diagram(d) for d in D]
            return D
    # now check if diagram is from cechmate or gudhi
    if isinstance(D, list):
        if set([type(x) for x in D]) == set(['tuple']):
            if set([len(x) for x in D]) != set([2]):
                raise Exception(error_message)
            if set([len(x[1]) for x in D]) != set([2]):
                raise Exception(error_message)
            dims = [x[0] for x in D]
            if len(dims < 0) > 0:
                raise Exception(error_message)
            if set([type(d) for d in dims]) != set(['int']):
                raise Exception(error_message)
            # convert to list of numpy arrays
            if ret == True:
                max_dim = max(dims)
                res = [empty((0, 2)) for i in range(max_dim + 1)]
                for pt in D:
                    append(res[pt[0]], array([pt[1][0], pt[1][1]]))
                lst = [check_diagram(d) for d in res]
                return res
        elif set([type(x) for x in D]) == set(['ndarray']):
            if set([x.shape[1] for x in D]) != set([2]):
                raise Exception(error_message)
            # no need to convert as this is the base format
            if ret == True:
                lst = [check_diagram(d) for d in D]
                return D
        else:
            raise Exception(error_message)

    
            
