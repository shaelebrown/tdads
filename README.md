# tdads

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![JOSS DOI](https://joss.theoj.org/papers/10.21105/joss.06321/status.svg)](https://doi.org/10.21105/joss.06321)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10814141.svg)](https://doi.org/10.5281/zenodo.10814141)

Data science (ds) for topological data analysis (tda) (i.e. tdads = tda+ds).

## Installation

```bash
$ pip install tdads
```

## API

`tdads` has two major modules:

1.  Machine learning. The classes `diagram_mds` and `diagram_kpca` and
    can be used to project a group of diagrams
    into a low dimensional space (i.e. dimension reduction). 
2.  Statistics. The `permutation_test` class can carry out ANOVA-like tests
    for identifying group differences of persistence diagrams. 
    The `diagram_bootstrap` class can be used to identify 
    statistically significant topological features in a dataset.

## Usage

As an example we will 
1. create 10 persistence diagrams from two distinct groups, 
2. describe the significant topological features in each diagram,
3. resolve the two groups with MDS, and 
4. capture the group difference using a permutation test.

```python
from tdads.machine_learning import *
from tdads.inference import *
from numpy.random import uniform
from numpy import array
from math import cos, sin, pi
from ripser import ripser
import matplotlib.pyplot as plt

# function to create a circle dataset and
# compute its diagram
def circle_diagram():
    # sample 100 points from the unit circle
    theta = uniform(low = 0, high = 2*pi, size = 100)
    data = array([[cos(theta[i]), sin(theta[i])] for i in range(100)])
    # compute persistence diagram
    diag = ripser(data, maxdim = 2)
    return [data, diag]

# function to create a sphere dataset and
# compute its diagram
def sphere_diagram():
    # sample 100 points from the unit sphere
    phi = uniform(low = 0, high = 2*pi, size = 100)
    theta = uniform(low = 0, high = pi, size = 100)
    data = array([[sin(theta[i])*cos(phi[i]), sin(theta[i])*sin(phi[i]), cos(theta[i])] for i in range(100)])
    # compute persistence diagram
    diag = ripser(data, maxdim = 2)
    return [data, diag]

# create 10 diagrams, five from circle datasets and
# five from sphere datasets
result = [circle_diagram(), circle_diagram(), circle_diagram(), circle_diagram(), circle_diagram(),
          sphere_diagram(), sphere_diagram(), sphere_diagram(), sphere_diagram(), sphere_diagram()]
data = [r[0] for r in result]
diagrams = [r[1] for r in result]

# use the bootstrap procedure to determine the significant
# topological features in each diagram
def diag_fun(X, thresh):
    return ripser(X = X, thresh = thresh, maxdim = 2)
boot = diagram_bootstrap(diag_fun = diag_fun, dims = [0,1,2], alpha = 0.01)
boot_diagrams = [boot.compute(X = d, thresh = 2) for d in data]

# the subsetted diagrams show that only the first five diagrams have 
# one loop and only the last five diagrams have one void:
for i in range(10):
    print('Num clusters:' + str(len(boot_diagrams[i]['subsetted_diagram'][0])) + ', num loops: ' + str(len(boot_diagrams[i]['subsetted_diagram'][1])) + ', num voids: ' + str(len(boot_diagrams[i]['subsetted_diagram'][2])))

# a 2D MDS projection of the 10 diagrams resolves the two groups:
mds = diagram_mds(dim = 1) # for 1-dimensional homology
emb = mds.fit_transform(diagrams)
plt.scatter(emb[:,0], emb[:,1], color = ['red','red','red','red','red','blue','blue','blue','blue','blue'])
plt.xlabel('Embedding dim 1')
plt.ylabel('Embedding dim 2')
plt.show()

# a permutation test captures the group differences in all dimensions
pt = perm_test(p = float('inf'), iterations = 50, dims = [0,1,2])
res = pt.test([[d for d in diagrams[0:5]], [d for d in diagrams[5:10]]])
res['p_values']
```

## Citation

If you use `tdads`, please consider citing as:

- Brown et al., (2024). TDApplied: An R package for machine learning and inference with persistence diagrams. Journal of Open Source Software, 9(95), 6321, https://doi.org/10.21105/joss.06321

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`tdads` was created by Shael Brown. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`tdads` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
