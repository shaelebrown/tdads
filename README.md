# tdads

[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![CRAN version](http://www.r-pkg.org/badges/version/TDApplied)](https://CRAN.R-project.org/package=TDApplied)
[![CRAN Downloads](http://cranlogs.r-pkg.org/badges/grand-total/TDApplied)](https://CRAN.R-project.org/package=TDApplied)

[![JOSS DOI](https://joss.theoj.org/papers/10.21105/joss.06321/status.svg)](https://doi.org/10.21105/joss.06321)
[![Zenodo DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10814141.svg)](https://doi.org/10.5281/zenodo.10814141)

Data science (ds) for topological data analysis (tda) (i.e. tdads = tda+ds).

## Installation

```bash
$ pip install tdads
```

## API

`tdads` has two major modules:

1.  Machine learning. The functions `diagram_mds`, `diagram_kpca` and
    `predict_diagram_kpca` can be used to project a group of diagrams
    into a low dimensional space (i.e. dimension reduction). The
    functions `diagram_kkmeans` and `predict_diagram_kkmeans` can be
    used to cluster a group of diagrams. The functions `diagram_ksvm`
    and `predict_diagram_ksvm` can be used to link, through a prediction
    function, persistence diagrams and an outcome (i.e. dependent)
    variable.
2.  Statistics. The `permutation_test` function acts like an ANOVA test
    for identifying group differences of persistence diagrams. The `bootstrap_persistence_thresholds` function can be used to identify 
    statistically significant topological features in a dataset.

Not only does `tdads` provide methods for the applied analysis of
persistence diagrams which were previously unavailable, but an emphasis
on speed and scalability through parallelization, C code, avoiding
redundant slow computations, etc., makes `tdads` a powerful tool
for carrying out applied analyses of persistence diagrams.

## Usage

- TODO

## Citation

If you use `tdads`, please consider citing as:

- Brown et al., (2024). TDApplied: An R package for machine learning and inference with persistence diagrams. Journal of Open Source Software, 9(95), 6321, https://doi.org/10.21105/joss.06321

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`tdads` was created by Shael Brown. It is licensed under the terms of the GNU General Public License v3.0 license.

## Credits

`tdads` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
