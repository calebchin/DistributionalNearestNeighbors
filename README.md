# Distributional Nearest Neighbors
Implementation of distributional nearest neighbors with 2-Wasserstein and MMD^2 metrics.

## Algorithms:
**2-Wasserstein NN**: nearest neighbors using the $Wasserstein_2^2$ distance. Implemented in `wasserstein_nn.py` \
**Kernel-NN**: nearest neighbors using maximum mean discrepancy $MMD_k^2$. Currently available kernels: `linear`, `square`, and `exponential` (Gaussian). Implemented in `kernel_nn.py`

## Plug and play:
To use your own metric within our algorithm, create a subclass of `nnimputer.py` and implemented the following functions: \
- `estimate`: given a set of distances, compute the estimated distributions.
- `distances`: compute row/column-wise distributional distances
- `avg_error`: the average error (or distance) between a set of empirical distributions.

## Features coming soon:
- More flexible structures for cross validation (row-wise, col-wise, block-wise, and more)
- Plug and play demonstration with a different metric
- Full documentation and code examples for easy use!
- Complete notebooks for reproducible experiments
