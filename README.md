# Distributional Nearest Neighbors
Implementation of distributional nearest neighbors with $\text{Wasserstein}_2^2$ distance and $\text{MMD}_k^2$ metrics.


## Algorithms:
**2-Wasserstein NN**: nearest neighbors using the $\text{Wasserstein}_2^2$ distance. Implemented in `wasserstein_nn.py` \
**Kernel-NN**: nearest neighbors using maximum mean discrepancy $\text{MMD}_k^2$. Currently available kernels: `linear`, `square`, and `exponential` (Gaussian). Implemented in `kernel_nn.py`

## Experiments:
- `structure_hs_data.ipynb` details the data cleaning process for the [HeartSteps V1 dataset](https://github.com/klasnja/HeartStepsV1?tab=readme-ov-file).
- `mmd_simulations.ipynb` demonstrates how to use methods on simulated data.

## Plug and play:
To use your own metric with our algorithm, create a subclass of `NNImputer` and implement the following functions:
- `estimate`: given a set of distances, compute the estimated distributions.
- `distances`: compute row/column-wise distributional distances
- `avg_error`: the average error (or distance) between a set of empirical distributions
You can also override `cross_validate` to add your own cross validation strategy