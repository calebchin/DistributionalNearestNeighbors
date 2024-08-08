from nnimputer import NNImputer
import numpy as np
from hyperopt import fmin, Trials, tpe


class WassersteinNN(NNImputer):
    def __init__(
        self,
        nn_type="ii",
        eta_axis=0,
        eta_space=None,
        search_algo=tpe.suggest,
        k=None,
        rand_seed=None,
    ):
        """
        Parameters:
        -----------

        """
        super().__init__(
            nn_type=nn_type,
            eta_axis=eta_axis,
            eta_space=eta_space,
            search_algo=search_algo,
            k=k,
            rand_seed=rand_seed,
        )

    def wasserstein2_dist(Y_i, Y_j):
        """
        Returns: 2-Wasserstein^2 distance between two n sample empirical distributions

        Parameters:
        -----------
        Y_i : n x 1 vector
        Y_j : n x 1 vector
        """
        return np.nanmean((Y_i - Y_j) ** 2)

    def avg_wasserstein2_dist(Y_i, Y_j):
        """
        Return the average Wasserstein2^2 distance between two sets of measurements

        Assume that Y_i and Y_j are sorted
        """
        return np.nanmean((np.nanmean((Y_i - Y_j) ** 2, axis=1)), axis=0)

    def estimate(self, Z, M, eta, inds, dists, ret_nn=False, *args, **kwargs):
        """ """
        N, T, n, d = Z.shape
        Z_cp = Z.copy()
        Z_cp[M == 0] = np.nan
        Z_cp[M == 2] = np.nan
        # ii -> dists are cols, avg across row
        # ASSUMPTION: in ii, inds are from a row. in uu, inds are from a col
        # TODO: should be able to relax this
        nn_count = np.full([N, T], np.nan)
        list_neighbors = [[None] * T] * N
        ests = [[None] * T] * N

        # create a table of indices to slice and average over
        neighborhoods = dists <= eta

        # print("in here")

        for i, j in inds:
            t_nn = neighborhoods[j] if self.nn_type == "ii" else neighborhoods[i]
            msk_inp_full = M[i, :] if self.nn_type == "ii" else M[:, j]

            neighbors = Z[i, t_nn] if self.nn_type == "ii" else Z[t_nn, j]
            if np.size(neighbors) > 0:
                est = np.nanmean(neighbors, axis=0).flatten()
                nn_count_rc = neighbors.shape[0]
                list_neighbors_rc = np.nonzero(t_nn & msk_inp_full)
            else:
                est = np.full([n], -np.inf)
                nn_count_rc = 0
                list_neighbors_rc = None

            ests[i, j] = est
            nn_count[i, j] = nn_count_rc
            list_neighbors[i, j] = list_neighbors_rc

            if ret_nn:
                return ests, nn_count, list_neighbors
            return ests

    def distances(self, Z, M, *args, **kwargs):
        """
        Compute the MMD distances between rows/columns
        """
        Z_cp = Z.copy()
        N, T, n, d = Z_cp.shape
        Z_cp[M != 1] = np.nan
        diffs = 0
        # # TODO: see if this can be vectorized
        diffs = np.full([T, T], 0.0) if self.nn_type == "ii" else np.full([N, N], 0.0)
        if self.nn_type == "ii":
            Z_cp = Z_cp.reshape((N, T, n))
            diffs = np.nanmean(
                (Z_cp.swapaxes(0, 1)[:, None, :, :] - Z_cp.swapaxes(0, 1)) ** 2, axis=-1
            )
            diffs = np.nanmean(diffs, axis=-1)
        elif self.nn_type == "uu":
            Z_cp = Z_cp.reshape((N, T, n))
            diffs = np.nanmean((Z_cp[:, None, :, :] - Z_cp) ** 2, axis=-1)
            diffs = np.nanmean(diffs, axis=-1)
        # since MMD_hat is estimate, could be negative. if negative, assume that the MMD distance is 0
        diffs = diffs + diffs.T
        diffs = np.clip(diffs, a_min=0, a_max=None)
        np.fill_diagonal(diffs, np.inf)
        return diffs

    def avg_error(self, ests, truth, inds=None, *args, **kwargs):
        """
        Returns the average U-statistics estimate of MMD_k^2 distance between
        entries in est and entries in truth

        Parameters:
        ----------
        ests : vector of n x d entries
        truth : vector of n x d entries (length must be the same as ests)

        Returns:
        --------
        err : avg mmd^2 error over len(truth) entries
        """
        return np.nanmean((np.nanmean((ests - truth) ** 2, axis=1)), axis=0)
