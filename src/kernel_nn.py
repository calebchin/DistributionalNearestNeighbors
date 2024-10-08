from nnimputer import NNImputer
import numpy as np
from hyperopt import fmin, Trials, tpe
from hyperopt import hp


class KernelNN(NNImputer):

    valid_kernels = ["square", "exponential", "linear"]

    def __init__(
        self,
        kernel="exponential",
        nn_type="ii",
        eta_axis=0,
        eta_space=hp.uniform('eta', 0, 1),
        search_algo=tpe.suggest,
        k=None,
        rand_seed=None,
    ):
        """
        Parameters:
        -----------
        kernel : string in valid_kernels
        nn_type : string in ("ii", "uu")
            represents the type of nearest neighbors to use
            "ii" is "item-item" nn, which is column-wise
            "uu" is "user-user" nn, which is row-wise
        eta_axis : integer in [0, 1].
                   Indicates which axis to compute the eta search over. If eta search is
                   done via blocks (i.e. not row-wise or column-wise), then this parameter is ignored
        eta_space : a hyperopt hp search space
                    for example: hp.uniform('eta', 0, 1). If no eta_space is given,
                    then this example will be the default search space.
        search_algo : a hyperopt algorithm
                      for example: tpe.suggest, default is tpe.suggest.
        k : integer > 1, the number of folds in k-fold cross validation over.
            If k = None (default), the LOOCV is used. 
        rand_seed : the random seed to be used for reproducible results. 
                    If None is used (default), then the system time is used (not reproducible)
        """
        if kernel not in self.valid_kernels:
            raise ValueError(
                "{} is not a valid kernel. Currently supported kernels are {}".format(
                    kernel, ", ".join(self.valid_kernels)
                )
            )
        super().__init__(
            nn_type=nn_type,
            eta_axis=eta_axis,
            eta_space=eta_space,
            search_algo=search_algo,
            k=k,
            rand_seed=rand_seed,
        )
        self.kernel = kernel

    def _sqmmd_est2(self, dat1, dat2):
        """
        Computes U-statistics estimate of squared MMD_k, when number of samples from each distribution are different


        Parameters:
        -----------
        dat1 : N x m x d data matrix coming from first d dim distribution
              If dat1 is shape (m, d), then shape (1, m, d) is assumed.
              The first and last dimensions of dat1 and dat2 must be the same.
        dat2 : N x n x d data matrix coming from second d dim distribution
              If dat2 is shape (n, d) then shape (1, n, d) is assumed.
        kernel : k(x, y) that defines MMD_k^2

        Returns:
        --------
        Mean MMD_k^2 estimator between datasets with N entries (N = 1 is a single cell)
        """
        if len(dat1.shape) == 2:
            dat1 = dat1[None, :]
            dat2 = dat2[None, :]

        N, m, d1 = dat1.shape
        N, n, d2 = dat2.shape

        if d1 != d2:
            print("Data dimension do not match!")
            return

        XX = np.matmul(
            dat1, np.transpose(dat1, axes=(0, 2, 1))
        )  # m by m matrix with x_i^Tx_j
        YY = np.matmul(
            dat2, np.transpose(dat2, axes=(0, 2, 1))
        )  # n by n matrix with y_i^Ty_j
        XY = np.matmul(
            dat1, np.transpose(dat2, axes=(0, 2, 1))
        )  # m by n matrix with x_i^Ty_j

        if self.kernel == "linear":
            kXX, kYY, kXY = XX, YY, XY
        if self.kernel == "square":
            kXX, kYY, kXY = (
                (XX + np.ones((m, m))) ** 2,
                (YY + np.ones((n, n))) ** 2,
                (XY + np.ones((m, n))) ** 2,
            )
        if self.kernel == "exponential":
            dXX_mm = np.broadcast_to(np.diagonal(XX.T)[:, None], (N, m, m))
            dYY_nn = np.broadcast_to(np.diagonal(YY.T)[:, None], (N, n, n))
            dXX_mn = np.broadcast_to(np.diagonal(XX.T)[:, :, None], (N, m, n))
            dYY_mn = np.broadcast_to(np.diagonal(YY.T)[:, None], (N, m, n))
            kXX = np.exp(
                -0.5 * (dXX_mm + np.transpose(dXX_mm, axes=(0, 2, 1)) - 2 * XX)
            )
            kYY = np.exp(
                -0.5 * (dYY_nn + np.transpose(dYY_nn, axes=(0, 2, 1)) - 2 * YY)
            )
            kXY = np.exp(-0.5 * (dXX_mn + dYY_mn - 2 * XY))

        val_all = (
            (np.sum(kXX, axis=(1, 2)) - np.sum(np.diagonal(kXX.T), axis=1))
            / (m * (m - 1))
            + (np.sum(kYY, axis=(1, 2)) - np.sum(np.diagonal(kYY.T), axis=1))
            / (n * (n - 1))
            - 2 * np.sum(kXY, axis=(1, 2)) / (n * m)
        )
        val = np.nanmean(val_all)
        if val < 0:
            val = 0
        return val

    def estimate(self, Z, M, eta, inds, dists, ret_nn=False, *args, **kwargs):
        """
        Estimate entries in inds using entries M = 1 and an eta-neighborhood

        Parameters:
        ----------
        Z : np.array of shape (N, T, d)
            The data matrix.
        M : np.array of shape (N, T)
            The missingness/treatment assignment pattern
        eta : the threshold for the neighborhood
        inds : an array-like of indices into Z that will be estimated
        dists : the row/column distances of Z

        Returns:
        --------
        est : an np.array of shape (N, T, d) that consists of the estimates
              at inds.

        """
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
            inp = Z_cp[i, t_nn] if self.nn_type == "ii" else Z_cp[t_nn, j]
            inp_full = Z_cp[i, :] if self.nn_type == "ii" else Z_cp[:, j]
            msk_inp = M[i, t_nn] if self.nn_type == "ii" else M[t_nn, j]
            msk_inp_full = M[i, :] if self.nn_type == "ii" else M[:, j]

            nan_logic = ~np.any(np.isnan(inp), axis=1)
            t_nn_nonan = inp[(msk_inp == 1) & np.all(nan_logic, axis=1)]
            if np.size(t_nn_nonan) > 0:
                est = np.concatenate(t_nn_nonan, axis=0)
                nn_count_rc = t_nn_nonan.shape[0]
                nan_logic_full = ~np.any(np.isnan(inp_full), axis=1)
                if d == 1:
                    nan_logic[:, None]
                list_neighbors_rc = np.nonzero(
                    t_nn & (msk_inp_full == 1) & np.all(nan_logic_full, axis=1)
                )
            else:
                est = np.full([n, d], -np.inf)
                nn_count_rc = 0
                list_neighbors_rc = None
            ests[i][j] = est
            nn_count[i, j] = nn_count_rc
            list_neighbors[i][j] = list_neighbors_rc

        if ret_nn:
            return ests, nn_count, list_neighbors
        return ests

    def distances(self, Z, M, *args, **kwargs):
        """
        Compute the MMD distances between rows/columns

        Parameters:
        -----------
        Z : np.array of shape (N, T, n, d)
        M : np.array of shape (N, T)

        Returns:
        --------
        dists : np.array of shape (N, N) if nn_type is uu, (T, T) if nn_type is ii
        """
        Z_cp = Z.copy()
        N, T, n, d = Z_cp.shape
        Z_cp[M != 1] = np.nan
        diffs = 0
        # # TODO: see if this can be vectorized
        diffs = np.full([T, T], 0.0) if self.nn_type == "ii" else np.full([N, N], 0.0)
        if self.nn_type == "ii":
            for i in range(T - 1):
                for j in range(i + 1, T):
                    col_diffs = self._sqmmd_est2(Z_cp[:, i], Z_cp[:, j])
                    diffs[i, j] = col_diffs
        elif self.nn_type == "uu":
            for i in range(N - 1):
                for j in range(i + 1, N):
                    row_diffs = self._sqmmd_est2(Z_cp[i, :], Z_cp[j, :])
                    diffs[i, j] = row_diffs
        # since MMD_hat is estimate, could be negative. if negative, assume that the MMD distance is 0
        diffs = diffs + diffs.T
        diffs = np.clip(diffs, a_min=0, a_max=None)
        np.fill_diagonal(diffs, np.inf)
        return diffs

    def avg_error(self, ests, truth, inds, *args, **kwargs):
        """
        Returns the average U-statistics estimate of MMD_k^2 distance between
        entries in est and entries in truth

        Parameters:
        ----------
        ests : list of estimated entries (the entries may be of varying size)
        truth : vector of n x d entries (length must be the same as ests)
        inds : a list of indices into ests to compare across to truth

        Returns:
        --------
        err : avg mmd^2 error over len(truth) entries
        """
        err = 0
        num_val = len(inds)
        for val in inds:
            est_val = ests[val]
            if np.any(np.isinf(est_val * -1)):
                # print("No neighbors")
                err_onecell = np.nan
            else:
                err_onecell = self._sqmmd_est2(est_val, truth[val])
            if ~np.isnan(err_onecell):
                err += err_onecell
            else:
                num_val -= 1
        if num_val == 0:
            return np.nan
        return err / num_val
