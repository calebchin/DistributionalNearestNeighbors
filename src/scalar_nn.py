from nnimputer import NNImputer
import numpy as np
from hyperopt import fmin, Trials, tpe
from hyperopt import hp


class ScalarNN(NNImputer):
  def __init__(
      self,
      nn_type="ii",
      eta_axis=0,
      eta_space=hp.uniform('eta', 0, 1),
      search_algo=tpe.suggest,
      k=None,
      rand_seed=None, ):
    """
    Implementation of nearest neighbors strategy from Li 2019 paper (Nearest Neighbors
    for Matrix Estimation Interpreted as Blind Regression for Latent Variable Model).
    This setting only for datasets of shape N, T, d (i.e. the number of measurements
    per cell is n = 1).
    """
    super().__init__(
            nn_type=nn_type,
            eta_axis=eta_axis,
            eta_space=eta_space,
            search_algo=search_algo,
            k=k,
            rand_seed=rand_seed,
        )
  
  def distances(self, Z, M, *args, **kwargs):
    """
    Computes the row/column dissimilarities between rows/columns in matrix Z
    masked by matrix M

    Parameters:
    -----------
    Z : np.array of shape (N, T, d )
    M : np.array of shape (N, T)

    Returns:
    --------
    dists : np.array of shape (N, N) if nn_type is uu, (T, T) if nn_type is ii
    """
    # apply mask to the original matrix -> if something is not observed, then
    # should not include in the mean calcs
    Z_cp = Z.copy()
    a, b, d = Z_cp.shape
    Z_cp[M != 1] = np.nan 
    if self.nn_type == "ii":
        Z_cp = np.swapaxes(Z_cp, 0, 1)
        M = M.T
    Z_br = Z_cp[:, None] # add extra dim for broadcast operation
    # all row dissims between pairs of rows
    if d == 1:
        dis = (Z_br - Z_cp)**2
        #print(dis.shape)
    else:
        dis = np.linalg.norm((Z_br - Z_cp)**2, axis = -1)
    
    # take mean over the sample dimension (now the 2nd dim)
    mean_row_dis = np.nanmean(dis, axis = 2)
    #print(mean_row_dis.shape)
    # overlap between every
    overlap = np.nansum(M[:, None] * M, axis = 2).astype('float64')
    zs = np.nonzero(overlap == 0)
    overlap[zs] = np.nan
    overlap = overlap[:, :, None]
    
    # rows with 0 overlap will have nan in this matrix
    # entry cannot be own neighbor, so dist is infinite
    mean_ovr = (mean_row_dis / overlap).squeeze()
    np.fill_diagonal(mean_ovr, np.inf)
    #print(mean_ovr.shape)
    return mean_ovr

  
  def estimate(self, Z, M, eta, inds, dists):
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
    Z_cp = Z.copy()
    N, T, d = Z_cp.shape
    Z_cp[M != 1] = np.nan
    
    ests = np.full([N, T], np.nan)

    for i, j in inds:
       
      if self.nn_type == "uu":
          # nn_inds already accounts for no overlap entries
          nn_inds = dists[i] <= eta

          ests = np.full([T, d], np.nan)
          if np.sum(nn_inds & (M[i, :] == 1)) == 0:
             ests[i, j] = np.full(d, -np.inf)
          else:
            ests[i, j] = np.nanmean(Z[nn_inds, j] * (M[nn_inds, j] == 1), axis = 0)
      elif self.nn_type == "ii": 
          nn_inds = dists[j] <= eta 
            # len(nn_inds) x d
          if np.sum(nn_inds & (M[i, :] == 1)) == 0:
            ests[i, j] = np.full(d, -np.inf)
          else:
            ests[i, j] = np.nanmean(Z_cp[i,nn_inds][M[i, nn_inds] == 1], axis = 0)
      else:
        raise ValueError("Invalid nearest neighbors type " + str(self.nn_type))
    return ests
    

  def avg_error(self, ests, truth, inds = None):
    """
    The L2-loss of the estimate

    Parameters:
    -----------
    Z_hat : array of estimates (estimates are d dimensional)
    Z : array of estimates/true values (d dimensional entries)
    """
    return np.nanmean(np.linalg.norm((ests - truth)**2, axis = -1))
