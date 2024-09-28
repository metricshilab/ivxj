import numpy as np

from ivxj.split_mat_into_cells import split_mat_into_cells

def delete_period_obs(A, Tlens, h, fromStart=True):
    """
    Delete `h` observations from the start or end of each submatrix.

    This function removes the first `h` observations from each submatrix in unbalanced 
    panel data to create a lag effect. If `fromStart` is set to `False`, the function 
    will delete the last `h` observations instead.

    Parameters
    ----------
    A : 1D array-like, dtype=float64
        The full dataset, represented as a stacked 1D array.
    Tlens : 1D array-like, dtype=int
        The lengths of each submatrix (individual time series) in the panel.
    h : int
        The number of observations to delete from each submatrix.
    fromStart : bool, default=True
        If `True`, delete the first `h` observations; if `False`, delete the last `h` observations.

    Returns
    -------
    B : 2D array-like, dtype=float64
        The dataset after deleting `h` observations from each submatrix.
    """

    # Ensure everything is in float64 for consistency
    A = np.array(A, dtype=np.float64)

    # Split A into submatrices according to Tlens
    subMatList = split_mat_into_cells(A, Tlens)

    if fromStart:
        # Remove the first h observations from each submatrix
        B = np.concatenate([x[h:] for x in subMatList])
    else:
        # Remove the last h observations from each submatrix
        B = np.concatenate([x[:-h] for x in subMatList])

    return B