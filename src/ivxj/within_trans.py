import numpy as np

from ivxj.split_mat_into_cells import split_mat_into_cells

def within_trans(A, Tlens):
    """
    Perform within-group transformation for unbalanced panel data.

    This function applies a within-group transformation to unbalanced panel data by 
    removing individual-specific means from each submatrix.

    Parameters
    ----------
    A : 2D array-like, dtype=float64
        The full dataset, represented as a panel of stacked submatrices (one for each 
        individual).
    Tlens : 1D array-like, dtype=int
        The lengths of each submatrix (individual time series) in the panel.

    Returns
    -------
    B : 2D array-like, dtype=float64
        The dataset after applying the within-group transformation, with 
        individual-specific means removed.
    """

    # Ensure everything is in float64 for consistency
    A = np.array(A, dtype=np.float64)
    
    # Split A into submatrices according to Tlens
    subMatList = split_mat_into_cells(A, Tlens)
    
    # Apply within transformation (subtracting the mean) to each submatrix
    B = np.concatenate([x - np.mean(x, axis=0) for x in subMatList])

    return B




