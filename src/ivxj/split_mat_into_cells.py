import numpy as np

def split_mat_into_cells(A, Tlens):
    """
    Split matrix A into submatrices based on the lengths specified in Tlens.

    This function divides the full dataset into individual submatrices, where the size 
    of each submatrix is determined by the corresponding values in Tlens.

    Parameters
    ----------
    A : 2D array-like, dtype=float64
        The full dataset to be split into submatrices.
    Tlens : 1D array-like, dtype=int
        The lengths of each submatrix, indicating how many rows to include in each 
        split.

    Returns
    -------
    list of 2D array-like
        A list of submatrices split from the full dataset according to Tlens.
    """

    # Ensure everything is in float64 for consistency
    A = np.array(A, dtype=np.float64)

    end_indices = np.cumsum(Tlens)
    start_indices = np.concatenate(([0], end_indices[:-1]))
    
    return [A[start:end] for start, end in zip(start_indices, end_indices)]