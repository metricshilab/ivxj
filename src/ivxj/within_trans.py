import numpy as np

from ivxj.split_mat_into_cells import split_mat_into_cells

def within_trans(A, Tlens):
    """
    Perform within transformation for unbalanced panel data.
    
    Parameters:
        A: 2D array-like, the full dataset (panel data)
        Tlens: 1D array-like, the length of each submatrix

    Returns:
        B: 2D array-like, the dataset after within transformation
    """
    # Ensure everything is in float64 for consistency
    A = np.array(A, dtype=np.float64)
    
    # Split A into submatrices according to Tlens
    subMatList = split_mat_into_cells(A, Tlens)
    
    # Apply within transformation (subtracting the mean) to each submatrix
    B = np.concatenate([x - np.mean(x, axis=0) for x in subMatList])

    return B




