import numpy as np

def within_trans(A, Tlens):
    """
    Perform within transformation for unbalanced panel data.
    
    Parameters:
        A: 2D array-like, the full dataset (panel data)
        Tlens: 1D array-like, the length of each submatrix

    Returns:
        B: 2D array-like, the dataset after within transformation
    """
    # Split A into submatrices according to Tlens
    subMatList = split_mat_into_cells(A, Tlens)
    
    # Apply within transformation (subtracting the mean) to each submatrix
    B = np.vstack([x - np.mean(x, axis=0) for x in subMatList])

    return B


def split_mat_into_cells(A, Tlens):
    """
    Splits matrix A into submatrices according to the lengths in Tlens.

    Parameters:
        A: 2D array-like, the full dataset
        Tlens: 1D array-like, the length of each submatrix

    Returns:
        List of submatrices
    """
    end_indices = np.cumsum(Tlens)
    start_indices = np.concatenate(([0], end_indices[:-1]))
    
    return [A[start:end] for start, end in zip(start_indices, end_indices)]

