import numpy as np

def split_mat_into_cells(A, Tlens):
    """
    Splits matrix A into submatrices according to the lengths in Tlens.

    Parameters:
        A: 2D array-like, the full dataset
        Tlens: 1D array-like, the length of each submatrix

    Returns:
        List of submatrices
    """
    # Ensure everything is in float64 for consistency
    A = np.array(A, dtype=np.float64)

    end_indices = np.cumsum(Tlens)
    start_indices = np.concatenate(([0], end_indices[:-1]))
    
    return [A[start:end] for start, end in zip(start_indices, end_indices)]