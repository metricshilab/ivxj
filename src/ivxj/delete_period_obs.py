import numpy as np

def delete_period_obs(A, Tlens, h, fromStart=True):
    """
    Delete h observations from the start or end of each submatrix in an unbalanced panel.

    Parameters:
        A: 2D array-like, the full dataset (matrix)
        Tlens: 1D array-like, the length of each submatrix
        h: int, number of periods to delete
        fromStart: bool, if True, delete from the start; if False, delete from the end

    Returns:
        B: 2D array-like, the matrix after deleting h observations
    """
    # Split A into submatrices according to Tlens
    subMatList = split_mat_into_cells(A, Tlens)

    if fromStart:
        # Remove the first h observations from each submatrix
        B = np.vstack([x[h:] for x in subMatList])
    else:
        # Remove the last h observations from each submatrix
        B = np.vstack([x[:-h] for x in subMatList])

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