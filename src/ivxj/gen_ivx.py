import numpy as np

from ivxj.split_mat_into_cells import split_mat_into_cells

def gen_ivx(x, rhoz, Tlens):
    """
    Generate IVX from unbalanced panel x = (x_1, ..., x_i, ..., x_n)'.
    
    Parameters:
        x: 2D array-like, the full dataset (panel data)
        rhoz: float, parameter for IVX generation
        Tlens: 1D array-like, the length of each submatrix

    Returns:
        zta: 2D array-like, the generated IVX for the full dataset
    """
    # Split x into submatrices according to Tlens
    subMatList = split_mat_into_cells(x, Tlens)
    
    # Generate IVX for each submatrix
    zta = np.concatenate([gen_ivx_for_one_time_series(xt, rhoz) for xt in subMatList])

    return zta


def gen_ivx_for_one_time_series(x, rhoz):
    """
    Generate IVX for one time series (column vector).
    
    Parameters:
        x: 1D array-like, the time series
        rhoz: float, parameter for IVX generation

    Returns:
        z: 1D array-like, the generated IVX for the time series
    """
    T = len(x)

    dx = np.diff(x)
    dx = np.insert(dx, 0, x[0]) # First difference, with dx_1 = x_1
    powers = rhoz ** np.arange(T-1, -1, -1)
  
    z = np.cumsum(powers * dx) / powers

    return z