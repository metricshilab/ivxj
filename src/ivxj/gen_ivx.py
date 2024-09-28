import numpy as np

from ivxj.split_mat_into_cells import split_mat_into_cells


def gen_ivx(x, rhoz, Tlens):
    """
    Generate Instrumental Variables for X (IVX) from unbalanced panel data.

    This function generates the IVX tool variables for unbalanced panel data, where the
    dataset consists of multiple submatrices (x_1, ..., x_n) with varying lengths.

    Parameters
    ----------
    x : 2D array-like, dtype=float64
        The full dataset as a panel of stacked submatrices.
    rhoz : float
        Parameter used for generating the IVX.
    Tlens : 1D array-like, dtype=int
        The lengths of each submatrix (individual time series) in the panel.

    Returns
    -------
    zta : 2D array-like, dtype=float64
        The generated IVX tool variables for the full dataset.
    """

    # Split x into submatrices according to Tlens
    subMatList = split_mat_into_cells(x, Tlens)

    # Generate IVX for each submatrix
    zta = np.concatenate([gen_ivx_for_one_time_series(xt, rhoz) for xt in subMatList])

    return zta


def gen_ivx_for_one_time_series(x, rhoz):
    """
    Generate Instrumental Variables for X (IVX) for a single time series.

    This function generates the IVX tool variable for a given time series, based on the
    specified IVX generation parameter.

    Parameters
    ----------
    x : 1D array-like, dtype=float64
        The input time series as a column vector.
    rhoz : float
        Parameter used for generating the IVX.

    Returns
    -------
    z : 1D array-like, dtype=float64
        The generated IVX tool variable for the input time series.
    """

    # Ensure everything is in float64 for consistency
    x = np.array(x, dtype=np.float64)
    rhoz = np.float64(rhoz)

    T = len(x)

    dx = np.diff(x)
    dx = np.insert(dx, 0, x[0])  # First difference, with dx_1 = x_1
    powers = rhoz ** np.arange(T - 1, -1, -1)

    z = np.cumsum(powers * dx) / powers

    return z
