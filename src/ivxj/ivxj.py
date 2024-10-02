import numpy as np
import pandas as pd

from ivxj.split_mat_into_cells import split_mat_into_cells
from ivxj.xjackk_unbalanced import xjackk_unbalanced
from ivxj.delete_period_obs import delete_period_obs
from ivxj.gen_ivx import gen_ivx
from ivxj.within_trans import within_trans

def ivxj(data, rhoz, identity=None, time=None, y_name=None, x_name=None):
    """
    Perform IVXJ estimation on unbalanced panel data in the univariate case.

    This function sorts panel data, extracts dependent and independent variables,
    and applies the IVXJ estimation method, returning the IVX and the IVXJ estimate of 
    coefficient, the standard error, and the XJ estimate of rho.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing the unbalanced panel data.
    rhoz : float
        User-defined IVX parameter (rho_z)
    identity : str, optional
        Column name in `data` representing the individual entity (cross-sectional unit).
        If None, the first column of `data` is used.
    time : str, optional
        Column name in `data` representing the time dimension.
        If None, the second column of `data` is used.
    y_name : str, optional
        Column name in `data` representing the dependent variable.
        If None, the third column of `data` is used.
    x_name : str, optional
        Column name in `data` representing the independent variable.
        If None, the fourth column of `data` is used.

    Returns
    -------
    btaHat : np.ndarray
        IVX estimate of the coefficient.
    btaHatDebias : np.ndarray
        IVXJ estimate the coefficient.
    se : np.ndarray
        Standard error.
    rhoHat : float
        XJ estimate of rho.

    Raises
    ------
    KeyError
        If the specified column names do not exist in the DataFrame.
    ValueError
        If `data` does not have enough columns for the default assignment of variables.

    Examples
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'id': [1, 1, 2, 2],
    ...     'time': [1, 2, 1, 2],
    ...     'y': [2.5, 3.5, 4.5, 5.5],
    ...     'x': [1.1, 1.2, 2.1, 2.2]
    ... })
    >>> rhoz = 0.9
    >>> ivxj(data, rhoz, 'id', 'time', 'y', 'x')
    (np.float64(...),
    np.float64(...),
    np.float64(...),
    np.float64(...))
    """
    # Default to first columns if no identity, time, y, x names are provided
    if identity is None and time is None and y_name is None and x_name is None:
        identity = data.columns[0]
        time = data.columns[1]
        y_name = data.columns[2]
        x_name = data.columns[3]
        
    # Sort the data by identity and time columns
    data_sorted = data.sort_values(by=[identity, time])

    # Extract y (dependent variable) and x (independent variable) as numpy arrays
    y = data_sorted[y_name].to_numpy(dtype=np.float64)
    x = data_sorted[x_name].to_numpy(dtype=np.float64)

    # Group by 'identity' and count occurrences (Tlens is the number of time periods for each entity)
    identity_counts = data_sorted.groupby(identity).size()

    # Convert counts to a numpy array (Tlens)
    Tlens = np.array(identity_counts.values, dtype=int)

    # Call raw_ivxj to perform the IVX-J estimation
    btaHat, btaHatDebias, se, rhoHat = raw_ivxj(y, x, rhoz, Tlens)

    return btaHat, btaHatDebias, se, rhoHat


def raw_ivxj(y, x, rhoz, Tlens):
    """
    Compute IVXJ estimates for unbalanced panel data in the univariate case.

    This function calculates IVXJ estimates using other helper functions in the package.
    It is designed to handle unbalanced panel data with different time lengths for each
    individual.

    Parameters
    ----------
    y : array-like of shape (n_total,), dtype=float64
        Dependent variable, a stacked column vector of all individuals: (y_1, ..., y_n).
    x : array-like of shape (n_total,), dtype=float64
        Regressor, a stacked column vector of all individuals: (x_1, ..., x_n)'.
    rhoz : float
        User-defined IVX parameter (rho_z) for IVX generation.
    Tlens : array-like of shape (n,), dtype=int
        Vector of individual time lengths: (T_1, ..., T_n).

    Returns
    -------
    btaHat : float
        IVX estimate of the coefficient beta.
    btaHatDebias : float
        Debiased IVXJ estimate of the coefficient beta.
    se : float
        Standard error of the estimate.
    rhoHat : float
        XJ estimate of rho.
    """

    # Ensure everything is in float64 for consistency
    y = np.array(y, dtype=np.float64)
    x = np.array(x, dtype=np.float64)
    rhoz = np.float64(rhoz)
    Tlens = Tlens.astype(int)

    # Estimate rho
    rhoHat = xjackk_unbalanced(x, Tlens)

    # Lost one period due to lag
    y = delete_period_obs(y, Tlens, 1)
    xLag = delete_period_obs(x, Tlens, 1, False)
    x = delete_period_obs(x, Tlens, 1)

    # Update Tlens
    Tlens -= 1
    obs_total = np.sum(Tlens)

    # Self-generated instrument
    zLag = gen_ivx(xLag, rhoz, Tlens)

    # Within transformation
    xTilde = within_trans(x, Tlens)
    xLagTilde = within_trans(xLag, Tlens)
    zLagTilde = within_trans(zLag, Tlens)
    yTilde = within_trans(y, Tlens)

    # Denominator ZX
    ZX = np.dot(zLagTilde.flatten(), xLag.flatten())

    # Estimate of beta
    btaHat = np.dot(zLagTilde.flatten(), y.flatten()) / ZX

    # Residuals
    uTilde = yTilde - btaHat * xLagTilde
    vTilde = xTilde - rhoHat * xLagTilde

    # Estimate of omega11
    omg11Hat = np.dot(uTilde.flatten(), uTilde.flatten()) / obs_total

    # Estimate of omega12
    omg12Hat = np.dot(vTilde.flatten(), uTilde.flatten()) / obs_total

    # Estimate of Nickell bias
    lam_seq = (
        (rhoz - rhoz**Tlens) / (1 - rhoz) - (rhoHat - rhoHat**Tlens) / (1 - rhoHat)
    ) / (rhoz - rhoHat)
    b = omg12Hat * np.sum(lam_seq / Tlens) / ZX

    # Standard error of betaHat
    se = np.sqrt(
        omg11Hat
        * (
            np.dot(zLag.flatten(), zLag.flatten())
            - (rhoHat >= 1) * sum_of_mean_sq(zLag, Tlens)
        )
    ) / abs(ZX)

    # Debiased beta
    btaHatDebias = btaHat + b

    return btaHat, btaHatDebias, se, rhoHat


def sum_of_mean_sq(A, Tlens):
    """
    Sum of within mean squares for unbalanced panel.
    """
    subMatList = split_mat_into_cells(A, Tlens)
    B = np.sum([np.mean(x) ** 2 for x in subMatList] * (Tlens**0.95))
    return B
