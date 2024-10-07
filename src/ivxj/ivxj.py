import numpy as np
import pandas as pd

from ivxj.split_mat_into_cells import split_mat_into_cells
from ivxj.xjackk_unbalanced import xjackk_unbalanced
from ivxj.delete_period_obs import delete_period_obs
from ivxj.gen_ivx import gen_ivx
from ivxj.within_trans import within_trans


def ivxj(data, rhoz, identity=None, time=None, y_name=None, x_name=None):
    """
    IVXJ Estimation for Unbalanced Panel Data (Univariate Case).

    This function performs Instrumental Variable with XJ (IVXJ) estimation on unbalanced panel data in a univariate setting. It sorts the panel data, extracts dependent and independent variables, and applies the IVXJ method. The function returns the IVX and IVXJ estimates of the coefficient, the standard error, and the XJ estimate of rho.

    The method is designed for use in unbalanced panel data, where the number of time periods may differ across individual entities.

    Parameters
    ----------
    data : pandas.DataFrame
        A DataFrame containing unbalanced panel data. It must include columns for an entity identifier, a time variable, and both dependent and independent variables.
    rhoz : float
        A user-defined IVX parameter, denoted as rho_z, controlling the strength of persistence in the instruments.
    identity : str, optional
        The name of the column in `data` representing the individual entity (cross-sectional unit). If None, the first column of `data` is used as the identity column.
    time : str, optional
        The name of the column in `data` representing the time dimension. If None, the second column of `data` is used as the time variable.
    y_name : str, optional
        The name of the column in `data` representing the dependent variable. If None, the third column of `data` is used as the dependent variable.
    x_name : str, optional
        The name of the column in `data` representing the independent variable. If None, the fourth column of `data` is used as the independent variable.

    Returns
    -------
    btaHat : numpy.ndarray
        The IVX estimate of the coefficient.
    btaHatDebias : numpy.ndarray
        The IVXJ estimate of the coefficient.
    se : numpy.ndarray
        The standard error of the IVXJ estimate.
    rhoHat : float
        The XJ estimate of rho.

    Raises
    ------
    KeyError
        If the specified column names for identity, time, y, or x do not exist in the `data` DataFrame.
    ValueError
        If the `data` does not contain enough columns to assign variables when default column indices are used.

    Examples
    --------
    Example 1: Applying IVXJ to an unbalanced panel dataset

    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'id': [1, 1, 1, 2, 2, 2],
    ...     'time': [1, 2, 3, 1, 2, 3],
    ...     'y': [0, 0, 1, 0, 0, 1],
    ...     'x': [1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
    ... })
    >>> rhoz = 0.9
    >>> btaHat, btaHatDebias, se, rhoHat = ivxj(data, rhoz, 'id', 'time', 'y', 'x')
    >>> print(btaHat, btaHatDebias, se, rhoHat)

    Example 2: Using default columns for entity, time, dependent, and independent variables

    >>> ivxj(data, rhoz)

    See Also
    --------
    - `pandas.DataFrame`: Structure used to store the panel data.
    - `numpy.ndarray`: Structure used for the return values.
    - IVX Estimation Methodology (for details on the IVX estimation technique).

    References
    ----------
    For more details on the IVXJ method, see the original paper by [Author et al. (Year)].
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
