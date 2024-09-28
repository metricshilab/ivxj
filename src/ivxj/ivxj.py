import numpy as np

from ivxj.split_mat_into_cells import split_mat_into_cells
from ivxj.xjackk_unbalanced import xjackk_unbalanced
from ivxj.delete_period_obs import delete_period_obs
from ivxj.gen_ivx import gen_ivx
from ivxj.within_trans import within_trans


def ivxj(y, x, rhoz, Tlens):
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
