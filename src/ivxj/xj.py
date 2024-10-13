import numpy as np


def xj(xlong, Tlens):
    """
    Compute the XJ estimate for unbalanced panel data with a single regressor.

    This function applies the X-jackknife method to panel data following an AR(1)
    process, allowing for unbalanced panels where individuals may have different time
    lengths.

    Parameters
    ----------
    xlong : array-like of shape (n_total,), dtype=float64
        Stacked column vector of the regressor for all individuals: (x_1, ..., x_n)'.
    Tlens : array-like of shape (n,), dtype=int
        Vector of individual time lengths: (T_1, ..., T_n).

    Returns
    -------
    rho_hat : float
        The XJ estimate of the AR(1) coefficient rho.
    """
    # Ensure everything is in float64 for consistency
    xlong = np.array(xlong, dtype=np.float64)

    xlong = xlong.flatten()  # must be a long column vector
    Tlens = Tlens.flatten()  # must be a row vector
    n = len(Tlens)

    end_indices = np.cumsum(Tlens)
    start_indices = np.concatenate(([0], end_indices[:-1]))

    numers = []
    denoms = []

    for i in range(n):
        # If the time length of the current cross-section is less than 4, skip to the next
        T = Tlens[i]
        if T <= 20:
            continue

        x = xlong[
            start_indices[i] : end_indices[i]
        ]  # A column vector since xlong is a column vector

        To = T if T % 2 else T - 1  # Max odd number less than or equal to T
        x_first_half = x[: (To - 1) // 2]
        x_first_half_fwd = x[1 : (To + 1) // 2]
        xx_for_correction = np.dot(x_first_half, x_first_half_fwd)
        x0_for_correction = 0.5 * (x[0] * np.sum(x[1:-1:2]) + x[1] * np.sum(x[0:-2:2]))

        if T % 2 != 0:
            numer, denom = xjackk_for_odd_time_len(
                x, xx_for_correction, x0_for_correction
            )
        else:
            # If T is even, delete the first observation
            numer, denom = xjackk_for_odd_time_len(
                x[1:], xx_for_correction, x0_for_correction
            )

        numers.append(numer)
        denoms.append(denom)

    rho_hat = np.sum(numers) / np.sum(denoms)

    return rho_hat


def xjackk_for_odd_time_len(x, xx_for_correction, x0_for_correction):
    """
    Auxiliary function to handle time series with odd lengths.

    This function adjusts calculations when the time series length is odd, using
    correction terms to ensure accurate estimation of the AR(1) coefficient.

    Parameters
    ----------
    x : array-like, dtype=float64
        The time series data.
    xx_for_correction : float
        Correction term applied to adjust for odd-length series in the calculation.
    x0_for_correction : float
        Additional correction term for adjustment in the calculation.

    Returns
    -------
    numer : float
        Numerator used in the calculation of rho_hat.
    denom : float
        Denominator used in the calculation of rho_hat.
    """
    T = len(x)

    # Ensure everything is in float64 for consistency
    x = np.array(x, dtype=np.float64)
    xx_for_correction = np.float64(xx_for_correction)
    x0_for_correction = np.float64(x0_for_correction)

    x_odd = x[0 : T - 2 : 2]
    x_odd_fwd = x[2:T:2]
    x_even = x[1 : T - 1 : 2]

    # Within transformation
    x_odd_tilde = x_odd - np.mean(x_odd, dtype=np.float64)
    x_even_tilde = x_even - np.mean(x_even, dtype=np.float64)

    numer = (
        np.dot(x_odd_tilde, x_even)
        + np.dot(x_even_tilde, x_odd_fwd)
        + 4 / (T - 1) * xx_for_correction
        - 4 / (T - 1) * x0_for_correction
    )
    denom = np.dot(x_odd_tilde, x_odd_tilde) + np.dot(x_even_tilde, x_even_tilde)

    return numer, denom
