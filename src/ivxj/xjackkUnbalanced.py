import numpy as np

def xjackk_unbalanced(xlong, Tlens):
    """
    X-jackknife for panel AR(1), allowing for unbalanced panel.
    
    Parameters:
        xlong: array-like, (x_1, ..., x_i, ..., x_n)'
        Tlens: array-like, (T_1, ..., T_i, ..., T_n)

    Returns:
        rho_hat: float, the estimated rho
    """

    xlong = xlong.flatten()    # must be a long column vector
    Tlens = Tlens.flatten()    # must be a row vector
    n = len(Tlens)

    end_indices = np.cumsum(Tlens)
    start_indices = np.concatenate(([0], end_indices[:-1]))

    numer_sum = 0
    denom_sum = 0

    for i in range(n):
        # If the time length of the current cross-section is less than 4, skip to the next
        T = Tlens[i]
        if T <= 20:
            continue

        x = xlong[start_indices[i]:end_indices[i]]  # A column vector since xlong is a column vector

        To = (T // 2) * 2 - 1  # Max odd number less than T
        x_first_half = x[:(To-1)//2]
        x_first_half_fwd = x[1:(To+1)//2]
        xx_for_correction = np.dot(x_first_half, x_first_half_fwd)
        x0_for_correction = 0.5 * (x[0] * np.sum(x[1::2]) + x[1] * np.sum(x[0::2]))

        if T % 2 != 0:
            numer, denom = xjackk_for_odd_time_len(x, xx_for_correction, x0_for_correction)
        else:
            # If T is even, delete the first observation
            numer, denom = xjackk_for_odd_time_len(x[1:], xx_for_correction, x0_for_correction)

        numer_sum += numer
        denom_sum += denom

    rho_hat = numer_sum / denom_sum

    return rho_hat


def xjackk_for_odd_time_len(x, xx_for_correction, x0_for_correction):
    """
    Auxiliary function to handle odd time lengths.
    
    Parameters:
        x: array-like, the time series
        xx_for_correction: float, correction term
        x0_for_correction: float, correction term

    Returns:
        numer: float, numerator for rho_hat calculation
        denom: float, denominator for rho_hat calculation
    """

    T = len(x)

    x_odd = x[0:T-2:2]
    x_odd_fwd = x[2:T:2]
    x_even = x[1:T-1:2]

    # Within transformation
    x_odd_tilde = x_odd - np.mean(x_odd)
    x_even_tilde = x_even - np.mean(x_even)

    numer = (np.dot(x_odd_tilde, x_even) + np.dot(x_even_tilde, x_odd_fwd)
             + 4/(T-1) * xx_for_correction - 4/(T-1) * x0_for_correction)
    denom = np.dot(x_odd_tilde, x_odd_tilde) + np.dot(x_even_tilde, x_even_tilde)

    return numer, denom