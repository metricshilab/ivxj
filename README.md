# ivxj

## Introduction

This repository hosts a Python package to implement the IVXJ procedure in 

* Liao, Mei and Shi (2024): "Nickell Meets Stambaugh: A Tale of Two Biases in Panel Predictive Regressions" [arXiv-link]

At this current status, it computes the IVXJ estimates and the corresponding $t$-statistics for unbalanced panel data under a simple regression specification. It is sufficient to replicate the empirical application in the paper.

Multivariate regression and long-horizon predictions will be subsumed in future versions.

Documentation is provided here [here](https://ivxj.readthedocs.io/en/latest/index.html).

## Installation

```bash
$ pip install git+https://github.com/metricshilab/ivxj.git
```

## Usage

The main function of this package is `ivxj`. It compute key estimates for unbalanced panel data analysis, including:

- **IVX Estimate**: The estimated coefficients using the IVX method.
- **IVXJ Estimate**: The debiased estimates from the IVXJ method.
- **Standard Error**: The standard errors associated with the estimates.
- **XJ-adjusted $\rho$ Estimate**: The adjusted estimate of $\rho$, reflecting the effects of the instruments.

Here's a step-by-step example demonstrating how to use the `ivxj` package:

```python
import pandas as pd
import ivxj

# Prepare your input data as a pandas DataFrame
data = pd.DataFrame({
    'id': [1, 1, 1, 2, 2, 2],
    'time': [1, 2, 3, 1, 2, 3],
    'y': [0, 0, 1, 0, 0, 1],
    'x': [1.1, 1.2, 1.3, 2.1, 2.2, 2.3]
})

# Define the user-defined IVX parameter (rho_z)
rhoz = 0.9

# Optional: Specify column names for identity, time, y, and x
identity = 'id'  # Column representing individual entities
time = 'time'    # Column representing time periods
y_name = 'y'     # Column representing the dependent variable
x_name = 'x'     # Column representing the independent variable

# Call the ivxj function to compute the estimates
btaHat, btaHatDebias, se, rhoHat = ivxj.ivxj(data, rhoz, identity, time, y_name, x_name)

# Output
print("IVX Estimate (btaHat):", btaHat)
print("IVXJ Estimate (btaHatDebias):", btaHatDebias)
print("Standard Error (se):", se)
print("XJ-adjusted rho Estimate (rhoHat):", rhoHat)
```

## License

`ivxj` is contributed by [Ji Pan](https://github.com/PanJi-0) and [Chengwang Liao](https://github.com/cwleo).

It is licensed under the terms of the MIT license.
