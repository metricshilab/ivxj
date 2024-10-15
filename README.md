# ivxj

## Introduction

This repository hosts a Python package to implement the IVXJ procedure in 

* Chengwang Liao, Ziwei Mei and Zhentao Shi (2024): "Nickell Meets Stambaugh: A Tale of Two Biases in Panel Predictive Regressions" [[ResearchGate-link]](http://dx.doi.org/10.13140/RG.2.2.35464.64004) [[arXiv-link]](http://arxiv.org/abs/2410.09825)

At this current status, it computes the IVXJ estimates and the corresponding \\(t\\)-statistics for unbalanced panel data under a simple regression specification. It can be used to replicate the empirical application in the paper.

Extensions of multivariate regressions and long-horizon predictions will be subsumed in future versions.

Documentation is provided here [here](https://ivxj.readthedocs.io/en/latest/index.html).

## Installation

```bash
$ pip install git+https://github.com/metricshilab/ivxj.git
```

## Usage

The main function of this package is `ivxj`. It computes the key estimates for unbalanced panel data analysis, including:

- **IVX Estimate**: The estimated coefficients using the IVX method.
- **IVXJ Estimate**: The debiased estimates from the IVXJ method.
- **Standard Error**: The standard errors associated with the estimates.
- **XJ-adjusted \\(\rho\\) Estimate**: The adjusted estimate of \\(\rho\\).

Here's a step-by-step example demonstrating the usage of the `ivxj` package:

```python
import pandas as pd
import numpy as np
import ivxj

# Prepare the input data as a pandas DataFrame
data = pd.DataFrame({
    'id': np.repeat([1, 2], 21),   # Two 'id's, 21 times each
    'time': np.tile(np.arange(1, 22), 2),  # 'time' from 1 to 21 for each 'id'
    'y': np.random.randint(0, 2, 42),  # Random binary values for 'y'
    'x': np.round(np.random.uniform(1, 3, 42), 1)  # Random 'x' values between 1 and 3, rounded to 1 decimal place
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
