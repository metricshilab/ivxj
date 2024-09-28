# ivxj

Compute IVXJ estimates for unbalanced panel data under univariate case. You can find the documentation [here](https://ivxj.readthedocs.io/en/latest/index.html).

## Installation

```bash
$ pip install git+https://github.com/metricshilab/ivxj.git
```

## Usage

`ivxj` can be used to compute the following estimates:
- **IVX estimate**
- **IVXJ estimate**
- **Standard error**
- **XJ-adjusted $\rho$ estimate**

Below is an example demonstrating how to use the package:

```python
import ivxj

# Define the input variables
y = dependent_variable       # Dependent variable, a stacked column vector
x = independent_variable     # Regressor, a stacked column vector
rhoz = ivx_parameter         # User-defined IVX parameter (ρ_z)
Tlens = length_vector        # A vector of length $n$, with $T_i$ being the element.

# Call the ivxj function to get the estimates
btaHat, btaHatDebias, se, rhoHat = ivxj.ivxj(y, x, rhoz, Tlens)

# Output:
# - btaHat: IVX estimate
# - btaHatDebias: IVXJ estimate
# - se: Standard error
# - rhoHat: XJ estimate of ρ
```


## License

`ivxj` is contributed by [Ji Pan](https://github.com/PanJi-0) and [Chengwang Liao](https://github.com/cwleo).

It is licensed under the terms of the MIT license.
