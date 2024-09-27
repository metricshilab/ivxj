# ivxj

Compute IVXJ estimates for unbalanced panel data under univariate case!

## Installation

```bash
$ pip install ivxj
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
Tlens = length_vector        # A vector recording the lengths of the individual column vectors in y and x

# Call the ivxj function to get the estimates
btaHat, btaHatDebias, se, rhoHat = ivxj.ivxj(y, x, rhoz, Tlens)

# Output:
# - btaHat: IVX estimate
# - btaHatDebias: IVXJ estimate
# - se: Standard error
# - rhoHat: XJ estimate of ρ
```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`ivxj` was created by Ji Pan. It is licensed under the terms of the MIT license.

## Credits

`ivxj` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
