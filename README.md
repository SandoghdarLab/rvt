# Radial Variance Transform

This is the Python implementation of the radial variance transform described in the [original paper](https://doi.org/10.1364/OE.420670)

## Installation

This code is available as a PyPi package:
```
pip install imgrvt
```

You can use it as
```python
import imgrvt as rvt

transformed=rvt.rvt(image,rmin=2,rmax=25)
```

The details and application recommendations are described in the [tutorial](https://github.com/SandoghdarLab/rvt/blob/main/docs/tutorial.md).