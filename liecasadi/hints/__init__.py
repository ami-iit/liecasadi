from typing import Union
import numpy.typing as ntp
import casadi as cs

Vector = Union[ntp.NDArray, cs.MX]
Scalar = Union[float, cs.MX]
Matrix = Union[ntp.NDArray, cs.MX]
Angle = Vector
TangentVector = Vector
