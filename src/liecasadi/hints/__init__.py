from typing import Union

import casadi as cs
import numpy.typing as ntp
Vector = Union[cs.DM, cs.MX, ntp.NDArray]
Scalar = Union[cs.DM, cs.MX, float]
Matrix = Union[cs.DM, cs.MX, ntp.NDArray]
Angle = Vector
TangentVector = Vector
