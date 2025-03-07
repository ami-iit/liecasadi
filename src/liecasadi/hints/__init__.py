import casadi as cs
import numpy.typing as ntp

Vector = cs.DM | cs.MX | ntp.NDArray
Scalar = cs.DM | cs.MX | float
Matrix = cs.DM | cs.MX | ntp.NDArray
Angle = Vector
TangentVector = Vector
