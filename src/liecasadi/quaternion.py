# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import dataclasses
from typing import List

import casadi as cs

from liecasadi.hints import Scalar, Vector


@dataclasses.dataclass
class Quaternion:
    xyzw: Vector

    def __repr__(self) -> str:
        return f"Quaternion: {self.xyzw}"

    def __str__(self) -> str:
        return str(self.xyzw)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(xyzw=Quaternion.product(self.xyzw, other.xyzw))

    def __rmul__(self, other: Scalar) -> "Quaternion":
        """Multiplication with  a scalar

        Returns:
            Quaternion
        """
        return Quaternion(xyzw=other * self.xyzw)

    def __add__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(xyzw=self.xyzw + other.xyzw)

    def __radd__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(xyzw=self.xyzw + other.xyzw)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(xyzw=self.xyzw - other.xyzw)

    def __neg__(self) -> "Quaternion":
        return Quaternion(xyzw=-self.xyzw)

    def __rsub__(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(xyzw=self.xyzw - other.xyzw)

    def __truediv__(self, other: Scalar) -> "Quaternion":
        return Quaternion(xyzw=self.xyzw / other)

    def conjugate(self) -> "Quaternion":
        return Quaternion(xyzw=cs.vertcat(-self.xyzw[:3], self.xyzw[3]))

    def normalize(self) -> "Quaternion":
        xyzw_n = self.xyzw / cs.norm_2(self.xyzw)
        return Quaternion(xyzw=xyzw_n)

    @staticmethod
    def product(q1: Vector, q2: Vector) -> Vector:
        p1 = q1[3] * q2[3] - cs.dot(q1[:3], q2[:3])
        p2 = q1[3] * q2[:3] + q2[3] * q1[:3] + cs.cross(q1[:3], q2[:3])
        return cs.vertcat(p2, p1)

    def cross(self, other: "Quaternion") -> "Quaternion":
        return Quaternion(xyzw=cs.cross(self.xyzw, other.xyzw))

    def coeffs(self) -> Vector:
        return self.xyzw

    @property
    def x(self) -> float:
        return self.xyzw[0]

    @property
    def y(self) -> float:
        return self.xyzw[1]

    @property
    def z(self) -> float:
        return self.xyzw[2]

    @property
    def w(self) -> float:
        return self.xyzw[3]

    def inverse(self):
        return self.conjugate() / cs.dot(self.xyzw, self.xyzw)

    @staticmethod
    def slerp(q1: "Quaternion", q2: "Quaternion", n: Scalar) -> List["Quaternion"]:
        """Spherical linear interpolation between two quaternions
        check https://en.wikipedia.org/wiki/Slerp for more details

        Args:
            q1 (Quaternion): First quaternion
            q2 (Quaternion): Second quaternion
            n (Scalar): Number of interpolation steps

        Returns:
            List[Quaternion]: Interpolated quaternion
        """
        q1 = q1.coeffs()
        q2 = q2.coeffs()
        return [Quaternion.slerp_step(q1, q2, t) for t in cs.np.linspace(0, 1, n)]

    @staticmethod
    def slerp_step(q1: Vector, q2: Vector, t: Scalar) -> Vector:
        """Step for the splerp function

        Args:
            q1 (Vector): First quaternion
            q2 (Vector): Second quaternion
            t (Scalar): Interpolation parameter

        Returns:
            Vector: Interpolated quaternion
        """

        dot = cs.dot(q1, q2)
        angle = cs.acos(dot)
        return Quaternion(
            (cs.sin((1.0 - t) * angle) * q1 + cs.sin(t * angle) * q2) / cs.sin(angle)
        )
