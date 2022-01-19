# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import dataclasses

import casadi as cs

Vector = cs.MX


@dataclasses.dataclass
class Quaternion:
    xyzw: Vector

    def __repr__(self) -> str:
        return "Quaternion: " + str(self.xyzw)

    def __mul__(self, other) -> "Quaternion":
        return Quaternion(xyzw=Quaternion.product(self.xyzw, other.xyzw))

    def __rmul__(self, other) -> "Quaternion":
        return Quaternion(xyzw=Quaternion.product(other.xyzw, self.xyzw))

    def __add__(self, other: "Quaternion"):
        return Quaternion(xyzw=self.xyzw + other.xyzw)

    def __radd__(self, other: "Quaternion"):
        return Quaternion(xyzw=self.xyzw + other.xyzw)

    def __sub__(self, other: "Quaternion"):
        return Quaternion(xyzw=self.xyzw - other.xyzw)

    def __rsub__(self, other: "Quaternion"):
        return Quaternion(xyzw=self.xyzw - other.xyzw)

    def conjugate(self):
        return Quaternion(xyzw=cs.vertcat(-self.xyzw[:3], self.xyzw[3]))

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
    def x(self):
        return self.xyzw[0]

    @property
    def y(self):
        return self.xyzw[1]

    @property
    def z(self):
        return self.xyzw[2]

    @property
    def w(self):
        return self.xyzw[3]
