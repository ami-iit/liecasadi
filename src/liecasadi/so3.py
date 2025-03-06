# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import dataclasses
from dataclasses import field

import casadi as cs
import numpy as np

from liecasadi import Quaternion
from liecasadi.hints import Angle, Matrix, Scalar, TangentVector, Vector


@dataclasses.dataclass
class SO3:
    xyzw: Vector
    quat: Quaternion = field(init=False)

    def __post_init__(self) -> None:
        self.quat = Quaternion(xyzw=self.xyzw)

    def __repr__(self) -> str:
        return f"SO3 quaternion: {self.quat.coeffs()}"

    @staticmethod
    def Identity():
        return SO3(xyzw=cs.vertcat(0, 0, 0, 1))

    @staticmethod
    def from_quat(xyzw: Vector) -> "SO3":
        assert xyzw.shape == (4, 1) or (4,)
        return SO3(xyzw=xyzw)

    @staticmethod
    def from_euler(rpy: Vector) -> "SO3":
        assert rpy.shape == (3,) or (3, 1)
        return SO3.q_from_rpy(rpy)

    @staticmethod
    def from_matrix(matrix: Matrix) -> "SO3":
        m = matrix
        assert m.shape == (3, 3)
        qw = 0.5 * cs.sqrt(m[0, 0] + m[1, 1] + m[2, 2] + 1)
        qx = 0.5 * cs.sign(m[2, 1] - m[1, 2]) * cs.sqrt(m[0, 0] - m[1, 1] - m[2, 2] + 1)
        qy = 0.5 * cs.sign(m[0, 2] - m[2, 0]) * cs.sqrt(m[1, 1] - m[2, 2] - m[0, 0] + 1)
        qz = 0.5 * cs.sign(m[1, 0] - m[0, 1]) * cs.sqrt(m[2, 2] - m[0, 0] - m[1, 1] + 1)
        return SO3(xyzw=cs.vertcat(qx, qy, qz, qw))

    @staticmethod
    def from_axis_angle(axis: Vector, angle: Scalar) -> "SO3":
        axis = axis / cs.norm_2(axis)
        return SO3(xyzw=cs.vertcat(axis * cs.sin(angle / 2), cs.cos(angle / 2)))

    @staticmethod
    def from_rotation_vector(rotation_vector: Vector) -> "SO3":
        """From a rotation vector to a SO3 object.
        See: https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Rotation_vector
        """
        angle = cs.norm_2(rotation_vector)
        axis = rotation_vector / angle
        return SO3.from_axis_angle(axis, angle)

    def as_quat(self) -> Quaternion:
        return self.quat

    def as_matrix(self) -> Matrix:
        return (
            cs.DM.eye(3)
            + 2 * self.quat.coeffs()[3] * cs.skew(self.quat.coeffs()[:3])
            + 2 * cs.mpower(cs.skew(self.quat.coeffs()[:3]), 2)
        )

    def as_euler(self) -> Vector:
        [qx, qy, qz, qw] = [self.xyzw[0], self.xyzw[1], self.xyzw[2], self.xyzw[3]]
        roll = cs.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = cs.arcsin(2 * (qw * qy - qz * qx))
        yaw = cs.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return cs.vertcat(roll, pitch, yaw)

    @staticmethod
    def qx(q: Angle) -> "SO3":
        return SO3(xyzw=cs.vertcat(cs.sin(q / 2), 0, 0, cs.cos(q / 2)))

    @staticmethod
    def qy(q: Angle) -> "SO3":
        return SO3(xyzw=cs.vertcat(0, cs.sin(q / 2), 0, cs.cos(q / 2)))

    @staticmethod
    def qz(q: Angle) -> "SO3":
        return SO3(xyzw=cs.vertcat(0, 0, cs.sin(q / 2), cs.cos(q / 2)))

    def inverse(self) -> "SO3":
        return SO3(xyzw=self.quat.conjugate().coeffs())

    def transpose(self) -> "SO3":
        return SO3(xyzw=cs.vertcat(-self.quat.coeffs()[:3], self.quat.coeffs()[3]))

    @staticmethod
    def q_from_rpy(rpy) -> "SO3":
        return SO3.qz(rpy[2]) * SO3.qy(rpy[1]) * SO3.qx(rpy[0])

    def act(self, pos: Vector) -> Vector:
        return self.as_matrix() @ pos

    def __mul__(self, other) -> "SO3":
        return SO3(xyzw=(self.quat * other.quat).coeffs())

    def __rmul__(self, other) -> "SO3":
        return SO3(xyzw=(other.quat * self.xyzw).coeffs())

    def log(self) -> "SO3Tangent":
        # Θ = 2 * v * np.arctan2(||v||, w) / ||v||
        norm = cs.norm_2(self.quat.coeffs()[:3] + cs.np.finfo(np.float64).eps)
        theta = (
            2 * self.quat.coeffs()[:3] * cs.atan2(norm, self.quat.coeffs()[3]) / norm
        )
        return SO3Tangent(vec=theta)

    def __sub__(self, other) -> "SO3Tangent":
        if type(self) is type(other):
            return (other.inverse() * self).log()
        else:
            raise RuntimeError("[SO3: __add__] Hey! Someone is not a Lie element.")

    def quaternion_derivative(
        self,
        omega: Vector,
        omega_in_body_fixed: bool = False,
        baumgarte_coefficient: float | None = None,
    ):
        if baumgarte_coefficient is not None:
            baumgarte_term = (
                baumgarte_coefficient
                * cs.norm_2(omega)
                * (1 - cs.norm_2(self.as_quat().coeffs()))
            )
            _omega = Quaternion(
                cs.vertcat(
                    omega,
                    baumgarte_term,
                )
            )
        else:
            _omega = Quaternion(cs.vertcat(omega, 0))
        # using the quaternion product formula
        return (
            0.5 * self.as_quat() * _omega
            if omega_in_body_fixed
            else 0.5 * _omega * self.as_quat()
        ).coeffs()

    @staticmethod
    def slerp(r1: "SO3", r2: "SO3", n: int) -> list["SO3"]:
        """
        Spherical linear interpolation between two rotations.

        Args:
            r1 (SO3): First quaternion
            r2 (SO3): Second quaternion
            n (Scalar): Number of interpolation steps

        Returns:
            list[SO3]: Interpolated rotations
        """
        q1 = r1.as_quat()
        q2 = r2.as_quat()
        interpolated_quats = Quaternion.slerp(q1, q2, n)
        return [SO3(xyzw=q.coeffs()) for q in interpolated_quats]


@dataclasses.dataclass
class SO3Tangent:
    vec: TangentVector

    def __repr__(self) -> str:
        return f"SO3Tangent vector:{str(self.vec)}"

    def exp(self) -> SO3:
        theta = cs.norm_2(self.vec + cs.np.finfo(np.float64).eps)

        def exact(self):
            u = self.vec / theta
            return SO3(xyzw=cs.vertcat(u * cs.sin(theta / 2), cs.cos(theta / 2)))

        def approx(self):
            # sin(x/2)/2 -> 1/2 - x^2/48 + x^4/3840
            return SO3(
                xyzw=cs.vertcat(
                    self.vec
                    * (1 / 2 - cs.power(theta, 2) / 48 + cs.power(theta, 4) / 3840),
                    cs.cos(theta / 2),
                )
            )

        return exact(self)

    def __add__(self, other):
        if type(other) is SO3:
            return self.exp() * other
        else:
            raise RuntimeError("[SO3: __add__] Hey! Someone is not a Lie element.")

    def __radd__(self, other):
        if type(other) is SO3:
            return other * self.exp()
        else:
            raise RuntimeError("[SO3: __add__] Hey! Someone is not a Lie element.")

    def __mul__(self, other):
        if type(other) is float:
            return SO3Tangent(vec=self.vec * other)
        else:
            raise RuntimeError("[SO3: __add__] Hey! Someone is not a Lie element.")
