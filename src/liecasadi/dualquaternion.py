# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import dataclasses
import warnings
from dataclasses import field

import casadi as cs

from liecasadi import SE3, SO3, Quaternion
from liecasadi.hints import Matrix, Scalar, Vector


@dataclasses.dataclass
class DualQuaternion:
    qr: Vector
    qd: Vector
    Qr: Quaternion = field(init=False)
    Qd: Quaternion = field(init=False)

    def __post_init__(self):
        warnings.warn("[DualQuaternion]: This class is under development!")
        self.Qr = Quaternion(self.qr)
        self.Qd = Quaternion(self.qd)

    def __repr__(self) -> str:
        return f"Rotation quaternion: {self.Qr.xyzw} \nTranslation quaternion: {self.Qd.coeffs()}"

    def __mul__(self, other: "DualQuaternion") -> "DualQuaternion":
        """Dual quaternion multiplication

        Args:
            other (DualQuaternion): a Dual Quaternion

        Returns:
            DualQuaternion: the multiplication product
        """
        qr = self.Qr * other.Qr
        qd = self.Qr * other.Qd + self.Qd * other.Qr
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    def __rmul__(self, other: Scalar) -> "DualQuaternion":
        """Multiplication with a scalar

        Returns:
            Dual Quaternion
        """
        return DualQuaternion(qr=other * self.qr, qd=other * self.qd)

    def __add__(self, other: "DualQuaternion") -> "DualQuaternion":
        """Sum of 2 Dual quaternions

        Args:
            other (DualQuaternion): a Dual Quaternion

        Returns:
            DualQuaternion: the sum of Dual Quaternions
        """
        qr = self.Qr + self.Qr
        qd = self.Qd + self.Qd
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    def __sub__(self, other: "DualQuaternion") -> "DualQuaternion":
        """Difference of 2 Dual quaternions

        Args:
            other (DualQuaternion): a Dual Quaternion

        Returns:
            DualQuaternion: the difference of Dual Quaternions
        """
        qr = self.Qr - self.Qr
        qd = self.Qd - self.Qd
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    @staticmethod
    def from_quaternion_and_translation(
        quat: Vector, transl: Vector
    ) -> "DualQuaternion":
        t = Quaternion(cs.vertcat(transl, 0))
        r = Quaternion(quat)
        qd = 0.5 * (t * r).coeffs()
        return DualQuaternion(qr=r.coeffs(), qd=qd)

    @staticmethod
    def from_matrix(m: Matrix) -> "DualQuaternion":
        se3 = SE3.from_matrix(m)
        r = se3.rotation().as_quat()
        t = Quaternion(cs.vertcat(se3.translation(), 0))
        qd = 0.5 * (t * r).coeffs()
        return DualQuaternion(qr=r.coeffs(), qd=qd)

    def translation(self) -> Vector:
        return 2.0 * (self.Qd * self.Qr.conjugate()).coeffs()

    def rotation(self) -> SO3:
        return SO3(xyzw=self.Qr.coeffs())

    def inverse(self) -> "DualQuaternion":
        qr_inv = self.Qr.conjugate()
        qd = -qr_inv * self.Qd * qr_inv
        return DualQuaternion(qr=qr_inv.coeffs(), qd=qd.coeffs())

    def conjugate(self):
        qr = self.Qr.conjugate()
        qd = self.Qd.conjugate()
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    def as_matrix(self):
        r = self.rotation().as_matrix()
        t = self.translation()

    @staticmethod
    def Identity():
        return DualQuaternion(
            qr=SO3.Identity().as_quat().coeffs, qd=Quaternion([0, 0, 0, 0])
        )
