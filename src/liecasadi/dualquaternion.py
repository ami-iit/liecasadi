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
    """
    Dual Quaternion class
    Have a look at the documents https://cs.gmu.edu/~jmlien/teaching/cs451/uploads/Main/dual-quaternion.pdf
    https://faculty.sites.iastate.edu/jia/files/inline-files/dual-quaternion.pdf for some intuitions.
    """

    qr: Vector
    qd: Vector
    Qr: Quaternion = field(init=False)
    Qd: Quaternion = field(init=False)

    def __post_init__(self):
        """Build two Quaternion objects from xyzw quaternion and xyz translation"""
        self.Qr = Quaternion(self.qr)
        self.Qd = Quaternion(self.qd)

    def __repr__(self) -> str:
        """
        Returns:
            str: when you print the dual quaternion
        """
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
        qr = self.Qr + other.Qr
        qd = self.Qd + other.Qd
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    def __sub__(self, other: "DualQuaternion") -> "DualQuaternion":
        """Difference of 2 Dual quaternions

        Args:
            other (DualQuaternion): a Dual Quaternion

        Returns:
            DualQuaternion: the difference of Dual Quaternions
        """
        qr = self.Qr - other.Qr
        qd = self.Qd - other.Qd
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    @staticmethod
    def from_quaternion_and_translation(
        quat: Vector, transl: Vector
    ) -> "DualQuaternion":
        """Build dual quaternion from a quaternion (xyzw) and a translation vector (xyz)

        Args:
            quat (Vector): xyzw quaternion vector
            transl (Vector): xyz translation vector

        Returns:
            DualQuaternion: a dual quaternion
        """
        t = Quaternion(cs.vertcat(transl, 0))
        r = Quaternion(quat)
        qd = 0.5 * (t * r).coeffs()
        return DualQuaternion(qr=r.coeffs(), qd=qd)

    @staticmethod
    def from_matrix(m: Matrix) -> "DualQuaternion":
        """Build dual quaternion from an homogenous matrix

        Args:
            m (Matrix): homogenous matrix

        Returns:
            DualQuaternion: a dual quaternion
        """
        se3 = SE3.from_matrix(m)
        r = se3.rotation().as_quat()
        t = Quaternion(cs.vertcat(se3.translation(), 0))
        qd = 0.5 * (t * r).coeffs()
        return DualQuaternion(qr=r.coeffs(), qd=qd)

    def coeffs(self) -> Vector:
        """
        Returns:
            Vector: the dual quaternion vector xyzwxyz
        """
        return cs.vertcat(self.qd, self.qr)

    def translation(self) -> Vector:
        """
        Returns:
            Vector: Translation vector xyz
        """
        return 2.0 * (self.Qd * self.Qr.conjugate()).coeffs()[:3]

    def rotation(self) -> SO3:
        """
        Returns:
            SO3: an SO3 object
        """
        return SO3(xyzw=self.Qr.coeffs())

    def inverse(self) -> "DualQuaternion":
        """
        Returns:
            DualQuaternion: a dual quaternion, inverse of the original
        """
        qr_inv = self.Qr.conjugate()
        qd = -qr_inv * self.Qd * qr_inv
        return DualQuaternion(qr=qr_inv.coeffs(), qd=qd.coeffs())

    def conjugate(self) -> "DualQuaternion":
        """
        Returns:
            DualQuaternion: conjugate dual quaternion
        """
        qr = self.Qr.conjugate()
        qd = self.Qd.conjugate()
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    def dual_conjugate(self) -> "DualQuaternion":
        """
        Returns:
            DualQuaternion: dual number conjugate, used in point transformation
        """
        qr = self.Qr.conjugate()
        qd = self.Qd.conjugate()
        return DualQuaternion(qr=qr.coeffs(), qd=-qd.coeffs())

    def as_matrix(self) -> Matrix:
        """
        Returns:
            Matrix: the corresponding homogenous matrix
        """
        r = self.rotation().as_matrix()
        t = self.translation()
        return cs.vertcat(
            cs.horzcat(r, t),
            cs.horzcat([0, 0, 0, 1]).T,
        )

    @staticmethod
    def Identity() -> "DualQuaternion":
        """
        Returns:
            DualQuaternion: the identity dual quaternion
        """
        return DualQuaternion(
            qr=SO3.Identity().as_quat().coeffs(), qd=Quaternion(cs.DM.zeros(4)).coeffs()
        )

    def transform_point(self, xyz: Vector) -> Vector:
        """Rototranlates a point
        Args:
            xyz (Vector): the point

        Returns:
            Vector: the transformed point
        """
        p = DualQuaternion.Identity()
        xyzw = cs.vertcat(xyz[0], xyz[1], xyz[2], 0)
        p.Qd = Quaternion(xyzw)
        return (self * p * self.dual_conjugate()).coeffs()[:3]
