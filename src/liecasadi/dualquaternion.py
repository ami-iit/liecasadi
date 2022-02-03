# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import dataclasses
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
        self.Qr = Quaternion(self.qr)
        self.Qd = Quaternion(self.qd)

    def __repr__(self) -> str:
        return f"Rotation quaternion: {self.Qr.xyzw} \nTranslation quaternion: {self.Qd.coeffs()}"

    def __mul__(self, other: "DualQuaternion") -> "DualQuaternion":
        qr = self.Qr * other.Qr
        qd = self.Qr * other.Qd + self.Qd * other.Qr
        return DualQuaternion(qr=qr.coeffs(), qd=qd.coeffs())

    def __rmul__(self, other: Scalar) -> "DualQuaternion":
        """Multiplication with a scalar

        Returns:
            Dual Quaternion
        """
        return DualQuaternion(qr=other * self.qr, qd=other * self.qd)

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
        t = se3.rotation().as_quat()
        r = Quaternion(cs.vertcat(se3.translation(), 0))
        qd = 0.5 * (t * r).coeffs()
        return DualQuaternion(qr=r.coeffs(), qd=qd)

    def translation(self) -> Vector:
        return 2.0 * (self.Qd * self.Qr.conjugate()).coeff()

    def rotation(self) -> SO3:
        return SO3(xyzw=self.Qr.coeffs())

    def inverse(self) -> "DualQuaternion":
        qr_inv = self.Qr.conjugate()
        qd = -qr_inv * self.Qd * qr_inv
        return DualQuaternion(qr=qr_inv, qd=qd)

    def translation(self):
        return 2 * (self.Qd * self.Qr.conjugate()).coeffs()


if __name__ == "__main__":
    import numpy as np

    quat = np.random.randn(4) * 4
    quat = quat / np.linalg.norm(quat)

    trans = np.random.randn(3) * 4
    dq = DualQuaternion.from_quaternion_and_translation(quat, trans)

    quat2 = np.random.randn(4) * 4
    quat2 = quat2 / np.linalg.norm(quat2)

    trans2 = np.random.randn(4) * 4

    dq2 = DualQuaternion(quat2, trans2)

    d3 = DualQuaternion.from_matrix(np.eye(4))

    print((3 * dq2).inverse())
    print(dq.translation())
