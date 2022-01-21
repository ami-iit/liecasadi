# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import casadi as cs

from liecasadi import Quaternion
from liecasadi.hints import Vector


class DualQuaternion:
    def __init__(self, qr, qd):
        self.qr = Quaternion(qr)
        self.qd = Quaternion(qd)

    def __repr__(self) -> str:
        return "q_r: {} \nq_d: {}".format(str(self.qr), str(self.qd))

    def __mul__(self, other):
        qr = self.qr * other.qr
        qd = self.qr * other.qd + self.qd * other.qr
        return DualQuaternion(qr=qr.q, qd=qd.q)

    @staticmethod
    def from_quaternion_and_translation(quat, transl):
        t = Quaternion(cs.vertcat(transl, 0))
        r = Quaternion(quat)
        qd = 0.5 * (t * r).q
        return DualQuaternion(qr=r.q, qd=qd)

    def inverse(self):
        qr_inv = self.qr.inverse()
        qd = -qr_inv * self.qd * qr_inv
        return DualQuaternion(qr=qr_inv, qd=qd)

    def translation(self):
        return 2 * (self.qd * self.qr.inverse()).q


if __name__ == "__main__":
    import numpy as np

    quat = np.random.randn(4) * 4
    quat = quat / np.linalg.norm(quat)

    trans = np.random.randn(3) * 4
    print(trans)
    dq = DualQuaternion.from_quaternion_and_translation(quat, trans)

    quat2 = np.random.randn(4) * 4
    quat2 = quat2 / np.linalg.norm(quat2)

    trans2 = np.random.randn(4) * 4

    dq2 = DualQuaternion(quat2, trans2)

    print((dq * dq2).inverse())
    print(dq.translation())
