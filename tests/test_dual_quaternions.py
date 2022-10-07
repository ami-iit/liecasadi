import numpy as np
import pytest
from dual_quaternions import DualQuaternion as TestDualQuat
from icecream import ic

from liecasadi import SE3, DualQuaternion

# quat generation
quat = (np.random.rand(4) - 0.5) * 5
quat = quat / np.linalg.norm(quat)

quat2 = (np.random.rand(4) - 0.5) * 5
quat2 = quat2 / np.linalg.norm(quat2)

pos = (np.random.rand(3) - 0.5) * 5
pos2 = (np.random.rand(3) - 0.5) * 5


H1 = SE3.from_position_quaternion(pos, quat).as_matrix()
H2 = SE3.from_position_quaternion(pos2, quat2).as_matrix()

dual_q1 = DualQuaternion.from_matrix(H1)
dual_q2 = DualQuaternion.from_matrix(H2)


def test_concatenation():
    conc_dual_q = dual_q1 * dual_q2
    conc_H = H1 @ H2
    assert conc_dual_q.as_matrix() - conc_H == pytest.approx(0.0, abs=1e-4)


def test_transform_point():
    point = np.random.rand(4, 1)
    point[3] = 1.0
    assert dual_q1.transform_point(point) - (H1 @ point)[:3] == pytest.approx(
        0.0, abs=1e-4
    )


def test_to_matrix():
    assert dual_q1.as_matrix() - H1 == pytest.approx(0.0, abs=1e-4)


def test_translation():
    assert dual_q1.translation() - pos == pytest.approx(0.0, abs=1e-4)


def test_rotation():
    assert dual_q1.rotation().as_quat().coeffs() - quat == pytest.approx(
        0.0, abs=1e-4
    ) or dual_q1.rotation().as_quat().coeffs() + quat == pytest.approx(0.0, abs=1e-4)


def test_inverse():
    assert dual_q1.inverse().as_matrix() - np.linalg.inv(H1) == pytest.approx(
        0.0, abs=1e-4
    )
