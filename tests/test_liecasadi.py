import manifpy
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from liecasadi import SE3, SO3, SE3Tangent, SO3Tangent

# quat generation
quat = (np.random.rand(4) - 0.5) * 5
quat = quat / np.linalg.norm(quat)


# SO3 objects
mySO3 = SO3(quat)
manifSO3 = manifpy.SO3(quat)


# SO3Tangent objects
angle = (np.random.rand(3) - 0.5) * 2 * np.pi
mySO3Tang = SO3Tangent(angle)
manifSO3Tang = manifpy.SO3Tangent(angle)


def test_SO3():
    assert quat - mySO3.as_quat().coeffs() == pytest.approx(0.0, abs=1e-4)
    assert mySO3.as_matrix() - manifSO3.rotation() == pytest.approx(0.0, abs=1e-4)


def test_euler():
    rpy = np.random.randn(3) * np.pi
    assert SO3.from_euler(rpy).as_matrix() - Rotation.from_euler(
        "xyz", rpy
    ).as_matrix() == pytest.approx(0.0, abs=1e-4)


def test_exp():
    assert (
        mySO3Tang.exp().as_quat().coeffs() - manifSO3Tang.exp().quat()
        == pytest.approx(0.0, abs=1e-4)
    )


def test_log():
    assert mySO3Tang.exp().log().vec - angle == pytest.approx(0.0, abs=1e-4)
    # test vs manif using the exp of the log
    assert (
        mySO3.log().exp().as_matrix() - manifSO3.log().exp().rotation()
        == pytest.approx(0.0, abs=1e-4)
    )

def test_rotation_vector():
    rotation_vector = (np.random.rand(3) - 0.5) * 5
    assert SO3.from_rotation_vector(rotation_vector).as_matrix() - Rotation.from_rotvec(
        rotation_vector
    ).as_matrix() == pytest.approx(0.0, abs=1e-4)


def test_inv():
    assert mySO3.inverse().as_matrix() - manifSO3.inverse().rotation() == pytest.approx(
        0.0, abs=1e-4
    )


def test_right_sum():
    assert (mySO3 + mySO3Tang).as_matrix() - (
        manifSO3 + manifSO3Tang
    ).rotation() == pytest.approx(0.0, abs=1e-4)


def test_left_sum():
    assert (mySO3Tang + mySO3).as_matrix() - (
        manifSO3Tang + manifSO3
    ).rotation() == pytest.approx(0.0, abs=1e-4)


quat2 = (np.random.rand(4) - 0.5) * 5
quat2 = quat2 / np.linalg.norm(quat2)
mySO32 = SO3.from_quat(quat2)
manifSO32 = manifpy.SO3(quat2)


def test_sub():
    quat2 = (np.random.rand(4) - 0.5) * 5
    quat2 = quat2 / np.linalg.norm(quat2)
    mySO32 = SO3.from_quat(quat2)
    manifSO32 = manifpy.SO3(quat2)
    assert (mySO3 - mySO32).exp().as_matrix() - (
        (manifSO3.minus(manifSO32)).exp().rotation()
    ) == pytest.approx(0.0, abs=1e-4)


def test_multiplication():
    assert (mySO3 * mySO32).as_matrix() - manifSO3.compose(
        manifSO32
    ).rotation() == pytest.approx(0.0, abs=1e-4)


pos = (np.random.rand(3) - 0.5) * 5
pos2 = (np.random.rand(3) - 0.5) * 5


def test_act():
    assert (mySO3.act(pos)) - (manifSO3.act(pos)) == pytest.approx(0.0, abs=1e-4)


manifSE3 = manifpy.SE3(pos, quat)
mySE3 = SE3(pos=pos, xyzw=quat)

manifSE3_2 = manifpy.SE3(pos2, quat2)
mySE3_2 = SE3(pos=pos2, xyzw=quat2)


# def test_inverse():
assert (mySE3.inverse().transform() - manifSE3.inverse().transform()) == pytest.approx(
    0.0, abs=1e-4
)


def test_mul():
    assert (mySE3 * mySE3_2).transform() - manifSE3.compose(
        manifSE3_2
    ).transform() == pytest.approx(0.0, abs=1e-4)


tangent_ang = np.random.randn(3)
tangent_lin = np.random.randn(3)
tangent_vec = np.concatenate([tangent_lin, tangent_lin]) * 5

mySE3Tangent = SE3Tangent(tangent_vec)
manifSE3Tangent = manifpy.SE3Tangent(tangent_vec)


def test_exp_SE3():
    assert (
        mySE3Tangent.exp().transform() - manifSE3Tangent.exp().transform()
        == pytest.approx(0.0, abs=1e-4)
    )


def test_log_SE3():
    assert (
        mySE3.log().exp().transform() - manifSE3.log().exp().transform()
        == pytest.approx(0.0, abs=1e-4)
    )


def test_sub_SE3():
    assert (mySE3 - mySE3_2).exp().transform() - (
        manifSE3 - manifSE3_2
    ).exp().transform() == pytest.approx(0.0, abs=1e-4)
