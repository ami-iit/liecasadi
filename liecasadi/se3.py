# Copyright (C) 2021 Istituto Italiano di Tecnologia (IIT). All rights reserved.
# This software may be modified and distributed under the terms of the
# GNU Lesser General Public License v2.1 or any later version.

import dataclasses

import casadi as cs
import numpy as np

from liecasadi import SO3, SO3Tangent
from liecasadi.hints import Matrix, Vector


@dataclasses.dataclass
class SE3:
    pos: Vector
    xyzw: Vector

    def __repr__(self) -> str:
        return "Position: \t {} \nQuaternion: \t {}".format(
            str(self.pos), str(self.xyzw)
        )

    def from_matrix(H: Matrix) -> "SE3":
        assert H.shape == (4, 4)
        return SE3(pos=H[:3, 3], xyzw=SO3.from_matrix(H[:3, :3]).as_quat())

    @property
    def rotation(self) -> SO3:
        return SO3(self.xyzw)

    def translation(self) -> Vector:
        return self.pos

    def transform(self) -> Matrix:
        a = SO3(self.xyzw)
        pos = cs.reshape(self.pos, -1, 1)
        return cs.vertcat(
            cs.horzcat(SO3(self.xyzw).as_matrix(), pos),
            cs.horzcat([0, 0, 0, 1]).T,
        )

    def from_position_quaternion(xyz: Vector, xyzw: Vector) -> "SE3":
        assert xyz.shape in [(3,), (3, 1)]
        assert xyzw.shape in [(4,), (4, 1)]
        return SE3(pos=xyz, xyzw=xyzw)

    def from_translation_and_rotation(translation: Vector, rotation: SO3) -> "SE3":
        return SE3(pos=translation, xyzw=rotation.as_quat())

    def inverse(self) -> "SE3":
        return SE3(
            pos=-SO3(self.xyzw).inverse().act(self.pos),
            xyzw=SO3(self.xyzw).transpose().as_quat(),
        )

    def log(self) -> "SE3Tangent":
        vec = SO3(self.xyzw).log().vec
        theta = cs.norm_2(vec)
        theta_eps = cs.norm_2(vec + cs.np.finfo(np.float64).eps)
        u = vec / theta_eps
        V = (
            cs.DM.eye(3)
            + (1 - cs.cos(theta_eps)) / theta_eps @ cs.skew(u)
            + (theta_eps - cs.sin(theta_eps)) / theta_eps @ cs.skew(u) @ cs.skew(u)
        )
        tangent_vec = cs.vertcat(cs.inv(V) @ self.pos, vec)
        return SE3Tangent(vec=tangent_vec)

    def __mul__(self, other):
        rotation = SO3(self.xyzw) * SO3(other.xyzw)
        position = self.pos + SO3(self.xyzw).act(other.pos)
        return SE3(pos=position, xyzw=rotation.as_quat())

    def __rmul__(self, other):
        rotation = SO3(other.xyzw) @ SO3(self.xyzw)
        position = other.pos + SO3(other.xyzw).act(self.pos)
        return SE3(pos=position, xyzw=rotation.as_quat())

    def __sub__(self, other) -> "SE3Tangent":
        if type(self) is type(other):
            return (other.inverse() * self).log()
        else:
            raise RuntimeError("[SO3: __add__] Hey! Someone is not a Lie element.")

    def __rsub__(self, other) -> "SE3Tangent":
        if type(self) is type(other):
            return (self.inverse() * other).log()
        else:
            raise RuntimeError("[SO3: __add__] Hey! Someone is not a Lie element.")


@dataclasses.dataclass
class SE3Tangent(SO3Tangent):
    vec = Vector

    def __repr__(self) -> str:
        return "Tangent vector: {}".format(self.vec)

    def exp(self):
        assert self.vec.shape in [(6,), (6, 1)]
        vec = cs.reshape(self.vec, -1, 1)
        rot = SO3Tangent(vec[3:]).exp()
        # theta = cs.norm_2(vec[3:])
        theta_eps = cs.norm_2(vec[3:] + cs.np.finfo(np.float64).eps)
        u = vec[3:] / theta_eps
        V = (
            cs.DM.eye(3)
            + (1 - cs.cos(theta_eps)) / theta_eps @ cs.skew(u)
            + (theta_eps - cs.sin(theta_eps)) / theta_eps @ cs.skew(u) @ cs.skew(u)
        )
        trans = V @ vec[:3]
        # trans = SO3.from_matrix(V).act(vec[:3])
        return SE3(pos=trans, xyzw=rot.as_quat())

    def vector(self):
        return self.vec
