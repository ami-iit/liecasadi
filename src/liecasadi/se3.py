# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import dataclasses

import casadi as cs
import numpy as np

from liecasadi import SO3, SO3Tangent
from liecasadi.hints import Matrix, Vector


@dataclasses.dataclass
class SE3:
    """Class representing the Special Euclidean group SE(3)"""

    pos: Vector
    xyzw: Vector

    def __repr__(self) -> str:
        return f"Position: \t {self.pos} \nQuaternion: \t {self.xyzw}"

    @staticmethod
    def from_matrix(H: Matrix) -> "SE3":
        """Create an SE3 object from an homogeneous transformation matrix

        Args:
            H (Matrix): 4x4 homogeneous transformation matrix

        Returns:
            SE3: SE3 object
        """
        assert H.shape == (4, 4)
        return SE3(pos=H[:3, 3], xyzw=SO3.from_matrix(H[:3, :3]).as_quat().coeffs())

    def rotation(self) -> SO3:
        """Extract the rotation part of the SE3 object as an SO3 object

        Returns:
            SO3: SO3 object
        """
        return SO3(self.xyzw)

    def translation(self) -> Vector:
        """Extract the translation part of the SE3 object

        Returns:
            Vector: 3x1 translation vector
        """
        return self.pos

    def transform(self) -> Matrix:
        """Return the homogeneous transformation matrix

        Returns:
            Matrix: 4x4 homogeneous transformation matrix
        """
        return self.as_matrix()

    def as_matrix(self) -> Matrix:
        """Return the homogeneous transformation matrix

        Returns:
            Matrix: 4x4 homogeneous transformation matrix
        """
        pos = cs.reshape(self.pos, -1, 1)
        return cs.vertcat(
            cs.horzcat(SO3(self.xyzw).as_matrix(), pos),
            cs.horzcat([0, 0, 0, 1]).T,
        )

    def as_adjoint_transform(self) -> Matrix:
        """Return the adjoint transformation matrix
        The adjoint transformation matrix is used to transform the twist (a combination of linear and angular velocity) from one coordinate frame to another
        .. math::
            v_A = Ad_{T_{AB}} v_B

        Returns:
            Matrix: 6x6 adjoint transformation matrix
        """
        R = SO3(self.xyzw).as_matrix()
        p = self.pos
        return cs.vertcat(
            cs.horzcat(R, cs.skew(p) @ R), cs.horzcat(np.zeros((3, 3)), R)
        )

    def adjoint(self) -> Matrix:
        """Return the adjoint transformation matrix
        The adjoint transformation matrix is used to transform the twist (a combination of linear and angular velocity) from one coordinate frame to another
        .. math::
            v_A = Ad_{T_{AB}} v_B

        Returns:
            Matrix: 6x6 adjoint transformation matrix
        """
        return self.as_adjoint_transform()

    def adjoint_inverse(self) -> Matrix:
        """Return the inverse adjoint transformation matrix
        The inverse adjoint transformation matrix is used to transform the twist (a combination of linear and angular velocity) from one coordinate frame to another
        .. math::
            v_B = Ad_{T_{AB}}^{-1} v_A = Ad_{T_{BA}} v_A

        Returns:
            Matrix: 6x6 inverse adjoint transformation matrix
        """
        R = SO3(self.xyzw).as_matrix()
        p = self.pos
        return cs.vertcat(
            cs.horzcat(R.T, -R.T @ cs.skew(p)), cs.horzcat(np.zeros((3, 3)), R.T)
        )

    def as_coadjoint_transform(self) -> Matrix:
        """Return the coadjoint transformation matrix
        The coadjoint transformation matrix is used to transform the wrench (a combination of force and torque) from one coordinate frame to another
        .. math::
            f_A = Ad_{T_{AB}}^T f_B

        Returns:
            Matrix: 6x6 coadjoint transformation matrix
        """
        R = SO3(self.xyzw).as_matrix()
        p = self.pos
        return cs.vertcat(
            cs.horzcat(R, np.zeros((3, 3))), cs.horzcat(cs.skew(p) @ R, R)
        )

    def coadjoint(self) -> Matrix:
        """Return the coadjoint transformation matrix
        The coadjoint transformation matrix is used to transform the wrench (a combination of force and torque) from one coordinate frame to another
        .. math::
            f_A = Ad_{T_{AB}}^T f_B

        Returns:
            Matrix: 6x6 coadjoint transformation matrix
        """
        return self.as_coadjoint_transform()

    def coadjoint_inverse(self) -> Matrix:
        """Return the inverse coadjoint transformation matrix
        The inverse coadjoint transformation matrix is used to transform the wrench (a combination of force and torque) from one coordinate frame to another
        .. math::
            f_B = Ad_{T_{AB}}^{-T} f_A = Ad_{T_{BA}}^T f_A

        Returns:
            Matrix: 6x6 inverse coadjoint transformation matrix
        """
        R = SO3(self.xyzw).as_matrix()
        p = self.pos
        return cs.vertcat(
            cs.horzcat(R.T, np.zeros((3, 3))), cs.horzcat(-R.T @ cs.skew(p), R.T)
        )

    @staticmethod
    def from_position_quaternion(xyz: Vector, xyzw: Vector) -> "SE3":
        """Create an SE3 object from a position and a quaternion

        Args:
            xyz (Vector): Position vector
            xyzw (Vector): Quaternion vector

        Returns:
            SE3: Roto-translation SE3 object
        """
        assert xyz.shape in [(3,), (3, 1)]
        assert xyzw.shape in [(4,), (4, 1)]
        return SE3(pos=xyz, xyzw=xyzw)

    @staticmethod
    def from_translation_and_rotation(translation: Vector, rotation: SO3) -> "SE3":
        """Create an SE3 object from a translation and a SO3 rotation object

        Args:
            translation (Vector): Translation vector (R^3)
            rotation (SO3): SO3 rotation object

        Returns:
            SE3: Roto-translation SE3 object
        """
        return SE3(pos=translation, xyzw=rotation.as_quat().coeffs())

    def inverse(self) -> "SE3":
        """Return the inverse of the SE3 object

        Returns:
            SE3: Inverse of the SE3 object
        """
        return SE3(
            pos=-SO3(self.xyzw).inverse().act(self.pos),
            xyzw=SO3(self.xyzw).transpose().as_quat().coeffs(),
        )

    def log(self) -> "SE3Tangent":
        """Return the tangent vector of the SE3 object

        Returns:
            SE3Tangent: Tangent vector of the SE3 object
        """
        vec = SO3(self.xyzw).log().vec
        # theta = cs.norm_2(vec) # norm_2 is not differentiable at 0
        theta_eps = cs.norm_2(vec + cs.np.finfo(np.float64).eps)
        u = vec / theta_eps
        V = (
            cs.DM.eye(3)
            + (1 - cs.cos(theta_eps)) / theta_eps @ cs.skew(u)
            + (theta_eps - cs.sin(theta_eps)) / theta_eps @ cs.skew(u) @ cs.skew(u)
        )
        tangent_vec = cs.vertcat(cs.inv(V) @ self.pos, vec)
        return SE3Tangent(vec=tangent_vec)

    def __mul__(self, other: "SE3") -> "SE3":
        """Multiplication of two SE3 objects

        .. math::
            "{}^A H_{C} = {}^A H_{B} \cdot {}^B H_{C}"

        Args:
            other (SE3): SE3 object

        Raises:
            RuntimeError: If the input is not an SE3 object

        Returns:
            SE3: Product of the two SE3 objects
        """
        if type(self) is not type(other):
            raise RuntimeError("[SE3: __mul__] Please provide an SE3 object!")
        rotation = SO3(self.xyzw) * SO3(other.xyzw)
        position = self.pos + SO3(self.xyzw).act(other.pos)
        return SE3(pos=position, xyzw=rotation.as_quat().coeffs())

    def __rmul__(self, other: "SE3") -> "SE3":
        """Multiplication of two SE3 objects

        Args:
            other (SE3): SE3 object

        Raises:
            RuntimeError: If the input is not an SE3 object

        Returns:
            SE3: Product of the two SE3 objects
        """
        if type(self) is not type(other):
            raise RuntimeError("[SE3: __rmul__] Please provide an SE3 object!")
        rotation = SO3(other.xyzw) * SO3(self.xyzw)
        position = other.pos + SO3(other.xyzw).act(self.pos)
        return SE3(pos=position, xyzw=rotation.as_quat().coeffs())

    def __sub__(self, other: "SE3") -> "SE3Tangent":
        """Subtraction of two SE3 objects

        Args:
            other (SE3): SE3 object

        Raises:
            RuntimeError: If the input is not an SE3 object

        Returns:
            SE3Tangent: Tangent vector of the difference of the two SE3 objects
        """
        if type(self) is type(other):
            return (other.inverse() * self).log()
        else:
            raise RuntimeError("[SE3: __sub__] Please provide an SE3 object!")

    def __rsub__(self, other: "SE3") -> "SE3Tangent":
        """Subtraction of two SE3 objects

        Args:
            other (SE3): SE3 object

        Raises:
            RuntimeError: If the input is not an SE3 object

        Returns:
            SE3Tangent: Tangent vector of the difference of the two SE3 objects
        """
        if type(self) is type(other):
            return (self.inverse() * other).log()
        else:
            raise RuntimeError("[SE3: __rsub__] Please provide an SE3 object!")


@dataclasses.dataclass
class SE3Tangent:
    vec: Vector

    def __repr__(self) -> str:
        return f"Tangent vector: {self.vec}"

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
        return SE3(pos=trans, xyzw=rot.as_quat().coeffs())

    def vector(self):
        return self.vec
