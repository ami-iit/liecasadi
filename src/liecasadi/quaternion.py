# Copyright (C) Istituto Italiano di Tecnologia (IIT). All rights reserved.

import dataclasses

import casadi as cs

from liecasadi.hints import Scalar, Vector


@dataclasses.dataclass
class Quaternion:
    """Class for quaternions in the form [x, y, z, w]"""
    xyzw: Vector

    def __getattr__(self, attr):
        return getattr(self.xyzw, attr)

    def __repr__(self) -> str:
        return f"Quaternion: {self.xyzw}"

    def __str__(self) -> str:
        return str(self.xyzw)

    def __mul__(self, other: "Quaternion") -> "Quaternion":
        """Multiplication with a quaternion

        Args:
            other (Quaternion): Quaternion to multiply by

        Returns:
            Quaternion: Multiplied quaternion
        """
        return Quaternion(xyzw=Quaternion.product(self.xyzw, other.xyzw))

    def __rmul__(self, other: Scalar) -> "Quaternion":
        """Multiplication by a scalar

        Args:
            other (Scalar): Scalar to multiply by

        Returns:
            Quaternion: Quaternion multiplied by the scalar
        """
        return Quaternion(xyzw=other * self.xyzw)

    def __add__(self, other: "Quaternion") -> "Quaternion":
        """Addition with a quaternion

        Args:
            other (Quaternion): Quaternion to add

        Returns:
            Quaternion: Added quaternion
        """
        return Quaternion(xyzw=self.xyzw + other.xyzw)

    def __radd__(self, other: "Quaternion") -> "Quaternion":
        """Addition with a quaternion

        Args:
            other (Quaternion): Quaternion to add

        Returns:
            Quaternion: Added quaternion
        """
        return Quaternion(xyzw=self.xyzw + other.xyzw)

    def __sub__(self, other: "Quaternion") -> "Quaternion":
        """Subtraction with a quaternion

        Args:
            other (Quaternion): Quaternion to subtract

        Returns:
            Quaternion: Subtracted quaternion
        """
        return Quaternion(xyzw=self.xyzw - other.xyzw)

    def __neg__(self) -> "Quaternion":
        """Negation of the quaternion

        Returns:
            Quaternion: Negated quaternion
        """
        return Quaternion(xyzw=-self.xyzw)

    def __rsub__(self, other: "Quaternion") -> "Quaternion":
        """Subtraction with a quaternion

        Args:
            other (Quaternion): Quaternion to subtract

        Returns:
            Quaternion: Subtracted quaternion
        """
        return Quaternion(xyzw=self.xyzw - other.xyzw)

    def __truediv__(self, other: Scalar) -> "Quaternion":
        """Division by a scalar

        Args:
            other (Scalar): Scalar to divide by

        Returns:
            Quaternion: Quaternion divided by the scalar
        """
        return Quaternion(xyzw=self.xyzw / other)

    def conjugate(self) -> "Quaternion":
        """Conjugate of the quaternion

        Returns:
            Quaternion: Conjugate of the quaternion
        """
        return Quaternion(xyzw=cs.vertcat(-self.xyzw[:3], self.xyzw[3]))

    def normalize(self) -> "Quaternion":
        """Normalize the quaternion

        Returns:
            Quaternion: Normalized quaternion
        """
        xyzw_n = self.xyzw / cs.norm_2(self.xyzw)
        return Quaternion(xyzw=xyzw_n)

    @staticmethod
    def product(q1: Vector, q2: Vector) -> Vector:
        """Product between two quaternions

        Args:
            q1 (Vector): first quaternion as [x, y, z, w]
            q2 (Vector): second quaternion as [x, y, z, w]

        Returns:
            Vector: Product between the two quaternions as [x, y, z, w]
        """
        p1 = q1[3] * q2[3] - cs.dot(q1[:3], q2[:3])
        p2 = q1[3] * q2[:3] + q2[3] * q1[:3] + cs.cross(q1[:3], q2[:3])
        return cs.vertcat(p2, p1)

    def cross(self, other: "Quaternion") -> "Quaternion":
        """Cross product between two quaternions

        Args:
            other (Quaternion): Quaternion to cross with

        Returns:
            Quaternion: Cross product between the two quaternions
        """
        return Quaternion(xyzw=cs.cross(self.xyzw, other.xyzw))

    def coeffs(self, scalar_first: bool = False) -> Vector:
        """Returns the coefficients of the quaternion xyzw

        Args:
            scalar_first (bool, optional): If True, the scalar part is the first element. Defaults to False.

        Returns:
            Vector: Coefficients of the quaternion
        """
        if scalar_first:
            return cs.vertcat(self.w, self.x, self.y, self.z)
        return self.xyzw

    @property
    def x(self) -> Scalar:
        """The x component of the quaternion

        Returns:
            Scalar: x component of the quaternion
        """
        return self.xyzw[0]

    @property
    def y(self) -> Scalar:
        """The y component of the quaternion

        Returns:
            Scalar: y component of the quaternion
        """
        return self.xyzw[1]

    @property
    def z(self) -> Scalar:
        """The z component of the quaternion

        Returns:
            Scalar: z component of the quaternion
        """
        return self.xyzw[2]

    @property
    def w(self) -> Scalar:
        """The w component of the quaternion

        Returns:
            Scalar: w component of the quaternion
        """
        return self.xyzw[3]

    def inverse(self) -> "Quaternion":
        """Inverse of the quaternion

        Returns:
            Quaternion: Inverse of the quaternion
        """
        return self.conjugate() / cs.dot(self.xyzw, self.xyzw)

    @staticmethod
    def slerp(q1: "Quaternion", q2: "Quaternion", n: Scalar) -> list["Quaternion"]:
        """Spherical linear interpolation between two quaternions
        check https://en.wikipedia.org/wiki/Slerp for more details

        Args:
            q1 (Quaternion): First quaternion
            q2 (Quaternion): Second quaternion
            n (Scalar): Number of interpolation steps

        Returns:
            list[Quaternion]: Interpolated quaternion
        """
        q1 = q1.coeffs()
        q2 = q2.coeffs()
        return [Quaternion.slerp_step(q1, q2, t) for t in cs.np.linspace(0, 1, n)]

    @staticmethod
    def slerp_step(q1: Vector, q2: Vector, t: Scalar) -> Vector:
        """Step for the splerp function

        Args:
            q1 (Vector): First quaternion
            q2 (Vector): Second quaternion
            t (Scalar): Interpolation parameter

        Returns:
            Vector: Interpolated quaternion
        """

        dot = cs.dot(q1, q2)
        angle = cs.acos(dot)
        # if the angle is small (meaning the quaternions are "equal") we return the first quaternion
        return Quaternion(
            cs.if_else(
                angle < 1e-6,
                q1,
                (cs.sin((1.0 - t) * angle) * q1 + cs.sin(t * angle) * q2)
                / cs.sin(angle),
            )
        )
