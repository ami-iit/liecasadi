# liecasadi

[![liecasadi](https://github.com/ami-iit/liecasadi/actions/workflows/tests.yml/badge.svg)](https://github.com/ami-iit/liecasadi/actions/workflows/tests.yml)

`liecasadi` implements Lie groups operation written in CasADi, mainly directed to optimization problem formulation.

Inspired by [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf) and the library [Manif](https://github.com/artivis/manif).

## Install

```
pip install "liecasadi @ git+https://github.com/ami-iit/lie-casadi.git"
```

## Implemented Groups

| **Group** | Description        |
| --------- | ------------------ |
| SO3       | 3D Rotations       |
| SE3       | 3D Rigid Transform |

## Example

```python
from liecasadi import SE3, SO3, SE3Tangent, SO3Tangent

# Random quaternion + normalization
quat = (np.random.rand(4) - 0.5) * 5
quat = quat / np.linalg.norm(quat)
# Random vector
vector3d = (np.random.rand(3) - 0.5) * 2 * np.pi

# Create SO3 object
rotation = SO3(quat)

# Create Identity
identity = SO3.Identity()

# Create SO3Tangent object
tangent = SO3Tangent(vector3d)

# Random translation vector
pos = (np.random.rand(3) - 0.5) * 5

# Create SE3 object
transform = SE3(pos=pos, xyzw=quat)

# Random vector
vector6d = (np.random.rand(3) - 0.5) * 5

# Create SE3Tangent object
tangent = SO3Tangent(vector6d)
```
