# liecasadi

[![liecasadi](https://github.com/ami-iit/liecasadi/actions/workflows/tests.yml/badge.svg)](https://github.com/ami-iit/liecasadi/actions/workflows/tests.yml)

`liecasadi` implements Lie groups operation written in CasADi, mainly directed to optimization problem formulation.

Inspired by [A micro Lie theory for state estimation in robotics](https://arxiv.org/pdf/1812.01537.pdf) and the library [Manif](https://github.com/artivis/manif).

## 🐍 Install

Create a [virtual environment](https://docs.python.org/3/library/venv.html#venv-def), if you prefer. For example:

```bash
pip install virtualenv
python3 -m venv your_virtual_env
source your_virtual_env/bin/activate
```

Inside the virtual environment, install the library from pip:

```bash
pip install liecasadi
```

If you want the last version:

```bash
pip install "liecasadi @ git+https://github.com/ami-iit/lie-casadi.git"
```

## Implemented Groups

| **Group** | Description        |
| --------- | ------------------ |
| SO3       | 3D Rotations       |
| SE3       | 3D Rigid Transform |

### 🚀 Operations

Being:

- $X, Y \in SO3, \ SE3$

- $w \in \text{SO3Tangent}, \ \text{SE3Tangent}$

- $v \in \mathbb{R}^3$

| Operation           |                                       |     Code      |
| :------------------ | :-----------------------------------: | :-----------: |
| Inverse             |               $X^{-1}$                | `X.inverse()` |
| Composition         |              $X \circ Y$              |     `X*Y`     |
| Exponential         |            $\text{exp}(w)$            | `phi.exp() `  |
| Act on vector       |              $X \circ v$              |  `X.act(v)`   |
| Logarithm           |            $\text{log}(X)$            |   `X.log()`   |
| Manifold right plus | $X \oplus  w = X \circ \text{exp}(w)$ |   `X + phi`   |
| Manifold left plus  | $w \oplus X = \text{exp}(w) \circ X$  |   `phi + X`   |
| Manifold minus      |  $X-Y = \text{log}(Y^{-1} \circ X)$   |     `X-Y`     |

## 🦸‍♂️ Example

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

## 🦸‍♂️ Contributing

**liecasadi** is an open-source project. Contributions are very welcome!

Open an issue with your feature request or if you spot a bug. Then, you can also proceed with a Pull-requests! :rocket:

## ⚠️ Work in progress

- Dual Quaternion class
