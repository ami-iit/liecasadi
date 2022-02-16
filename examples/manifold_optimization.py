import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits import mplot3d

from liecasadi import SO3, SO3Tangent

opti = cs.Opti()

T = opti.variable(1)
N = 100

quat = [opti.variable(4, 1) for _ in range(N + 1)]
vel = [opti.variable(3, 1) for _ in range(N + 1)]
dt = T / N


for k in range(N):
    vector_SO3 = SO3Tangent(vel[k] * dt)
    rotation_SO3 = SO3(quat[k])
    opti.subject_to(quat[k + 1] == (vector_SO3 + rotation_SO3).as_quat().coeffs())


C = sum(cs.sumsqr(vel[i]) for i in range(N)) + T

# Initial rotation and velocity
opti.subject_to(quat[0] == SO3.Identity().as_quat().coeffs())
opti.subject_to(vel[0] == 0)
opti.subject_to(opti.bounded(0, T, 10))

# Set random initial guess
quat_rnd = np.random.randn(4, 1)
quat_rnd = quat_rnd / np.linalg.norm(quat_rnd)
for k in range(N + 1):
    opti.set_initial(quat[k], quat_rnd)
for k in range(N):
    opti.set_initial(vel[k], np.zeros([3, 1]))


opti.subject_to(vel[N - 1] == 0)
final_delta_increment = SO3Tangent([cs.pi / 3, cs.pi / 6, cs.pi / 2])

opti.subject_to(quat[N] == (final_delta_increment + SO3.Identity()).as_quat().coeffs())

opti.minimize(C)

opti.solver("ipopt")
try:
    sol = opti.solve()
except:
    print("Can't solve the problem!")


fig1 = plt.figure()
fig1.suptitle("Problem sparsity")
plt.spy(opti.debug.value(cs.jacobian(opti.g, opti.x)))


x = [sol.value(quat[i]) for i in range(N + 1)]
v = [sol.value(vel[i]) for i in range(N)]
time = sol.value(T)
plt.figure()
plt.suptitle("Velocity")
plt.plot(np.linspace(0, time, N), v)

figure = plt.figure()
axes = mplot3d.Axes3D(figure)
x_cords = np.array([1, 0, 0])
y_cords = np.array([0, 1, 0])
z_cords = np.array([0, 0, 1])

axes.set_box_aspect((1, 1, 1))

(xax,) = axes.plot([0, 1], [0, 0], [0, 0], "red")
(yax,) = axes.plot([0, 0], [0, 1], [0, 0], "green")
(zax,) = axes.plot([0, 0], [0, 0], [0, 1], "blue")


# final orientation
x_N = np.array(SO3(x[N]).act(x_cords)).reshape(
    3,
)
y_N = np.array(SO3(x[N]).act(y_cords)).reshape(
    3,
)
z_N = np.array(SO3(x[N]).act(z_cords)).reshape(
    3,
)

(xaxN,) = axes.plot([0, x_N[0]], [0, x_N[1]], [0, x_N[2]], "red")
(yaxN,) = axes.plot([0, y_N[0]], [0, y_N[1]], [0, y_N[2]], "green")
(zaxN,) = axes.plot([0, z_N[0]], [0, z_N[1]], [0, z_N[2]], "blue")


def update_points(i):
    x_i = np.array(SO3(x[i]).act(x_cords)).reshape(
        3,
    )
    y_i = np.array(SO3(x[i]).act(y_cords)).reshape(
        3,
    )
    z_i = np.array(SO3(x[i]).act(z_cords)).reshape(
        3,
    )
    # update properties
    xax.set_data(np.array([[0, x_i[0]], [0, x_i[1]]]))
    xax.set_3d_properties(np.array([0, x_i[2]]), "z")

    yax.set_data(np.array([[0, y_i[0]], [0, y_i[1]]]))
    yax.set_3d_properties(np.array([0, y_i[2]]), "z")

    zax.set_data(np.array([[0, z_i[0]], [0, z_i[1]]]))
    zax.set_3d_properties(np.array([0, z_i[2]]), "z")

    # return modified axis
    return (
        xax,
        yax,
        zax,
    )


ani = animation.FuncAnimation(figure, update_points, frames=N, repeat=False)
writergif = animation.PillowWriter(fps=5)
ani.save("animation.gif", writer=writergif)

plt.show()
