# Please note that for running this example you need to install `matplotlib` and `scipy`.
# You can do this by running the following command in your terminal:
# pip install matplotlib scipy
# If you are using anaconda, you can also run the following command:
# conda install matplotlib scipy

import casadi as cs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from liecasadi import SO3, SO3Tangent

N = 10

r1 = SO3.Identity()

final_delta_increment = SO3Tangent([cs.pi / 3, cs.pi / 6, cs.pi / 2])

r2 = final_delta_increment + SO3.Identity()

x = SO3.slerp(r1, r2, N)

# If you want to work directly with quaternion, you can use the following code:
# x = Quaternion.slerp(q1, q1, N)
# where q1 and q2 are Quaternion objects.

figure = plt.figure()
axes = figure.add_subplot(projection="3d")
x_cords = np.array([1, 0, 0])
y_cords = np.array([0, 1, 0])
z_cords = np.array([0, 0, 1])

axes.set_box_aspect((1, 1, 1))

(xax,) = axes.plot([0, 1], [0, 0], [0, 0], "red")
(yax,) = axes.plot([0, 0], [0, 1], [0, 0], "green")
(zax,) = axes.plot([0, 0], [0, 0], [0, 1], "blue")

print("qui", x[N - 1].act(x_cords))

# final orientation
x_N = np.array(x[N - 1].act(x_cords)).reshape(
    3,
)
y_N = np.array(x[N - 1].act(y_cords)).reshape(
    3,
)
z_N = np.array(x[N - 1].act(z_cords)).reshape(
    3,
)

(xaxN,) = axes.plot([0, x_N[0]], [0, x_N[1]], [0, x_N[2]], "red")
(yaxN,) = axes.plot([0, y_N[0]], [0, y_N[1]], [0, y_N[2]], "green")
(zaxN,) = axes.plot([0, z_N[0]], [0, z_N[1]], [0, z_N[2]], "blue")


def update_points(i):
    x_i = np.array(x[i].act(x_cords)).reshape(
        3,
    )
    y_i = np.array(x[i].act(y_cords)).reshape(
        3,
    )
    z_i = np.array(x[i].act(z_cords)).reshape(
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
