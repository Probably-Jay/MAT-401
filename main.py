import matplotlib.pyplot as plt
import numpy as np

from Vector import Vector


def calclate():
    time_step = 0.08
    number_of_iterations = 800

    angular_momenta = [Vector() for i in range(number_of_iterations)]

    time_measurements = [float(0)] * number_of_iterations

    mass = 3.14

    dimensions = Vector(3, 2, 1)

    angular_momenta[0] = Vector(1, 1, 1)

    time_measurements[0] = 0

    inertial_tensor_principal_axis: Vector = Vector()

    inertial_tensor_principal_axis.x = mass * (dimensions.y ** 2 + dimensions.z ** 2) / 5
    inertial_tensor_principal_axis.y = mass * (dimensions.x ** 2 + dimensions.z ** 2) / 5
    inertial_tensor_principal_axis.z = mass * (dimensions.x ** 2 + dimensions.y ** 2) / 5

    # todo rename this
    principle_moment = Vector()

    principle_moment.x = (inertial_tensor_principal_axis.z - inertial_tensor_principal_axis.y) / inertial_tensor_principal_axis.x
    principle_moment.y = (inertial_tensor_principal_axis.x - inertial_tensor_principal_axis.z) / inertial_tensor_principal_axis.y
    principle_moment.z = (inertial_tensor_principal_axis.y - inertial_tensor_principal_axis.x) / inertial_tensor_principal_axis.z

    for n in range(number_of_iterations - 1):
        current_angular_momentum = angular_momenta[n]

        runge_kutta2 = perform_runge_kutta(current_angular_momentum, principle_moment, time_step)

        next_angular_momentum = angular_momenta[n + 1]
        next_angular_momentum.elements = current_angular_momentum.elements + runge_kutta2.elements

        time_measurements[n + 1] = n * time_step

    return time_measurements, angular_momenta


def perform_runge_kutta(current_angular_momentum, principle_moment, time_step):
    runge_kutta1 = runge_kutta_one(time_step, principle_moment, current_angular_momentum)
    runge_kutta2 = runge_kutta_two(current_angular_momentum, principle_moment, runge_kutta1, time_step)

    runge_kutta3 = runge_kutta_three(current_angular_momentum, principle_moment, runge_kutta2, time_step)

    return runge_kutta3


def runge_kutta_three(current_angular_momentum, principle_moment, runge_kutta2, time_step):
    runge_kutta3 = Vector()
    runge_kutta3.x = (
            -time_step
            * principle_moment.x
            * (current_angular_momentum.y + 0.5 * runge_kutta2.y)
            * (current_angular_momentum.z + 0.5 * runge_kutta2.z)
    )
    runge_kutta3.y = (
            -time_step
            * principle_moment.y
            * (current_angular_momentum.x + 0.5 * runge_kutta2.x)
            * (current_angular_momentum.z + 0.5 * runge_kutta2.z)
    )
    runge_kutta3.z = (
            -time_step
            * principle_moment.z
            * (current_angular_momentum.x + 0.5 * runge_kutta2.x)
            * (current_angular_momentum.y + 0.5 * runge_kutta2.y)
    )

    return runge_kutta3


def runge_kutta_two(current_angular_momentum, principle_moment, runge_kutta1, time_step) -> Vector:
    runge_kutta2 = Vector()
    runge_kutta2.x = (
            -time_step
            * principle_moment.x
            * (current_angular_momentum.y + 0.5 * runge_kutta1.y)
            * (current_angular_momentum.z + 0.5 * runge_kutta1.z)
    )

    runge_kutta2.y = (
            -time_step
            * principle_moment.y
            * (current_angular_momentum.x + 0.5 * runge_kutta1.x)
            * (current_angular_momentum.z + 0.5 * runge_kutta1.z)
    )

    runge_kutta2.z = (
            -time_step
            * principle_moment.z
            * (current_angular_momentum.x + 0.5 * runge_kutta1.x)
            * (current_angular_momentum.y + 0.5 * runge_kutta1.y)
    )

    return runge_kutta2


def runge_kutta_one(time_step, principle_moment, current_angular_momentum) -> Vector:
    runge_kutta1 = Vector()
    runge_kutta1.x = - time_step * principle_moment.x * current_angular_momentum.y * current_angular_momentum.z
    runge_kutta1.y = - time_step * principle_moment.y * current_angular_momentum.x * current_angular_momentum.z
    runge_kutta1.z = - time_step * principle_moment.z * current_angular_momentum.x * current_angular_momentum.y
    return runge_kutta1


def plot(time_measurements, angular_momenta):
    plt.plot(time_measurements, list(map(lambda am: am.x, angular_momenta)))
    plt.plot(time_measurements, list(map(lambda am: am.y, angular_momenta)))
    plt.plot(time_measurements, list(map(lambda am: am.z, angular_momenta)))
    plt.show()


plot(*calclate())

# To summarise the algorithm:
# • Lines 2 to 11: define known constants, the stepsize, the total number of iterative
# steps, nmax , the size of the required arrays to store the output from the algorithm.
