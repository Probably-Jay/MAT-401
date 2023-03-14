import matplotlib.pyplot as plt

from Vector import Vector


def calculate():
    # simulation variables
    time_step = 0.01
    duration = 20

    number_of_iterations = round(duration / time_step)

    # initialise data arrays
    angular_momentum = [Vector() for _ in range(number_of_iterations)]
    velocity = [Vector() for _ in range(number_of_iterations)]
    position = [Vector() for _ in range(number_of_iterations)]
    time_measurements = [float(0) for _ in range(number_of_iterations)]

    # object variables
    mass = 10
    radius = 1
    height = 4

    point_p_local = Vector(0, 0.75*radius, 0)

    # inertial tensor
    inertial_tensor_principal_axis = Vector()

    inertial_tensor_principal_axis.x = mass * 3 * (radius ** 2 + 0.25 * (height ** 2)) / 20.0
    inertial_tensor_principal_axis.y = mass * 3 * (radius ** 2 + 0.25 * (height ** 2)) / 20.0
    inertial_tensor_principal_axis.z = mass * 3 * (2 * (radius ** 2)) / 20.0

    # principle axis moments
    principle_moments = Vector()

    principle_moments.x = ((inertial_tensor_principal_axis.z - inertial_tensor_principal_axis.y)
                           / inertial_tensor_principal_axis.x)

    principle_moments.y = ((inertial_tensor_principal_axis.x - inertial_tensor_principal_axis.z)
                           / inertial_tensor_principal_axis.y)

    principle_moments.z = ((inertial_tensor_principal_axis.y - inertial_tensor_principal_axis.x)
                           / inertial_tensor_principal_axis.z)

    # simulation initial conditions
    angular_momentum[0] = Vector(3, 1, 2)
    acceleration = Vector(0, 0, -9.8)
    velocity[0] = Vector(0, 0, 200)
    time_measurements[0] = 0

    # run simulation
    for n in range(number_of_iterations - 1):
        current_angular_momentum = angular_momentum[n]

        runge_kutta_total = perform_runge_kutta(current_angular_momentum, principle_moments, time_step)

        next_angular_momentum = angular_momentum[n + 1]
        next_angular_momentum.elements = current_angular_momentum.elements + runge_kutta_total.elements

        # perform euler's method to calculate integration of acceleration into velocity
        current_velocity = velocity[n]
        next_velocity = velocity[n + 1]
        next_velocity.elements = euler_method(current_velocity, acceleration, time_step)

        # perform euler's method to calculate integration of velocity into position
        # uses current position and next velocity, therefore is semi-implicit
        current_position = position[n]
        next_position = position[n + 1]
        next_position.elements = euler_method(current_position, next_velocity, time_step)

        time_measurements[n + 1] = n * time_step

    # return results
    return time_measurements, angular_momentum


def euler_method(current_element, derivative_element, time_step):
    """
    Performs euler's method calculation
    :param derivative_element: the element controlling the rate of change of the function
    :param current_element: the current value of the function
    :param time_step: the delta time of the calculation
    :return: the next value of the function; at the time of 'current_element' + time_step
    """
    return current_element.elements + (derivative_element.elements * time_step)

def perform_runge_kutta(current_angular_momentum, principle_moments, time_step):
    """
    Performs 4th order runge-kutta on an object given a current angular momentum,
    principle moments of inertia, and a timestep
    :return: The result of the runge-kutta calculation
    """
    runge_kutta1 = runge_kutta_one(time_step, principle_moments, current_angular_momentum)
    runge_kutta2 = runge_kutta_two_and_three(current_angular_momentum, principle_moments, runge_kutta1, time_step)
    runge_kutta3 = runge_kutta_two_and_three(current_angular_momentum, principle_moments, runge_kutta2, time_step)
    runge_kutta4 = runge_kutta_four(current_angular_momentum, principle_moments, runge_kutta3, time_step)

    runge_kutta_total = Vector()
    runge_kutta_total.elements = ((runge_kutta1.elements
                                   + (2 * runge_kutta2.elements)
                                   + (2 * runge_kutta3.elements)
                                   + runge_kutta4.elements)
                                  / 6.0)

    return runge_kutta_total


def runge_kutta_one(time_step, principle_moments, current_angular_momentum) -> Vector:
    runge_kutta1 = Vector()
    runge_kutta1.x = - time_step * principle_moments.x * current_angular_momentum.y * current_angular_momentum.z
    runge_kutta1.y = - time_step * principle_moments.y * current_angular_momentum.x * current_angular_momentum.z
    runge_kutta1.z = - time_step * principle_moments.z * current_angular_momentum.x * current_angular_momentum.y
    return runge_kutta1


def runge_kutta_two_and_three(current_angular_momentum, principle_moments, runge_kutta1, time_step) -> Vector:
    runge_kutta2 = Vector()
    runge_kutta2.x = (
            -time_step
            * principle_moments.x
            * (current_angular_momentum.y + 0.5 * runge_kutta1.y)
            * (current_angular_momentum.z + 0.5 * runge_kutta1.z)
    )

    runge_kutta2.y = (
            -time_step
            * principle_moments.y
            * (current_angular_momentum.x + 0.5 * runge_kutta1.x)
            * (current_angular_momentum.z + 0.5 * runge_kutta1.z)
    )

    runge_kutta2.z = (
            -time_step
            * principle_moments.z
            * (current_angular_momentum.x + 0.5 * runge_kutta1.x)
            * (current_angular_momentum.y + 0.5 * runge_kutta1.y)
    )

    return runge_kutta2


def runge_kutta_four(current_angular_momentum, principle_moments, runge_kutta3, time_step):
    runge_kutta4 = Vector()

    runge_kutta4.x = (
            -time_step
            * principle_moments.x
            * (current_angular_momentum.y + runge_kutta3.y)
            * (current_angular_momentum.z + runge_kutta3.z)
    )
    runge_kutta4.y = (
            -time_step
            * principle_moments.y
            * (current_angular_momentum.x + runge_kutta3.x)
            * (current_angular_momentum.z + runge_kutta3.z)
    )
    runge_kutta4.z = (
            -time_step
            * principle_moments.z
            * (current_angular_momentum.x + runge_kutta3.x)
            * (current_angular_momentum.y + runge_kutta3.y)
    )
    return runge_kutta4


def plot(time_measurements, angular_momenta):
    """
    Plot the results of the simulation using matplotlib
    """
    plt.plot(time_measurements, list(map(lambda am: am.x, angular_momenta)), label="x axis")
    plt.plot(time_measurements, list(map(lambda am: am.y, angular_momenta)), label="y axis")
    plt.plot(time_measurements, list(map(lambda am: am.z, angular_momenta)), label="z axis")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Momenta (rad s^-1)")
    plt.legend()
    plt.show()


# run the program
plot(*calculate())
