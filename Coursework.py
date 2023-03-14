import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from Vector import Vector


def calculate():
    # simulation variables
    time_step = 0.01
    duration = 20

    number_of_iterations = round(duration / time_step)

    # initialise data arrays
    angular_momenta = [Vector() for _ in range(number_of_iterations)]
    velocities = [Vector() for _ in range(number_of_iterations)]
    positions = [Vector() for _ in range(number_of_iterations)]
    point_p_positions = [Vector() for _ in range(number_of_iterations)]
    time_measurements = [float(0) for _ in range(number_of_iterations)]

    # object variables
    mass = 10
    radius = 1
    height = 4

    point_p_local = Vector(0, 0.75 * radius, 0)

    # inertial tensor principle axis
    inertial_tensor_principal_axis = Vector()

    inertial_tensor_principal_axis.x = mass * 3 * (radius ** 2 + 0.25 * (height ** 2)) / 20.0
    inertial_tensor_principal_axis.y = mass * 3 * (radius ** 2 + 0.25 * (height ** 2)) / 20.0
    inertial_tensor_principal_axis.z = mass * 3 * (2 * (radius ** 2)) / 20.0

    # euler equation for free rotation
    euler_gamma = Vector()

    euler_gamma.x = ((inertial_tensor_principal_axis.z - inertial_tensor_principal_axis.y)
                     / inertial_tensor_principal_axis.x)

    euler_gamma.y = ((inertial_tensor_principal_axis.x - inertial_tensor_principal_axis.z)
                     / inertial_tensor_principal_axis.y)

    euler_gamma.z = ((inertial_tensor_principal_axis.y - inertial_tensor_principal_axis.x)
                     / inertial_tensor_principal_axis.z)

    # simulation initial conditions
    angular_momenta[0] = Vector(3, 1, 2)
    acceleration = Vector(0, 0, -9.8)
    velocities[0] = Vector(0, 0, 200)
    point_p_positions[0].elements = positions[0].elements + point_p_local.elements
    time_measurements[0] = 0

    # run simulation
    for n in range(number_of_iterations - 1):
        next_time = n * time_step

        next_angular_momentum = calculate_next_angular_momentum(angular_momenta, euler_gamma, n, time_step)

        next_position = calculate_next_position(acceleration, n, positions, time_step, velocities)

        local_p_this_step = calculate_next_local_p_position(next_angular_momentum, next_time, point_p_local)

        next_p_position = point_p_positions[n + 1]
        next_p_position.elements = next_position.elements + local_p_this_step.elements

        time_measurements[n + 1] = next_time

    # return results
    return time_measurements, angular_momenta,  positions, velocities, point_p_positions


def calculate_next_angular_momentum(angular_momentum, euler_gamma, n, time_step) -> Vector:
    current_angular_momentum = angular_momentum[n]

    runge_kutta_total = perform_runge_kutta(current_angular_momentum, euler_gamma, time_step)

    next_angular_momentum = angular_momentum[n + 1]
    next_angular_momentum.elements = current_angular_momentum.elements + runge_kutta_total.elements

    return next_angular_momentum


def perform_runge_kutta(current_angular_momentum, euler_gamma, time_step) -> Vector:
    """
    Performs 4th order runge-kutta on an object given a current angular momentum,
    principle moments of inertia, and a timestep
    :return: The result of the runge-kutta calculation
    """
    runge_kutta1 = runge_kutta_one(time_step, euler_gamma, current_angular_momentum)
    runge_kutta2 = runge_kutta_two_and_three(current_angular_momentum, euler_gamma, runge_kutta1, time_step)
    runge_kutta3 = runge_kutta_two_and_three(current_angular_momentum, euler_gamma, runge_kutta2, time_step)
    runge_kutta4 = runge_kutta_four(current_angular_momentum, euler_gamma, runge_kutta3, time_step)

    runge_kutta_total = Vector()
    runge_kutta_total.elements = ((runge_kutta1.elements
                                   + (2 * runge_kutta2.elements)
                                   + (2 * runge_kutta3.elements)
                                   + runge_kutta4.elements)
                                  / 6.0)

    return runge_kutta_total


def runge_kutta_one(time_step, euler_gamma, current_angular_momentum) -> Vector:
    runge_kutta1 = Vector()
    runge_kutta1.x = - time_step * euler_gamma.x * current_angular_momentum.y * current_angular_momentum.z
    runge_kutta1.y = - time_step * euler_gamma.y * current_angular_momentum.x * current_angular_momentum.z
    runge_kutta1.z = - time_step * euler_gamma.z * current_angular_momentum.x * current_angular_momentum.y
    return runge_kutta1


def runge_kutta_two_and_three(current_angular_momentum, euler_gamma, runge_kutta1, time_step) -> Vector:
    runge_kutta2 = Vector()
    runge_kutta2.x = (
            -time_step
            * euler_gamma.x
            * (current_angular_momentum.y + 0.5 * runge_kutta1.y)
            * (current_angular_momentum.z + 0.5 * runge_kutta1.z)
    )

    runge_kutta2.y = (
            -time_step
            * euler_gamma.y
            * (current_angular_momentum.x + 0.5 * runge_kutta1.x)
            * (current_angular_momentum.z + 0.5 * runge_kutta1.z)
    )

    runge_kutta2.z = (
            -time_step
            * euler_gamma.z
            * (current_angular_momentum.x + 0.5 * runge_kutta1.x)
            * (current_angular_momentum.y + 0.5 * runge_kutta1.y)
    )

    return runge_kutta2


def runge_kutta_four(current_angular_momentum, euler_gamma, runge_kutta3, time_step) -> Vector:
    runge_kutta4 = Vector()

    runge_kutta4.x = (
            -time_step
            * euler_gamma.x
            * (current_angular_momentum.y + runge_kutta3.y)
            * (current_angular_momentum.z + runge_kutta3.z)
    )
    runge_kutta4.y = (
            -time_step
            * euler_gamma.y
            * (current_angular_momentum.x + runge_kutta3.x)
            * (current_angular_momentum.z + runge_kutta3.z)
    )
    runge_kutta4.z = (
            -time_step
            * euler_gamma.z
            * (current_angular_momentum.x + runge_kutta3.x)
            * (current_angular_momentum.y + runge_kutta3.y)
    )
    return runge_kutta4


def calculate_next_position(acceleration, n, position, time_step, velocity) -> Vector:
    # perform euler's method to calculate integration of acceleration into velocity
    current_velocity = velocity[n]
    next_velocity = velocity[n + 1]
    next_velocity.elements = euler_method(current_velocity, acceleration, time_step)

    # perform euler's method to calculate integration of velocity into position
    # uses current position and next velocity, therefore is semi-implicit
    current_position = position[n]
    next_position = position[n + 1]

    next_position.elements = euler_method(current_position, next_velocity, time_step)

    return next_position


def euler_method(current_element, derivative_element, time_step) -> Vector:
    """
    Performs euler's method calculation
    :param derivative_element: the element controlling the rate of change of the function
    :param current_element: the current value of the function
    :param time_step: the delta time of the calculation
    :return: the next value of the function; at the time of 'current_element' + time_step
    """
    return current_element.elements + (derivative_element.elements * time_step)


def calculate_next_local_p_position(next_angular_momentum, next_time, point_p_local) -> Vector:
    """
    Calculate the local position of point_p_local using the angular momentum and time
    """
    # calculate theta from the angular momentum magnitude
    angular_momentum_magnitude = np.linalg.norm(next_angular_momentum.elements, ord=1)
    theta = angular_momentum_magnitude * next_time

    # calculate variables needed for rotation matrix
    normalised_angular_momentum = Vector()
    normalised_angular_momentum.elements = next_angular_momentum.elements / angular_momentum_magnitude

    alpha = normalised_angular_momentum.x
    beta = normalised_angular_momentum.y
    gamma = normalised_angular_momentum.z

    # calculate the rotation matrix
    rotation_matrix = calculate_rotation_matrix(alpha, beta, gamma, theta)

    # apply rotation matrix
    local_p_this_step = Vector()
    local_p_this_step.elements = np.matmul(rotation_matrix, point_p_local.elements)

    # return transformed point
    return local_p_this_step


def calculate_rotation_matrix(alpha, beta, gamma, theta) -> np.array:
    """
    Calculates a rotation matrix for a given set of axis and theta
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    one_minus_cos_theta = (1 - cos_theta)
    alpha_times_beta = (alpha * beta)
    alpha_times_gamma = (alpha * gamma)
    beta_times_gamma = (beta * gamma)

    rotation_matrix = np.array([
        [((alpha ** 2) * one_minus_cos_theta) + cos_theta,
         (alpha_times_beta * one_minus_cos_theta) - (gamma * sin_theta),
         (alpha_times_gamma * one_minus_cos_theta) + (beta * sin_theta)],
        [(alpha_times_beta * one_minus_cos_theta) + (gamma * sin_theta),
         ((beta ** 2) * one_minus_cos_theta) + cos_theta,
         (beta_times_gamma * one_minus_cos_theta) - (alpha * sin_theta)],
        [(alpha_times_gamma * one_minus_cos_theta) - (beta * sin_theta),
         (beta_times_gamma * one_minus_cos_theta) - (alpha * sin_theta),
         ((gamma ** 2) * one_minus_cos_theta) + cos_theta]
    ])
    return rotation_matrix


def plot(time_measurements, angular_momenta,  positions, velocities, point_p_positions):
    """
    Plot the results of the simulation using matplotlib
    """
    # plot angular momenta
    plt.plot(time_measurements, list(map(lambda am: am.x, angular_momenta)), label="x axis")
    plt.plot(time_measurements, list(map(lambda am: am.y, angular_momenta)), label="y axis")
    plt.plot(time_measurements, list(map(lambda am: am.z, angular_momenta)), label="z axis")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular Momenta (rad s^-1)")
    plt.legend()
    plt.title("Angular momenta as function of time")
    plt.show()

    # plot positions and velocities of centre of mass
    plt.plot(time_measurements, list(map(lambda am: am.z, positions)), label="position")
    plt.plot(time_measurements, list(map(lambda am: am.z, velocities)), label="velocity")

    plt.xlabel("Time (s)")
    plt.ylabel("Vertical position (m), Vertical velocity (m s^-1)")
    plt.legend()
    plt.title("Vertical velocity and displacement of centre of mass as function of time")
    plt.show()

    # plot projection of point p
    plt.plot(list(map(lambda am: am.x, point_p_positions)), list(map(lambda am: am.y, point_p_positions)),
             label="x-y projection")

    plt.xlabel("x position (m)")
    plt.ylabel("y position (m)")
    plt.legend()
    plt.title("Projection of position in x-y of point P")
    plt.show()


# run the program
plot(*calculate())
