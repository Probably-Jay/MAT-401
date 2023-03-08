import matplotlib.pyplot as plt

from Vector import Vector


def calculate():
    # simulation variables
    time_step = 0.01
    duration = 20

    number_of_iterations = round(duration / time_step)

    # initialise data arrays
    velocity = [Vector() for _ in range(number_of_iterations)]
    position = [Vector() for _ in range(number_of_iterations)]
    time_measurements = [float(0) for _ in range(number_of_iterations)]

    # object variables
    mass = 10
    radius = 1
    height = 4

    # simulation initial conditions
    acceleration = Vector(0, 0, -9.8)
    velocity[0] = Vector(0, 0, 200)
    time_measurements[0] = 0

    # run simulation
    for n in range(number_of_iterations - 1):
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
    return time_measurements, position, velocity


def euler_method(current_element, derivative_element, time_step):
    """
    Performs euler's method calculation
    :param derivative_element: the element controlling the rate of change of the function
    :param current_element: the current value of the function
    :param time_step: the delta time of the calculation
    :return: the next value of the function; at the time of 'current_element' + time_step
    """
    return current_element.elements + (derivative_element.elements * time_step)


def plot(time_measurements, position, velocity):
    """
    Plot the results of the simulation using matplotlib
    """
    plt.plot(time_measurements, list(map(lambda am: am.z, position)), label="position")
    plt.plot(time_measurements, list(map(lambda am: am.z, velocity)), label="velocity")

    plt.xlabel("Time (s)")
    plt.ylabel("Vertical position (m), Vertical velocity (m s^-1)")
    plt.legend()
    plt.show()


# run the program
plot(*calculate())
