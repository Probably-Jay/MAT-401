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
        current_velocity = velocity[n]
        next_velocity = velocity[n + 1]
        next_velocity.elements = current_velocity.elements + (acceleration.elements * time_step)

        current_position = position[n]
        next_position = position[n + 1]
        next_position.elements = current_position.elements + (next_velocity.elements * time_step)

        time_measurements[n + 1] = n * time_step

    # return results
    return time_measurements, position, velocity


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
