import numpy as np
from typing import Sequence
from datetime import datetime, timedelta
from partner_juggling.tracker.data_types import State
from typing import Tuple, List, Union

def calc_throw_velocity(
    start: Sequence[float],
    target: Sequence[float],
    flight_time: float,
    gravity: np.array = np.array([0,0,-9.81]),
    ) -> np.ndarray:
    """
    Calculate the initial velocity vector needed to reach the target from start in given flight_time.
    
    Args:
        start (array-like): Starting position [x0, y0, z0].
        target (array-like): Target position [x1, y1, z1].
        flight_time (float): Time of flight in seconds.
        gravity (array-like): 3D Acceleration due to gravity (default [0,0, -9.81 m/s^2]).
    
    Returns:
        np.ndarray: Velocity vector [dx, dy, dz].
    """
    start = np.array(start, dtype=float)
    target = np.array(target, dtype=float)
    dx =  (target - start) / flight_time - 0.5 * gravity * flight_time
    
    return dx

def calc_touchdown(
    start: Sequence[float],
    velocity: Sequence[float],
    z_td: float,
    gravity: np.array = np.array([0,0,-9.81]),
    ) -> Tuple[float, np.ndarray]:
    """
    Calculate the time and position when the projectile reaches the given touchdown height z_td.

    Args:
        start (array-like): Starting position [x0, y0, z0].
        velocity (array-like): Initial velocity [vx, vy, vz].
        z_td (float): Touchdown height (z coordinate).
        gravity (array-like): 3D Acceleration due to gravity (default [0,0, -9.81 m/s^2]).

    Returns:
        tuple: (touchdown_time, touchdown_position [x, y, z_td])
    """
    start = np.array(start, dtype=float)
    velocity = np.array(velocity, dtype=float)
    x0, y0, z0 = start
    vx, vy, vz = velocity

    # Solve for t in z(t) = z_td: z0 + vz*t + 0.5*gz*t^2 = z_td
    gz = gravity[2]
    a = 0.5 * gz
    b = vz
    c = z0 - z_td

    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        raise ValueError("No real solution for touchdown time at given z_td.")

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b + sqrt_disc) / (2*a)
    t2 = (-b - sqrt_disc) / (2*a)

    # We want the positive, nonzero time
    touchdown_time = max(t1, t2)
    if touchdown_time < 0:
        raise ValueError("Touchdown time is negative; check inputs.")

    x_td = x0 + vx * touchdown_time
    y_td = y0 + vy * touchdown_time

    return touchdown_time, np.array([x_td, y_td, z_td])


def calc_targetTime_timestamp_kalman(
    track_state: State,
    target_height: float
    ) -> Tuple[datetime, float]:
    """
    Calculate the timestamp at which the tracked object (e.g., ball) will reach a specified target height,
    based on its current Kalman filter state.
    This function solves the kinematic equation for vertical motion under gravity to estimate the time
    required for the object to reach the target height, then computes the corresponding timestamp.
    Args:
        track_state: An object containing the Kalman filter state vector, where
            track_state.state_vector[6] is the current vertical position (z_pos),
            track_state.state_vector[7] is the current vertical velocity (z_velocity).
        target_height (float): The desired vertical position (height) to reach.
    Returns:
        tuple:
            target_time_timestamp (datetime): The estimated timestamp when the object reaches the target height.
            time_delta (float): The time interval (in seconds) from the current timestamp to reaching the target height.
    Raises:
        ValueError: If the calculation results in a complex number (i.e., the target height is not reachable).
    """
    # solve the equation a*t^2 + b*t + (c - target_height) = 0

    # timestamp = datetime.fromtimestamp(data_frame.header.stamp.to_sec()) 
    timestamp = track_state.timestamp

    z_pos = track_state.state_vector[6]
    z_velocity = track_state.state_vector[7]
    z_acc = -9.81  # acceleration due to gravity
    # z_acc = track_state.state_vector[8] 
    
    discriminant = z_velocity**2 - 2 * z_acc * (z_pos - target_height)
    if discriminant < 0:
        raise ValueError("Target height not reachable with current motion parameters. Discriminant is negative.")
        
    time_delta = (-z_velocity - np.sqrt(discriminant)) / z_acc
    target_time_timestamp = timestamp + timedelta(seconds=time_delta)
    
    return target_time_timestamp, time_delta
   
        
def calc_targetTime_timestamp_lstsq(
    coeffs: Union[List[float], Tuple[float, float, float]],
    timestamp: datetime,
    target_height: float,
    ) -> Tuple[datetime, float]:
    """
    Calculates the future timestamp at which a quadratic trajectory (z = a*t^2 + b*t + c) reaches a specified target height.
    Args:
        coeffs (list or tuple): Coefficients [a, b, c] of the quadratic equation representing the z-position over time.
        timestamp (datetime.datetime): The reference timestamp from which the time delta is calculated.
        target_height (float): The desired z-position to reach.
    Returns:
        tuple:
            target_time_timestamp (datetime.datetime): The timestamp at which the trajectory reaches the target height.
            time_delta (float): The time in seconds from the reference timestamp to the target height.
    Raises:
        ValueError: If the quadratic equation has no real solution (discriminant is negative).
    """
    # coeffs = [a, b, c] for z = a*t^2 + b*t + c
    # solve the equation a*t^2 + b*t + (c - target_height) = 0
    a, b, c = coeffs  # a = 0.5 * z_acc, b = z_velocity, c = z_pos
    
    discriminant = b**2 - 4*a*(c - target_height)
    
    if discriminant < 0:
        raise ValueError("No real solution for the quadratic equation. Discriminant is negative.")
    
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2*a)
    t2 = (-b - sqrt_discriminant) / (2*a)
    
    # choose the positive time solution
    time_delta = max(t1, t2)
    
    target_time_timestamp = timestamp + timedelta(seconds=time_delta)
    
    return target_time_timestamp, time_delta


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Example parameters
    start = [0, 0, 0]
    target = [2, 2, 1]
    flight_time = 1.5

    # Calculate initial velocity
    velocity = calc_throw_velocity(start, target, flight_time)

    print(f"Initial velocity: {velocity}")

    touchdown_time, touchdown_pos = calc_touchdown(start, velocity, target[2])

    print(f"touchdown time: {touchdown_time:.2f} seconds")
    print(f"touchdown position: {touchdown_pos}")

    assert np.allclose(touchdown_pos, target), "Touchdown position does not match target"
    assert np.isclose(touchdown_time, flight_time), f"Touchdown time {touchdown_time} does not match flight time {flight_time}"

    # Time samples
    t = np.linspace(0, flight_time, num=100)
    g = np.array([0, 0, -9.81])

    # Parabola points
    points = np.array([
        start + velocity * ti + 0.5 * g * ti**2
        for ti in t
    ])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(points[:,0], points[:,1], points[:,2], label='Parabolic trajectory')
    ax.scatter(*start, color='green', label='Start')
    ax.scatter(*target, color='red', label='Target')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
    