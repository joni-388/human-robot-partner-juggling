import numpy as np
import casadi as cas

from trajectory_planning.trajectory import SymbolicTrajectory

from trajectory_planning.cost_functions import average_of_squared_joint_accelerations_at_breakpoints,average_of_squared_joint_jerks_at_breakpoints,integral_squared_joint_acceleration_cost

def throw_cost_fn_old(traj):
            return acceleration_plus_jerk_cost(traj, cost_scale=0.01, jerk_weight=0.05)

def throw_cost_fn(traj):
            return acceleration_plus_jerk_cost(traj, cost_scale=0.01, jerk_weight=0.2)


def acceleration_plus_jerk_cost(traj, cost_scale=0.01, jerk_weight=0.1):
    """
    Linear combination of squared acceleration and squared jerk costs.
    
    Args:
        traj: SymbolicTrajectory object
        cost_scale: Overall scaling factor for the cost (default 0.01)
        jerk_weight: Weight factor for jerk (0-1). 
                    0.0 = pure acceleration, 1.0 = pure jerk, 0.1 = 90% accel, 10% jerk
    
    Returns:
        Weighted combination of acceleration and jerk costs
    """
    accel_cost = average_of_squared_joint_accelerations_at_breakpoints(traj)
    # accel_cost = integral_squared_joint_acceleration_cost(traj)
    # jerk_cost = average_of_squared_joint_jerks_at_breakpoints_weighted(traj,cas.diag(cas.SX([10.0, 1.0, 1.0, 1.0])))
    jerk_cost = integral_squared_joint_jerk_cost_weighted(traj, cas.diag(cas.SX([10.0, 1.0, 1.0, 1.0])))

    # Linear combination: emphasize acceleration by default, add smoothing from jerk
    combined_cost = (1.0 - jerk_weight) * accel_cost + jerk_weight * jerk_cost
    return cost_scale * combined_cost
    # return cost_scale *accel_cost
    return cost_scale *jerk_cost


def integral_squared_joint_jerk_cost_weighted(ctraj: SymbolicTrajectory, weights):
    """
    Integrates the square of piecewise constant joint jerks. Normalizes by trajectory duration.
    """
    cost = 0
    for k in range(ctraj.num_intervals):
        dt = ctraj.ctime_steps[k+1] - ctraj.ctime_steps[k]
        j = ctraj.cdddq[k, :]  # row vector
        weighted_quadratic_jerks = j @ weights @ j.T
        cost += cas.sum2(weighted_quadratic_jerks * dt)
    return cost / (ctraj.ctime_steps[-1] - ctraj.ctime_steps[0])

def average_of_squared_joint_jerks_at_breakpoints_weighted(ctraj: SymbolicTrajectory, weights):
    """
    Averages the square of joint jerks at breakpoints.
    """
    cost = 0
    for k in range(ctraj.num_intervals):
        j = ctraj.cdddq[k,:]
        weighted_quadratic_jerks = j @ weights @ j.T
        cost += cas.sum2(weighted_quadratic_jerks)
    return cost / ctraj.num_intervals    