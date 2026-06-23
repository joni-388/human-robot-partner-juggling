import numpy as np
import pinocchio as pin
from pinocchio import casadi as sym_pin
import rospy
from trajectory_planning.trajectory import Trajectory
from partner_juggling.online_test.utils import make_full_q, line_box_intersection
from typing import Sequence, List

# self.time_steps = time_steps
# self.q = q
# self.dq = dq
# self.ddq = dddq
# self.joint_names = joint_names if joint_names is not None else []

# Total joint limits
tol = np.deg2rad(9)
q1_min, q1_max = -2.6 + tol, 2.6 - tol
q2_min, q2_max = -2.0 + tol, 2.0 - tol
q3_min, q3_max = -2.8 + tol, 2.8 - tol
q4_min, q4_max = -0.9 + tol, 2.8 - tol

q_mins = np.array([q1_min, q2_min, q3_min, q4_min])
q_maxs = np.array([q1_max, q2_max, q3_max, q4_max])

# velocity_limits = np.deg2rad(2.0)  # rad/s
# acceleration_limits = np.deg2rad(5.0)  # rad/s^
# jerk_limits = np.deg2rad(10.0)  # rad/s^3

# low
# velocity_limit = 50.0  # rad/s
# acceleration_limit = 500.0  # rad/s^2
# jerk_limit = 20000
# dq_change_limit = 2.0  # rad/s

# high
velocity_limit = 90.0  # rad/s
acceleration_limit = 900.0  # rad/s^2
jerk_limit = 60000
dq_change_limit = 10.0  # rad/s

# inf for testing 
velocity_limit = np.inf   # rad/s
acceleration_limit = np.inf  # rad/s^2
jerk_limit = np.inf
dq_change_limit = np.inf  # rad/s


# print("Using custom joint limits for setup")
# print(f"q1: [{np.rad2deg(q1_min):.2f}, {np.rad2deg(q1_max):.2f}] deg")
# print(f"q2: [{np.rad2deg(q2_min):.2f}, {np.rad2deg(q2_max):.2f}] deg")
# print(f"q3: [{np.rad2deg(q3_min):.2f}, {np.rad2deg(q3_max):.2f}] deg")
# print(f"q4: [{np.rad2deg(q4_min):.2f}, {np.rad2deg(q4_max):.2f}] deg")

# custom joint limits for setup




# # Define boxes
# box1_min = np.array([-0.15, -0.15, 0.0])
# box1_max = np.array([0.15, 0.15, 1.5])
# box2_min = np.array([-0.25, -0.25, 0.0])
# box2_max = np.array([0.25, 0.25, 0.5])
# box3_min = np.array([-0.35, -0.35, 0.0])
# box3_max = np.array([0.35, 0.35, 0.3])
# boxes = [(box1_min, box1_max), (box2_min, box2_max), (box3_min, box3_max)]


# Define boxes
# box1_min = np.array([-0.21, -0.21, 0.0])
# box1_max = np.array([0.21, 0.21, 1.5])
# box2_min = np.array([-0.3, -0.3, 0.0])
# box2_max = np.array([0.3, 0.3, 0.5])
# box3_min = np.array([-0.375, -0.375, 0.0])
# box3_max = np.array([0.375, 0.375, 0.15])
# boxes = [(box1_min, box1_max), (box2_min, box2_max), (box3_min, box3_max)]


def check_trajectory(
    safety_boxes: List[dict],
    trajectory: Trajectory,
    pin_model: pin.Model,
    pin_data: pin.Data,
    ellbow_frame_id: int,
    tool_frame_id: int,
    joint_names: List[str],
    verbose: bool = True
) -> bool:
    """Check if planned trajectory is safe to apply to real robot 

    Args:
        trajectory Trajectory: Trajectory object containing time steps, joint positions, velocities, accelerations, and jerk

    Returns:
        bool: True if trajectory is safe, False otherwise
    """

    # Check if trajectory has collisions with safety boxes
    if not check_q_collision(safety_boxes, trajectory.q, pin_model, pin_data, joint_names, ellbow_frame_id, tool_frame_id):
        if verbose:
            rospy.logwarn("Trajectory has collisions with boxes, not safe to apply.")
        return False
    
    # Check if trajectory is within joint limits
    if not np.all((trajectory.q >= q_mins) & (trajectory.q <= q_maxs)):
        if verbose:
            rospy.logwarn("Joint limits exceeded in trajectory.")
        return False
    
    # Check if velocities are within limits
    if np.any(np.abs(trajectory.dq) > velocity_limit):
        if verbose:
            rospy.logwarn(f"Velocity limit exceeded in trajectory.")
            rospy.logwarn(f"Expected velocity limit: {velocity_limit}, actual max: {np.max(np.abs(trajectory.dq))}")
        return False

    # Check if accelerations are within limits    
    if np.any(np.abs(trajectory.ddq) > acceleration_limit):
        if verbose:
            rospy.logwarn(f"Acceleration limit exceeded in trajectory!")
            rospy.logwarn(f"Expected acceleration limit: {acceleration_limit}, actual max: {np.max(np.abs(trajectory.ddq))}")
        return False 
    
    # Check if jerk is within limits
    if np.any(np.abs(trajectory.dddq) > jerk_limit):
        if verbose:
            rospy.logwarn(f"Jerk limit exceeded in trajectory.")
            rospy.logwarn(f"Expected jerk limit: {jerk_limit}, actual max: {np.max(np.abs(trajectory.dddq))}")
        return False   
              

    # Check for abrupt changes in velocity ("wiggles") for each joint using numpy
    dq_diff = np.abs(np.diff(trajectory.dq, axis=0))
    if np.any(dq_diff > dq_change_limit):
        if verbose:
            idx = np.argwhere(dq_diff > dq_change_limit)[0]
            rospy.logwarn(f"Trajectory has abrupt change in velocity for joint {idx[1]} at time step {idx[0]+1}.")
            rospy.logwarn(f"Change in velocity: {dq_diff[idx[0], idx[1]]}, limit: {dq_change_limit}")
        return False
    
    if verbose:
        rospy.loginfo("Trajectory is safe to apply.")
    return True  


def check_q_collision(
    safety_boxes: List[dict],
    qs: Sequence[np.ndarray],
    pin_model: pin.Model,
    pin_data: pin.Data,
    joint_names: List[str],
    ellbow_frame_id: int,
    tool_frame_id: int
) -> bool:
    """
    Checks for collisions between the line segment defined by the ellbow and tool frames and a set of boxes for a sequence of joint configurations.
    Args:
        qs (Sequence[np.ndarray]): Sequence of joint configurations to check.
        pin_model (pin.Model): Pinocchio robot model.
        pin_data (pin.Data): Pinocchio data structure for the model.
        joint_names (List[str]): List of joint names corresponding to the configuration vectors.
        ellbow_frame_id (int): Frame ID of the ellbow.
        tool_frame_id (int): Frame ID of the tool.
    Returns:
        bool: True if no collision is detected for any configuration in qs, False if a collision is detected.
    """
    for q in qs:
        q_full, q_ids = make_full_q(q, pin_model, joint_names)
        pin.forwardKinematics(pin_model, pin_data, q_full)
        pin.updateFramePlacements(pin_model, pin_data)
        t_ellbow = pin_data.oMf[ellbow_frame_id].translation
        t_tool = pin_data.oMf[tool_frame_id].translation
        for box in safety_boxes:
            box_min = box['min']
            box_max = box['max']
            entry_exit = line_box_intersection(t_ellbow, t_tool, box_min, box_max)
            if entry_exit is not None:
                print(f"Collision detected between ellbow and tool with box {box_min} - {box_max}")
                return False
        
        # # maybe faster    
        # if any(line_box_intersection(t_ellbow, t_tool, box_min, box_max) is not None for box_min, box_max in boxes):
        #     print("Collision detected between ellbow and tool")
        #     return False
    return True
