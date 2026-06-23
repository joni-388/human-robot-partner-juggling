"""
"""

from typing import List
import numpy as np
import rospy
import trajectory_planning
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import matplotlib.pyplot as plt
import numpy as np

from trajectory_planning.trajectory import Trajectory
from partner_juggling.menus import flush_input
from control_msgs.msg import JointTrajectoryControllerState

def goto_joint(pos, command_pub, joint_names, duration=5, blocking=False):
    """ Send a trajectory command to the robot to move to a joint position.

    Args:
        pos         (np.array):        target joint position in radians
        command_pub (rospy.Publisher): publisher for JointTrajctory commands
        joint_names (list):            names of the joints in trajectory controller
        duration    (float, optional): duration of the trajectory in seconds (default: 5)
    """
    cmd_topic = getattr(command_pub, "name", None)
    if cmd_topic is None:
        cmd_topic = getattr(command_pub, "resolved_name", None)

    q_des = None
    qd_des = None
    if cmd_topic is not None and cmd_topic.endswith("/command"):
        state_topic = cmd_topic[:-8] + "/state"
        try:
            state_msg = rospy.wait_for_message(state_topic, JointTrajectoryControllerState, timeout=1.0)
            if state_msg.desired.positions:
                q_des = np.array(state_msg.desired.positions)
            if state_msg.desired.velocities:
                qd_des = np.array(state_msg.desired.velocities)
        except rospy.ROSException:
            pass

    if q_des is None:
        q_des = np.array(pos)
    if qd_des is None:
        qd_des = np.zeros(len(joint_names))

    msg = build_trajectory_msg(joint_names=joint_names,
                               t0=rospy.Time.now() + rospy.Duration(0.05),
                               time_steps_since_t0=np.array([0.0, duration]),
                               pos=np.array([q_des, pos]),
                               vel=np.array([qd_des, np.zeros(len(joint_names))]))
    command_pub.publish(msg)
    if blocking:
        rospy.sleep(duration)




def execute_trajectory_loop(q_0, traj_cmd_topic_name, joint_names, traj):
    rospy.init_node('single_traj_publish_node')
    traj_cmd_pub = rospy.Publisher(traj_cmd_topic_name, JointTrajectory, queue_size=1)
    rospy.sleep(1)
    
    # flush_input()
    # user_input = input("Press 'e' to execute the trajectory again, or press Enter to exit: ")
    # if user_input.strip().lower() == '':
    #     print("Exiting.")
    #     return
    # if user_input.strip().lower() == 'e':
    #     goto_joint(q_0, traj_cmd_pub, joint_names, blocking=True, duration=3)

    while True:
        flush_input()
        user_input = input("Press 'e' to execute the trajectory again, or press Enter to exit: ")
        if user_input.strip().lower() == '':
            print("Exiting.")
            break
        if user_input.strip().lower() == 'e':
            goto_joint(q_0, traj_cmd_pub, joint_names, blocking=True, duration=3)
            traj_msg = build_trajectory_msg_from_traj(traj, rospy.Time.now(), joint_names)
            traj_cmd_pub.publish(traj_msg)
            rospy.loginfo("Published trajectory again.")
            

def build_trajectory_msg(joint_names: List[str],
                         t0: rospy.Time, time_steps_since_t0: np.ndarray,
                         pos: np.ndarray, vel: np.ndarray=None, acc: np.ndarray=None) -> JointTrajectory:
    """ Build a JointTrajectory from joint positions, velocities and accelerations.

    Args:
        joint_names (list):               names of the joints in trajectory controller
        t0          (rospy.Time):         start time of the trajectory
        time_steps_since_t0 (np.array):   (num_steps,) time steps in seconds since t0
        pos         (np.array):           (num_steps, num_dof) joint positions in radians
        vel         (np.array, optional): (num_steps, num_dof) joint velocities in rad/s (default: None)
        acc         (np.array, optional): (num_steps, num_dof) joint accelerations in rad/s^2 (default: None)

    Returns:
        JointTrajectory: trajectory message for ROS
    """
    assert pos.ndim == 2
    assert vel is None or vel.ndim == 2
    assert acc is None or acc.ndim == 2
    assert pos.shape[0] == len(time_steps_since_t0)
    assert vel is None or vel.shape[0] == len(time_steps_since_t0)
    assert acc is None or acc.shape[0] == len(time_steps_since_t0)

    msg = JointTrajectory()
    msg.joint_names = joint_names
    msg.header.stamp = t0
    for i in range(len(time_steps_since_t0)):
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration.from_sec(time_steps_since_t0[i])
        if not pos is None:
            point.positions = pos[i]
        if not vel is None:
            point.velocities = vel[i]
        if not acc is None:
            point.accelerations = acc[i]
        msg.points.append(point)
    return msg


def build_trajectory_msg_from_traj(traj: trajectory_planning.Trajectory, t0: rospy.Time, joint_names: List[str]) -> JointTrajectory:
    """ Build a JointTrajectory from a trajectory_planning.Trajectory object.

    Args:
        traj (trajectory_planning.Trajectory): trajectory to be converted
        t0   (rospy.Time):                     start time of the trajectory
        joint_names (list):                    names of the joints in trajectory controller

    Returns:
        JointTrajectory: trajectory message for ROS
    """
    return build_trajectory_msg(joint_names, t0, traj.time_steps, traj.q, traj.dq, traj.ddq)



def interpolate_linear(t, t0, q0, dq0, ddq0, t1, q1, dq1, ddq1):
        # normalize alpha in [0,1]
        alpha = (t - t0) / float(t1 - t0)
        alpha = np.clip(alpha, 0.0, 1.0)

        q  = (1 - alpha) * np.array(q0)  + alpha * np.array(q1)
        dq = (1 - alpha) * np.array(dq0) + alpha * np.array(dq1)
        ddq= (1 - alpha) * np.array(ddq0)+ alpha * np.array(ddq1)

        return q, dq, ddq


def get_joint_states_from_traj_msg(traj_msg, time: rospy.Time):
    traj_start_time = traj_msg.header.stamp

    previous_point = None
    current_point = None

    for point in traj_msg.points:
        point_time = traj_start_time + point.time_from_start
        if point_time > time:
            current_point = point
            break
        if point_time == time:
            return point.positions,point.velocities, point.accelerations
        else:
            previous_point = point                

    if previous_point is not None and current_point is not None:
        
        # convert everything to float seconds
        t  = (time - traj_start_time).to_sec()
        t0 = previous_point.time_from_start.to_sec()
        t1 = current_point.time_from_start.to_sec()

        q, dq, ddq = interpolate_linear(
            t=t,
            t0=t0, q0=previous_point.positions, dq0=previous_point.velocities, ddq0=previous_point.accelerations,
            t1=t1, q1=current_point.positions, dq1=current_point.velocities, ddq1=current_point.accelerations,
        )
        return q, dq, ddq
    
    # time higher than last traj point
    if previous_point is not None and current_point is None:
        print("time higher than last traj point -> last traj point used!")
        point = traj_msg.points[-1]
        return point.positions,point.velocities, point.accelerations
        

    raise ValueError(
        f"get_joint_states_from_traj_msg failed: previous_point={previous_point}, current_point={current_point} point is before trajecotry"
    )
    return


def get_joint_states_from_saved_traj(traj, time: float):

    previous_point_idx = None
    current_point_idx = None

    for p_idx in range(len(traj.q)):
        point_time = traj.time_steps[p_idx]
        if point_time > time:
            current_point_idx = p_idx
            break
        if point_time == time:
            return traj.q[p_idx], traj.dq[p_idx], traj.ddq[p_idx]
        else:
            previous_point_idx = p_idx               

    if previous_point_idx is not None and current_point_idx is not None:
        
        t0 = traj.time_steps[previous_point_idx]
        t1 = traj.time_steps[current_point_idx]

        q0 = traj.q[previous_point_idx]
        dq0 = traj.dq[previous_point_idx]
        ddq0 = traj.ddq[previous_point_idx]
        
        q1 = traj.q[current_point_idx]
        dq1 = traj.dq[current_point_idx]
        ddq1 = traj.ddq[current_point_idx]
        

        q, dq, ddq = interpolate_linear(
            t=time,
            t0=t0, q0=q0, dq0=dq0, ddq0=ddq0,
            t1=t1, q1=q1, dq1=dq1, ddq1=ddq1,
        )
        return q, dq, ddq
    
    # time higher than last traj point
    if previous_point_idx is not None and current_point_idx is None:
        print("time higher than last traj point -> last traj point used!")
        return traj.q[-1],traj.dq[-1], traj.ddq[-1]
        
    raise ValueError(
        f"get_joint_states_from_traj_msg failed: previous_point={previous_point_idx}, current_point={current_point_idx}"
    )
    return



def plot_joint_trajectories(last_traj_msgs, traj_msg):
    """
    Plots joint trajectories for multiple previous trajectories and a new trajectory for comparison.

    Args:
        last_traj_msgs (list): List of previous JointTrajectory messages.
        traj_msg (JointTrajectory): The new JointTrajectory message.
    """
    joint_names = traj_msg.joint_names
    n_joints = len(joint_names)

    def extract_time_and_pos(traj_msg):
        stamp = traj_msg.header.stamp.to_sec() - rospy.Time.now().to_sec()
        times = [p.time_from_start.to_sec() for p in traj_msg.points]
        times = np.array(times) + stamp
        pos = np.array([p.positions for p in traj_msg.points])
        return np.array(times), pos

    fig, axes = plt.subplots(n_joints, 1, figsize=(8, 2 * n_joints), sharex=True)
    if n_joints == 1:
        axes = [axes]

    # Plot all previous trajectories
    for idx, last_traj_msg in enumerate(last_traj_msgs):
        times_last, pos_last = extract_time_and_pos(last_traj_msg)
        for i, ax in enumerate(axes):
            ax.plot(times_last, pos_last[:, i], label=f"Last trajectory {idx+1}", marker="o", alpha=0.5)

    # Plot the new trajectory
    times_new, pos_new = extract_time_and_pos(traj_msg)
    for i, ax in enumerate(axes):
        ax.plot(times_new, pos_new[:, i], label="New trajectory", marker="x", linewidth=2)

        ax.set_ylabel(f"{joint_names[i]} [rad]")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("Joint Trajectories: Last vs New (Replanning)")
    plt.tight_layout()
    plt.savefig("/home/jonathan/ias/catkin_ws/src/joint_trajectories_comparison.png")
    rospy.logwarn("Joint trajectories plot saved to /home/jonathan/ias/catkin_ws/src/joint_trajectories_comparison.png")


def combine_trajectories(traj1, traj2, joint_names):
    traj_time_steps = np.concatenate([traj1.time_steps, traj2.time_steps[1:]])
    traj_q = np.vstack([traj1.q, traj2.q[1:]])
    traj_dq = np.vstack([traj1.dq, traj2.dq[1:]])
    traj_ddq = np.vstack([traj1.ddq, traj2.ddq[1:]])
    traj_dddq = np.vstack([traj1.dddq, traj2.dddq])
    return Trajectory(
        time_steps=traj_time_steps,
        q=traj_q,
        dq=traj_dq,
        ddq=traj_ddq,
        dddq=traj_dddq,
        joint_names=joint_names
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    q0=[0.0, 1.0]
    dq0=[0.0, 0.0]
    ddq0=[0.0, 0.0]
    
    q1=[1.0, 2.0]
    dq1=[0.1, 0.1]
    ddq1=[0.0, 0.0]
    
    q, dq, ddq = interpolate_linear(
        t=1.7,
        t0=1.0, q0=[0.0, 1.0], dq0=[0.0, 0.0], ddq0=[0.0, 0.0],
        t1=2.0, q1=[1.0, 2.0], dq1=[0.1, 0.1], ddq1=[0.0, 0.0]
    )
    
    ps = np.array([q0,q1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(ps[:,0], ps[:,1])
    plt.scatter(q[0],q[1], color='red')
    plt.show()     
    
    
    




    # Create a fake trajectory (1 DOF, 2 points)
    traj = JointTrajectory()
    traj.header.stamp = rospy.Time(0.0)

    p0 = JointTrajectoryPoint()
    p0.positions = [0.0]
    p0.velocities = [0.0]
    p0.accelerations = [0.0]
    p0.time_from_start = rospy.Duration(0.0)

    p1 = JointTrajectoryPoint()
    p1.positions = [1.0]
    p1.velocities = [-1.0]
    p1.accelerations = [2.0]
    p1.time_from_start = rospy.Duration(2.0)
    
    p1 = JointTrajectoryPoint()
    p1.positions = [2.0]
    p1.velocities = [-2.0]
    p1.accelerations = [4.0]
    p1.time_from_start = rospy.Duration(3.0)

    traj.points = [p0, p1]

    # Evaluate across time
    times = np.linspace(0, 3, 50)
    q_vals, dq_vals, ddq_vals = [], [], []

    for t in times:
        q, dq, ddq = get_joint_states_from_traj_msg(traj, rospy.Time(t))
        q_vals.append(q[0])
        dq_vals.append(dq[0])
        ddq_vals.append(ddq[0])

    # -------------------------
    # Plot results
    # -------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(times, q_vals, label="q (position)")
    plt.plot(times, dq_vals, label="dq (velocity)")
    plt.plot(times, ddq_vals, label="ddq (acceleration)")
    plt.xlabel("time [s]")
    plt.ylabel("value")
    plt.title("Linear interpolation from trajectory message")
    plt.legend()
    plt.grid()
    plt.show()
