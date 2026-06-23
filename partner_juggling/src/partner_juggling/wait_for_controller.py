#!/usr/bin/env python
"""
This module provides a utility function to wait for a specific ROS controller to reach the 'running' state.

Functions:
    wait_for_controller(controller_name: str, timeout: float = 10.0) -> bool:
        Waits for the specified controller to become active ('running') within a given timeout period.
        Returns True if the controller is running, False if the timeout is exceeded or ROS is shut down.

Email: jonathan_rainer.lippert@stud.tu-darmstadt.de
"""
import rospy
import rosgraph
from controller_manager_msgs.srv import ListControllers

def wait_for_controller(controller_name: str, timeout: float = 10.0) -> bool:
    """
    Waits for a specific controller to reach the 'running' state within a given timeout.

    Args:
        controller_name (str): The name of the controller to wait for.
        timeout (float, optional): Maximum time to wait in seconds. Defaults to 10.0.

    Returns:
        bool: True if the controller is running within the timeout, False otherwise.
    """
    rospy.wait_for_service('/wam_right/controller_manager/list_controllers')
    list_controllers = rospy.ServiceProxy('/wam_right/controller_manager/list_controllers', ListControllers)

    start_time = rospy.Time.now()
    rate = rospy.Rate(2)  # 2 Hz

    while not rospy.is_shutdown():
        try:
            resp = list_controllers()
            for c in resp.controller:
                if c.name == controller_name and c.state == 'running':
                    rospy.loginfo(f"Controller {controller_name} is running.")
                    return True
        except rospy.ServiceException as e:
            rospy.logwarn(f"Service call failed: {e}")

        if (rospy.Time.now() - start_time).to_sec() > timeout:
            rospy.logwarn(f"Timed out waiting for controller {controller_name}")
            return False

        rate.sleep()
        

def is_ros_master_online():
    try:
        return rosgraph.is_master_online()
    except Exception:
        return False