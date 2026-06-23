#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

def create_box_marker(min_point, max_point, marker_id=0, color=(1,0,0,1)):
    """
    Creates a cube wireframe (LINE_STRIP) from min and max points.
    """
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "hand_box"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.02  # line thickness
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = color

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    x_min, y_min, z_min = min_point
    x_max, y_max, z_max = max_point

    corners = [
        Point(x_min, y_min, z_min),
        Point(x_max, y_min, z_min),
        Point(x_max, y_max, z_min),
        Point(x_min, y_max, z_min),
        Point(x_min, y_min, z_max),
        Point(x_max, y_min, z_max),
        Point(x_max, y_max, z_max),
        Point(x_min, y_max, z_max),
    ]

    # Wireframe connections
    marker.points.extend([
        corners[0], corners[1], corners[2], corners[3], corners[0],  # bottom
        corners[4], corners[5], corners[6], corners[7], corners[4],  # top
        corners[5], corners[1], corners[2], corners[6], corners[7], corners[3]  # vertical edges
    ])

    return marker

def main():
    rospy.init_node("fake_human_hand_bbox_visualizer")
    pub = rospy.Publisher("fake_human_hand_bbox_marker_array", MarkerArray, queue_size=1)
    rate = rospy.Rate(1)

    # Get sim parameter
    rosparam_sim = rospy.get_param("sim")  # default to "sim"

    # Get fake_human_hand_box from param server
    fake_hand_box = rospy.get_param("fake_human_hand_box")

    # Choose the box depending on sim
    if rosparam_sim == "mujoco":
        # pick 'side' box for simulation
        box_list = fake_hand_box["sim"]["side"]
    elif rosparam_sim == "real":
        box_list = fake_hand_box["real"]
    else:
        raise NotImplementedError("rosparam 'sim' must be 'sim' or 'real'")

    # Convert list-of-lists into min/max points
    x_range, y_range, z_range = box_list
    min_point = [x_range[0], y_range[0], z_range[0]]
    max_point = [x_range[1], y_range[1], z_range[1]]

    # Only one marker
    marker_array = MarkerArray()
    marker_array.markers.append(create_box_marker(min_point, max_point, marker_id=0, color=(0,1,0,1)))



    while not rospy.is_shutdown():
        now = rospy.Time.now()
        for marker in marker_array.markers:
            marker.header.stamp = now
        pub.publish(marker_array)
        rate.sleep()

if __name__ == "__main__":
    main()
