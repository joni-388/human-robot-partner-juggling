#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


def create_box_marker(width, height, marker_id, color):
    marker = Marker()
    marker.header.frame_id = "world"
    marker.header.stamp = rospy.Time.now()
    
    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.0  
    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0  
    marker.pose.orientation.w = 1.0 

    marker.ns = "box"
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Line width
    marker.color.r = color[0]
    marker.color.g = color[1]
    marker.color.b = color[2]
    marker.color.a = color[3]

    x_min = -width / 2
    x_max = width / 2
    y_min = -width / 2
    y_max = width / 2
    z_min = 0.0
    z_max = height

    corners = [
        Point(x_min, y_min, z_min),  # 0
        Point(x_max, y_min, z_min),  # 1
        Point(x_max, y_max, z_min),  # 2
        Point(x_min, y_max, z_min),  # 3
        Point(x_min, y_min, z_max),  # 4
        Point(x_max, y_min, z_max),  # 5
        Point(x_max, y_max, z_max),  # 6
        Point(x_min, y_max, z_max)   # 7
    ]
    
    
    print(f"marker id: {marker_id}, width: {width}, height: {height}, color: {color}")
    for point in corners:
        print(f"Point: ({point.x}, {point.y}, {point.z})")

    marker.points.extend([
        corners[0], corners[1], corners[2], corners[3], corners[0],  # Bottom face
        corners[4],  # Up to top face
        corners[5], corners[6], corners[7], corners[4],  # Top face
        corners[5], corners[1],  # Down to bottom
        corners[2], corners[6],  # Up to top
        corners[7], corners[3]   # Down to bottom
    ])

    return marker

def main():
    rospy.init_node('box_marker_publisher')
    pub = rospy.Publisher('box_marker_array', MarkerArray, queue_size=1)
    rate = rospy.Rate(1)
    
    safety_boxes = rospy.get_param("safety_boxes")
    
    color_list = [
            (1.0, 0.5, 0.0, 1.0),  # orange
            (0.0, 0.5, 1.0, 1.0),  # blue
            (0.0, 1.0, 0.0, 1.0),  # green
            (1.0, 0.0, 0.0, 1.0),  # red
            (1.0, 1.0, 0.0, 1.0),  # yellow
            (0.5, 0.0, 1.0, 1.0),  # purple
        ]

    marker_array = MarkerArray()
    
    for box in safety_boxes:
        min_point = box['min']
        max_point = box['max']
        width = max_point[0] - min_point[0]
        height = max_point[2] - min_point[2]
        box_index = safety_boxes.index(box)
        color = color_list[box_index % len(color_list)]
        
        marker_array.markers.append(create_box_marker(width, height, box_index, color))


    while not rospy.is_shutdown():
        now = rospy.Time.now()
        for marker in marker_array.markers:
            marker.header.stamp = now
        pub.publish(marker_array)
        rate.sleep()

if __name__ == '__main__':
    main()
