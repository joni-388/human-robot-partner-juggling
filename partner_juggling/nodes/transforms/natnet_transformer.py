#!/usr/bin/env python

import rospy
import tf2_ros
import tf2_geometry_msgs

from natnet_bridge.msg import NatNetFrame
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header


def transform_point(tf_buffer, point, stamp, src_frame, target_frame):
    stamped_point = PointStamped()
    stamped_point.header.stamp = stamp
    stamped_point.header.stamp = rospy.Time(0) # takes the latest available transform
    stamped_point.header.frame_id = src_frame
    stamped_point.point = point

    try:
        # Transform point
        transformed_point = tf_buffer.transform(stamped_point, target_frame, rospy.Duration(0.0001))
        return transformed_point.point
    except Exception as e:
        rospy.logwarn("TF transform failed: {}".format(e))
        return None

class NatNetTransformer:
    def __init__(self):
        rospy.init_node("natnet_transformer")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.sub = rospy.Subscriber("/natnet_node/natnet_frame", NatNetFrame, self.callback, queue_size=1)
        self.pub = rospy.Publisher("/natnet_frame_transformed", NatNetFrame, queue_size=1)

        self.source_frame = "optitrack_custom"
        self.target_frame = "world"

    def callback(self, msg):
    
        transformed_msg = NatNetFrame()
        transformed_msg.header = msg.header
        transformed_msg.reference_frame = self.target_frame
        transformed_msg.natnet_frame_number = msg.natnet_frame_number

        # Transform unidentified_markers
        for marker in msg.unidentified_markers:
            pt = Point(x=marker.x, y=marker.y, z=marker.z)
            transformed = transform_point(
                self.tf_buffer,
                pt,
                transformed_msg.header.stamp,
                self.source_frame,
                self.target_frame
            )
            if transformed:
                transformed_msg.unidentified_markers.append(transformed)

        # Transform unlabeled_markers
        for marker in msg.unlabeled_markers:
            transformed = transform_point(
                self.tf_buffer,
                marker.position,
                transformed_msg.header.stamp,
                self.source_frame,
                self.target_frame
            )
            if transformed:
                marker.position = transformed
                transformed_msg.unlabeled_markers.append(marker)

        # (Optional) Copy other elements
        transformed_msg.marker_sets = msg.marker_sets
        transformed_msg.rigid_bodies = msg.rigid_bodies
        transformed_msg.skeletons = msg.skeletons
        transformed_msg.labeled_markers = msg.labeled_markers

        self.pub.publish(transformed_msg)

        return

if __name__ == '__main__':
    try:
        NatNetTransformer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

