#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3, Point
from partner_juggling_msgs.msg import BallState, BallStates
import argparse
marker_duration = rospy.Duration(0.1)

class BallStateVisualizer:
    def __init__(self, reference_frame: str):    
        # Publisher for MarkerArray to visualize in RViz
        self.marker_pub = rospy.Publisher('ball_marker_array', MarkerArray, queue_size=1)

        # Subscriber to the BallStates topic
        self.ball_states_sub = rospy.Subscriber('/ball_states', BallStates, self.ball_states_callback)

        # Set the rate of the loop
        self.rate = rospy.Rate(30)  # 10 Hz
        self.reference_frame = reference_frame

    def ball_states_callback(self, msg):
        # Create a MarkerArray to store markers for all BallStates
        marker_array = MarkerArray()

        # For each BallState, create a marker and add it to the MarkerArray
        for i, ball_state in enumerate(msg.ball_states):
            marker, marker_text  = self.create_marker(ball_state.position, ball_state.id)
            marker_array.markers.append(marker)
            marker_array.markers.append(marker_text)

        # Publish the MarkerArray
        self.marker_pub.publish(marker_array)

    def create_marker(self, position, marker_id):
        
        # Create a Marker message
        marker = Marker()

        # Set the marker type to SPHERE for visualization
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set the ID (unique for each marker)
        marker.id = marker_id

        # Set the position of the marker
        marker.pose.position = position
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # Set the scale of the sphere (optional)
        marker.scale = Vector3(0.1, 0.1, 0.1)  # Example: 0.1m in each dimension

        # Set the color of the sphere (RGBA format)
        marker.color.r = 1.0  # Red
        marker.color.g = 0.0  # Green
        marker.color.b = 0.0  # Blue
        marker.color.a = 1.0  # Alpha (transparency)

        # Set the frame ID and timestamp for RViz
        marker.header.frame_id = self.reference_frame  # Replace "world" with the frame you're using
        marker.header.stamp = rospy.Time.now()
        marker.lifetime = marker_duration #rospy.Duration(0.1)  # 0.1s
        
        
        # add text
        position_copy = Point(position.x, position.y, position.z)
        marker_text = Marker()
        marker_text.type = Marker.TEXT_VIEW_FACING
        marker_text.action = Marker.ADD
        marker_text.id = marker_id + 1000
        marker_text.pose.position = position_copy
        marker_text.pose.position.z =  marker_text.pose.position.z  + 0.15
        marker_text.pose.orientation.x = 0.0
        marker_text.pose.orientation.y = 0.0
        marker_text.pose.orientation.z = 0.0
        marker_text.pose.orientation.w = 1.0
        # marker_text.scale.z = 0.5
        s=1
        marker_text.scale = Vector3(0.1 *s, 0.1*s, 0.2*s)
        marker_text.color.r = 1.0
        marker_text.color.g = 1.0
        marker_text.color.b = 1.0
        marker_text.color.a = 1.0
        marker_text.text = str(marker_id)
        marker_text.header.frame_id = self.reference_frame
        marker_text.header.stamp = rospy.Time.now()
        marker_text.lifetime = marker_duration #rospy.Duration(0.1)
        
        

        return marker, marker_text

    def run(self):
        # Keep the node running and processing messages
        rospy.spin()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Ball State Visualizer")
    parser.add_argument("--reference_frame", default="world", help="Reference frame for the markers: normally 'map' or 'world'")
    args, unknown = parser.parse_known_args() 
    reference_frame = args.reference_frame
    
    # Create the BallStateVisualizer object and run it
    rospy.init_node('ball_state_visualizer')
    visualizer = BallStateVisualizer(reference_frame=reference_frame)
    visualizer.run()
