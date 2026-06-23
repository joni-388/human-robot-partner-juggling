#!/usr/bin/env python3
import rospy
import numpy as np


from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA

# msgs
from partner_juggling_msgs.msg import BallInteractionStates, BallInteractionState
from partner_juggling_msgs.msg import TDPrediction, TDPredictions
from partner_juggling.interaction_state import InteractionState


from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA

class JugglingHUD:
    def __init__(self):
        rospy.Subscriber("ball_interaction_states", BallInteractionStates, self.state_callback)
        self.hud_pub = rospy.Publisher("juggling_hud", Marker, queue_size=1, latch=True)
        # Use a dictionary to keep track of the latest state for each ball
        self.ball_labels = {} 

    def state_callback(self, msg):
        # 1. Update internal tracking
        for s in msg.interaction_states:
            # Convert the enum integer to a readable string
            state_name = InteractionState(s.interaction_state).name
            self.ball_labels[s.ball_id] = state_name

        # 2. Build one big string for the HUD
        hud_text = "JUGGLING STATUS\n---------------\n"
        # Sort by ID so the list doesn't jump around
        for bid in sorted(self.ball_labels.keys()):
            hud_text += f"Ball {bid}: {self.ball_labels[bid]}\n"

        self.publish_hud(hud_text)

    def publish_hud(self, text):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "hud"
        marker.id = 999 # Unique ID for HUD
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position this high up and slightly to the side 
        # so it's out of the way of the juggling area
        marker.pose.position.x = 0.0
        marker.pose.position.y = 1.5
        marker.pose.position.z = 2.5
        
        marker.text = text
        marker.scale.z = 0.15 # Text size
        marker.color = ColorRGBA(1.0, 1.0, 1.0, 1.0) # White
        
        self.hud_pub.publish(marker)



if __name__ == '__main__':
    rospy.init_node('Juggling_HUD')
    node = JugglingHUD()
    rospy.spin()
