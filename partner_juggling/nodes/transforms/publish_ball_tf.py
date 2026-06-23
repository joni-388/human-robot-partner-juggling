#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import Point,Point32
import geometry_msgs.msg
import numpy as np

from natnet_bridge.msg import NatNetFrame
from std_msgs.msg import Header
from natnet_bridge.msg import  LabeledMarker, MarkerSet, RigidBody, Skeleton, LabeledMarker


noise_std_x = 0.002
noise_std_y = 0.002
noise_std_z = 0.004


def main():    
    rospy.init_node('translation_publisher') 
    
    pub = rospy.Publisher("/natnet_node/natnet_frame", NatNetFrame, queue_size=10)
    
    n_balls = rospy.get_param("n_balls")
    
    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)


    rate = rospy.Rate(100) # 500 hz 
    first_time = True
    while not rospy.is_shutdown():
        if first_time:
            rospy.loginfo("Waiting for tf_ball0 to be available")
            for i in range(n_balls):
                while not tfBuffer.can_transform('world', f"tf_ball{i}", rospy.Time(0), rospy.Duration(1.0)):
                    if rospy.is_shutdown():
                        return
                    rate.sleep()
            first_time = False
            rospy.loginfo(f"tf_ball{i} is now available")
        
        
        msg = NatNetFrame()
    
        # Header
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "world"
        
        msg.marker_sets = []
        msg.unidentified_markers = []
        msg.rigid_bodies = []
        msg.skeletons = []
        msg.labeled_markers = []
                        
        msg.unlabeled_markers = []
            
        
        
        for i in range(n_balls):
            try:
                trans = tfBuffer.lookup_transform('world', f"tf_ball{i}", rospy.Time(0), rospy.Duration(0.002))
                
                position = Point32()
                position.x = trans.transform.translation.x
                position.y = trans.transform.translation.y
                position.z = trans.transform.translation.z
                
                position = Point32()
                position.x = trans.transform.translation.x + np.random.normal(0.0, noise_std_x)
                position.y = trans.transform.translation.y + np.random.normal(0.0, noise_std_y)
                position.z = trans.transform.translation.z + np.random.normal(0.0, noise_std_z)

                
  
                # Unlabeled markers
                ulm = LabeledMarker()
                ulm.model_id = 0
                ulm.marker_id = i
                ulm.position = position
                ulm.size = 0.05
                msg.unlabeled_markers.append(ulm)
                
            except Exception as e:
                rospy.logwarn(str(e))
            
        pub.publish(msg)   
        rate.sleep()






if __name__ == '__main__':
    main()
