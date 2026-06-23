#!/usr/bin/env python

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import Point
import numpy as np


class CatchAreaVisualizer:
    def __init__(self, reference_frame: str = None):    
        # Publisher for MarkerArray to visualize in RViz
        self.marker_pub = rospy.Publisher('catch_area_marker_array', MarkerArray, queue_size=10)

        # Set the rate of the loop
        self.rate = rospy.Rate(1)  # 10 Hz
        self.reference_frame = "world" if reference_frame is None else reference_frame


        # ellipse defining catch area 
        self.center_x = 0
        self.center_y = -0.387
        self.a = 0.872
        self.b = 0.5
        self.z_plane = 0.57  # z-coordinate for the plane and ellipse
        
        # Update ellipse parameters with provided values
        self.center_x = 0
        self.center_y = -0.3676
        self.a = 0.87142
        self.b = 0.5269

        self.marker_array = MarkerArray()
        self.marker_array.markers.append(self.ellipse_minus_cirlce_marker())
        
        
    def is_point_in_ellipse(self,x, y, center_x, center_y, a, b):
        """
        Check if the point (x, y) lies inside or on the ellipse defined by
        center (center_x, center_y) and axes a (x direction), b (y direction).
        """
        return ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1

    def create_plane_marker(self):
        # Define the plane dimensions
        x_min = -0.1
        x_max =  0.7
        y_min = -0.7# -0.65
        y_max = 0 #-0.2
        y_min =  -0.65
        y_max = -0.2
        z_plane = 0.57

        # Calculate center and size
        width  = x_max - x_min
        height = y_max - y_min  # Note: y_max is more negative, so subtract properly
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        center_z = z_plane

        # Create the marker
        marker = Marker()
        marker.header = Header(frame_id=self.reference_frame)  # or "base_link", etc.
        marker.ns = "plane"
        marker.id = -1
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z
        
        marker.pose.orientation.w = 1.0  # No rotation

        marker.scale.x = width
        marker.scale.y = abs(height)
        marker.scale.z = 0.01  # Very thin to represent a plane
        

        marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.5)  # Semi-transparent green
        marker.lifetime = rospy.Duration(1)  # 0 = forever

        return marker

    def create_ellipse_marker(self, num_points=50):

        marker = Marker()
        marker.header = Header(frame_id=self.reference_frame)
        marker.ns = "ellipse"
        marker.id = -2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line strip properties
        marker.scale.x = 0.01  # Line width

        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)  # Red, semi-transparent

        # Generate ellipse points
        thetas = np.linspace(0, 2 * np.pi, num_points + 1)
        xs = self.center_x + self.a * np.cos(thetas)
        ys = self.center_y + self.b * np.sin(thetas)
        
        # Filter points where x > 0
        xs, ys = xs[xs > 0], ys[xs > 0]
        # filter points where inside cirle 
        mask = xs**2 + ys**2 >= 0.4**2
        xs, ys = xs[mask], ys[mask]

        # Find intersection points between the ellipse and the circle x^2 + y^2 = 0.4^2
        # Parametric ellipse: x = self.center_x + self.a * cos(t), y = self.center_y + self.b * sin(t)
        # Solve for t where (x^2 + y^2) = 0.16

        circle_thetas = []
        for t in np.linspace(0, 2 * np.pi, 500):
            x = self.center_x + self.a * np.cos(t)
            y = self.center_y + self.b * np.sin(t)
            if abs(x**2 + y**2 - 0.16) < 1e-3 and x > 0:
                circle_thetas.append(t)

        # Add these intersection points to xs and ys
        for t in circle_thetas:
            x = self.center_x + self.a * np.cos(t)
            y = self.center_y + self.b * np.sin(t)
            xs = np.append(xs, x)
            ys = np.append(ys, y)

        
        for x, y in zip(xs, ys):
            marker.points.append(
            Point(x=float(x), y=float(y), z=float(self.z_plane))
            )


        marker.lifetime = rospy.Duration(2)
        return marker


    def ellipse_minus_cirlce_marker(self, radius=0.4, num_points=50):

        marker = Marker()
        marker.header = Header(frame_id=self.reference_frame)
        marker.ns = "ellipse_minus_circle"
        marker.id = -3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.0  
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0  
        marker.pose.orientation.w = 1.0 


        # Line strip properties
        marker.scale.x = 0.01  # Line width

        marker.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.8)  # Red, semi-transparent

        # Generate ellipse points
        thetas = np.linspace(0, 2 * np.pi, num_points + 1)
        xs = self.center_x + self.a * np.cos(thetas)
        ys = self.center_y + self.b * np.sin(thetas)
        
        # Filter points where x > 0
        xs, ys = xs[xs > 0], ys[xs > 0]
        # filter points where inside cirle 
        mask = xs**2 + ys**2 >= radius**2
        xs, ys = xs[mask], ys[mask]
        false_indices = np.where(~mask)[0]
        # circle points in ellipse
        xs_circle = radius* np.cos(thetas)
        ys_circle = radius* np.sin(thetas)

        circle_points = np.array([xs_circle, ys_circle]).T

        in_my_ellipse = lambda x, y: self.is_point_in_ellipse(x, y, self.center_x, self.center_y, self.a, self.b)
        
        mask_circle = in_my_ellipse(circle_points[:, 0], circle_points[:, 1])
        false_indices_circle = np.where(~mask_circle)[0]
        circle_points = circle_points[mask_circle]
        circle_points = np.concatenate((circle_points[false_indices_circle[0]:],circle_points[:false_indices_circle[0]]))  
        circle_points = circle_points[circle_points[:, 0] > 0]  # Filter points where x > 0
        circle_points = circle_points[::-1]

        xs = np.concatenate((xs[:false_indices[0]], circle_points[:,0], xs[false_indices[0]:]))
        ys = np.concatenate((ys[:false_indices[0]], circle_points[:,1], ys[false_indices[0]:]))
        

        ###


        
        for x, y in zip(xs, ys):
            marker.points.append(
            Point(x=float(x), y=float(y), z=float(self.z_plane))
            )


        marker.lifetime = rospy.Duration(2)
        return marker

    def run(self):
        # Keep the node running and processing messages
        while not rospy.is_shutdown():
            self.marker_pub.publish(self.marker_array)
            self.rate.sleep()
        


if __name__ == '__main__':

    rospy.init_node('catch_area_visualizer')
    visualizer = CatchAreaVisualizer()
    visualizer.run()
