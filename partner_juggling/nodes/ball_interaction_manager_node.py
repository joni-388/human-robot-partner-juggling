#!/usr/bin/env python3
import rospy
import tf2_ros
import actionlib
import numpy as np
from typing import Tuple

# msgs
from partner_juggling_msgs.msg import CatchAction, CatchGoal
from partner_juggling_msgs.msg import BallState, BallStates, BallInteractionStates, BallInteractionState
from partner_juggling.interaction_state import InteractionState
from partner_juggling_msgs.msg import HandState, HandStates

class BallInteractionManager:
    def __init__(self):
        """ Initialize the BallInteractionManager node.
        This node manages the interaction states of juggling balls based on their positions and the position of the robot's Tool Center Point (TCP) and a (simulated) human hand.
        It subscribes to the "ball_states" topic to get the states of the balls and uses tf2_ros to get the transform from the 'world' frame to the TCP frame.
        The interaction states are managed in a state machine that runs at 100 Hz.
        The interaction states are:
            - UNKNOWN: The initial state, not used in the state machine.
            - HELD_BY_HUMAN: The ball is held by a human.                                       
            - THROWN_TO_ROBOT: The ball has been thrown to the robot.
            - IN_ROBOT_INTERACTION: The ball is in the robot's interaction range.
            - THROWN_TO_HUMAN: The ball has been thrown to a human.
            - FELL_TO_GROUND: The ball has fallen to the ground.
        """
        # general varibles
        self.use_robot = False
        self.use_robot = True
        
        # ball_states topic subscriber
        rospy.Subscriber("ball_states", BallStates, self.ball_states_callback, queue_size=1)
        
        self.hand_tracking = rospy.get_param('~hand_tracking', True)
        if self.hand_tracking:
            rospy.Subscriber("hand_states", HandStates, self.hand_states_callback, queue_size=1)
        else:
            pass
        
        self.ball_states = None
        self.hand_states = None
        
        # tf2_ros buffer and listener to get the transform from 'world' to 'TCP_frame'
        self.TCP_frame = "wam_right_tool" 
        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer) 
        
        rospy.loginfo(f"Waiting for tf_{self.TCP_frame} to be available")
        while not self.tfBuffer.can_transform('world', self.TCP_frame, rospy.Time(0), rospy.Duration(1.0)):
            rospy.sleep(0.1)
        rospy.loginfo(f"tf_{self.TCP_frame} is now available")
      
        self.FIXED_Z_PLANE = rospy.get_param("FIXED_Z_PLANE")  
       
    
        
        rosparam_sim = rospy.get_param("sim")
        if rosparam_sim == "mujoco":
            self.fake_human_hand_box_constraints  = rospy.get_param("fake_human_hand_box")["sim"]["side"]
        elif rosparam_sim == "real":  
            self.fake_human_hand_box_constraints  = rospy.get_param("fake_human_hand_box")["real"]
        else:
            raise NotImplementedError("rosparam 'sim' must be either 'sim' or 'real' ")


        # publisher for ball interaction states
        self.ball_interacition_states_publisher = rospy.Publisher("ball_interaction_states", BallInteractionStates, queue_size=1)
        
        # action server client
        if self.use_robot:
            self.catch_client = actionlib.SimpleActionClient('CatchBall', CatchAction)
            self.catch_client.wait_for_server()
        
        # constants parameters 
        self.state_machine_rate = rospy.Rate(100)  # 100 Hz
        self.distance_to_hand_threshold = 0.4 
        self.distance_to_hand_threshold_hysterese_buffer = 0.1
        
        self.distance_to_tcp_threshold = 0.25 # 0.1  worked  
        self.distance_to_tcp_threshold_release_buffer = 0.05  # hysteresis [m] 

        self.ground_threshold = 0.1  # Threshold to consider a ball as fallen to the ground

        self.catch_persistence_counters = {}
        self.catch_persistence_threshold = 50
        
        self.throw_back_min_z_velocity = 0.3 # m/s

        
        # start state machine manager
        rospy.loginfo("[%s] BallInteractionManager initialized with TCP frame: %s", rospy.get_name(),self.TCP_frame)
        self.state_machine_manager()  
        
        return 
    
    def catch_server_goal_done_cb(self, status, result):
        if status == actionlib.GoalStatus.SUCCEEDED:
            rospy.loginfo("[%s] Goal succeeded!", rospy.get_name())
        else:
            rospy.logwarn(f"[{rospy.get_name()}] Goal failed with status: {status}")

    def state_machine_manager(self):
        """ Main state machine manager for handling ball interactions.
        This function runs a loop that checks the state of each ball and updates their interaction states.
        It uses the ball states from the ROS topic and the TCP position  and human hand position to determine the interaction states.
        """
        
        if self.hand_tracking:
            while self.hand_states is None and not rospy.is_shutdown():
                rospy.loginfo("[%s] Waiting for hand states to be available...", rospy.get_name())
                rospy.sleep(0.1)
        
        while self.ball_states is None and not rospy.is_shutdown():
            rospy.loginfo("[%s] Waiting for ball states to be available...", rospy.get_name())
            rospy.sleep(0.1)
        
        stop = False
        current_interaction_states = {
            ball_state.id: {"interaction_state": InteractionState.UNKNOWN,
                            "throw_start_timestamp": rospy.Time(0),
                            "predict": False
                            }
            for ball_state in self.ball_states.ball_states
        }
        
        human_hand_position = self.get_human_hand_position()
        tcp_position, tcp_orientation = self.get_TCP_position()
        
        
        # initialize states
        for ball_state in self.ball_states.ball_states:
            ball_id = ball_state.id
            ball_position = np.array([ball_state.position.x, ball_state.position.y, ball_state.position.z])
            
            # distance_to_hand = np.linalg.norm(ball_position - human_hand_position)
            distance_to_hand = self.get_distance_ball_to_human_hand(ball_position, human_hand_position)
            distance_to_tcp = np.linalg.norm(ball_position - tcp_position)

            in_human_hand = distance_to_hand < self.distance_to_hand_threshold 
            in_tcp_reach = distance_to_tcp < self.distance_to_tcp_threshold
            
            if in_human_hand:
                current_interaction_states[ball_id]["interaction_state"] = InteractionState.HELD_BY_HUMAN
            elif in_tcp_reach:
                current_interaction_states[ball_id]["interaction_state"] = InteractionState.IN_ROBOT_INTERACTION
        
            else:
                current_interaction_states[ball_id]["interaction_state"] = InteractionState.FELL_TO_GROUND
                
            # assert in_human_hand or in_tcp_reach, "Ball must be either in human hand or in TCP reach"
            # assert not(in_human_hand and in_tcp_reach), "Ball cannot be in both human hand and TCP reach at the same time"
        
        for ball_state in self.ball_states.ball_states:
            ball_id = ball_state.id
            if ball_id not in self.catch_persistence_counters:
                self.catch_persistence_counters[ball_id] = 0
            
        
        # run the state machine loop
        while not stop and not rospy.is_shutdown():
            for ball_state in self.ball_states.ball_states:
                ball_id = ball_state.id
                
                ball_position = np.array([ball_state.position.x, ball_state.position.y, ball_state.position.z])
                ball_velocity = np.array([ball_state.velocity.x, ball_state.velocity.y, ball_state.velocity.z])
                human_hand_position = self.get_human_hand_position()
                human_hand_position = np.array([human_hand_position])
                tcp_position, tcp_orientation = self.get_TCP_position()
                
                # distance_to_hand = np.linalg.norm(ball_position - human_hand_position)
                distance_to_hand = self.get_distance_ball_to_human_hand(ball_position, human_hand_position)
                distance_to_tcp = np.linalg.norm(ball_position - tcp_position)
                # velocity_zero = True #np.linalg.norm(ball_velocity) < self.velocity_threshold
                
                previous_interaction_state = current_interaction_states[ball_id]["interaction_state"]


                # debug reset
                if previous_interaction_state == InteractionState.FELL_TO_GROUND:
                    in_human_hand = distance_to_hand < self.distance_to_hand_threshold
                    if in_human_hand:
                        rospy.logwarn(f"[{rospy.get_name()}] Ball {ball_id} is in human hand after falling to ground. You can now start again. state: HELD_BY_HUMAN.")
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.HELD_BY_HUMAN
                        current_interaction_states[ball_id]["throw_start_timestamp"] = rospy.Time(0)
                    


                # InteractionState.HELD_BY_HUMAN
                if previous_interaction_state == InteractionState.HELD_BY_HUMAN:
                    if distance_to_hand > self.distance_to_hand_threshold:
                        rospy.loginfo(f"[{rospy.get_name()}] Ball {ball_id}  state: THROWN_TO_ROBOT.")
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.THROWN_TO_ROBOT
                        current_interaction_states[ball_id]["throw_start_timestamp"] = rospy.Time.now()
                
                                
                        
                # InteractionState.THROWN_TO_ROBOT        
                elif previous_interaction_state == InteractionState.THROWN_TO_ROBOT:
                    if not current_interaction_states[ball_id]["predict"]:
                        # if ball_position[2] > self.height_threshold_to_predict:
                        current_interaction_states[ball_id]["predict"] = True
                        goal = CatchGoal()
                        # goal.catch_time = -1 # not implmented 
                        if self.use_robot:
                            rospy.loginfo(f"[{rospy.get_name()}] Sending catch goal for ball {ball_id} to the robot.")
                            # self.catch_client.send_goal(goal, done_cb=self.catch_server_goal_done_cb)
                        
                      
                      
                    # Transition to in robot contact
                    if distance_to_tcp < self.distance_to_tcp_threshold:
                        rospy.loginfo(f"[{rospy.get_name()}] Ball {ball_id} entered robot contact.")
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.IN_ROBOT_CONTACT
                        current_interaction_states[ball_id]["predict"] = False
                        current_interaction_states[ball_id]["throw_start_timestamp"] = rospy.Time(0)
                        self.catch_persistence_counters[ball_id] = 0

                    # Transition to fell to ground
                    elif ball_position[2] < self.ground_threshold:
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.FELL_TO_GROUND
                        current_interaction_states[ball_id]["predict"] = False
                        current_interaction_states[ball_id]["throw_start_timestamp"] = rospy.Time(0)    
                                            
                
                # InteractionState.IN_ROBOT_CONTACT 
                elif previous_interaction_state == InteractionState.IN_ROBOT_CONTACT:
                    # Contact lost → fell                    
                    if distance_to_tcp > self.distance_to_tcp_threshold + self.distance_to_tcp_threshold_release_buffer:
                        rospy.logwarn(f"[{rospy.get_name()}] Ball {ball_id} lost robot contact.")
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.FELL_TO_GROUND
                        self.catch_persistence_counters[ball_id] = 0

                    else:
                        # Still in contact
                        self.catch_persistence_counters[ball_id] += 1

                        if self.catch_persistence_counters[ball_id] >= self.catch_persistence_threshold:
                            rospy.loginfo(f"[{rospy.get_name()}] Ball {ball_id} successfully caught.")
                            current_interaction_states[ball_id]["interaction_state"] = InteractionState.IN_ROBOT_INTERACTION

                            self.catch_persistence_counters[ball_id] = 0
    
                        
                # InteractionState.IN_ROBOT_INTERACTION  
                elif previous_interaction_state == InteractionState.IN_ROBOT_INTERACTION:

                    if ball_position[2] < self.ground_threshold:
                        rospy.logerr(f"[RESULT] Ball {ball_id}: FAILURE (fell from robot).")
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.FELL_TO_GROUND

                    elif distance_to_tcp > self.distance_to_tcp_threshold + self.distance_to_tcp_threshold_release_buffer:

                        # Decide throw-back vs drop
                        # if ball_velocity[2] > self.throw_back_min_z_velocity:
                        if ball_position[2] > self.FIXED_Z_PLANE + 0.1:
                        # if True:
                            rospy.loginfo(f"[{rospy.get_name()}] Ball {ball_id} thrown back to human.")
                            current_interaction_states[ball_id]["interaction_state"] = InteractionState.THROWN_TO_HUMAN
                        else:
                            rospy.logerr(f"[RESULT] Ball {ball_id}: FAILURE (no upward release).")
                            current_interaction_states[ball_id]["interaction_state"] = InteractionState.FELL_TO_GROUND
                        
                # InteractionState.THROWN_TO_HUMAN        
                elif previous_interaction_state == InteractionState.THROWN_TO_HUMAN:
                    if distance_to_hand < self.distance_to_hand_threshold:
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.HELD_BY_HUMAN
                    elif ball_position[2] < self.ground_threshold:
                        current_interaction_states[ball_id]["interaction_state"] = InteractionState.FELL_TO_GROUND
                        current_interaction_states[ball_id]["predict"] = False
                        current_interaction_states[ball_id]["throw_start_timestamp"] = rospy.Time(0)
        
            # publish the current interaction states    
            self.ball_interacition_states_publisher.publish(
                BallInteractionStates(
                    header=rospy.Header(stamp=rospy.Time.now()),
                    interaction_states=[
                        BallInteractionState(
                            ball_id=ball_state.id,
                            interaction_state=current_interaction_states[ball_state.id]["interaction_state"],
                            throw_start_timestamp=current_interaction_states[ball_state.id]["throw_start_timestamp"],
                            predict=current_interaction_states[ball_state.id]["predict"],
                        ) for ball_state in self.ball_states.ball_states
                    ]
                )
            )    
            self.state_machine_rate.sleep()  
        
        
        return 
    

    def ball_states_callback(self, ball_states):
        """ Callback function to handle incoming ball states and store them in the class instance.r
        Args:
            ball_states (BallStates): The message containing the states of the balls.
        """
        self.ball_states = ball_states
        
        return 
    
    def hand_states_callback(self, hand_states):
        """ Callback function to handle incoming hand states and store them in the class instance.r
        Args:
            hand_states (HandStates): The message containing the states of the hands.
        """
        self.hand_states = hand_states
        
        return 
    
    def get_TCP_position(self) -> Tuple[tuple, tuple]: 
        """ Get the position of the TCP (Tool Center Point) of the robot in the world frame by looking up the transform from 'world' to 'self.TCP_frame'.
        Returns:
            tuple: (x, y, z),(q_x, q_y, q_z, q_qw) pose of the TCP in the world frame.
        Raises:
            tf2_ros.LookupException: If the transform is not found.
            tf2_ros.ConnectivityException: If there is a connectivity issue with the transform.
            tf2_ros.ExtrapolationException: If the transform is not available at the requested time
        """
        # rospy.loginfo("[%s] Waiting for tf_ball0 to be available", rospy.get_name())
        trans_stamped = self.tfBuffer.lookup_transform('world', self.TCP_frame, rospy.Time(0), rospy.Duration(0.002))
        transform = trans_stamped.transform
        
        x = transform.translation.x
        y = transform.translation.y
        z = transform.translation.z 
        
        q_x = transform.rotation.x  
        q_y = transform.rotation.y
        q_z = transform.rotation.z
        q_w = transform.rotation.w
         
        return  (x, y, z), (q_x, q_y, q_z, q_w) 
        

    def get_human_hand_position(self):
        """ Get the position of the human hand. For now it is faked as hand tracking not implemented yet.
        Returns:                                                                                          
            tuple: (x, y, z) position of the human hand in the world frame.
        """     
        
        if self.hand_tracking:
            self.hand_states.hand_states[0]
            return (self.hand_states.hand_states[0].position.x,self.hand_states.hand_states[0].position.y,self.hand_states.hand_states[0].position.z)
        else:
            return self.fake_human_hand_position()

        
    def fake_human_hand_position(self):
        """ Fake human hand position for testing purposes.
        Returns:
            tuple: (x, y, z) position of the human hand in the world frame.
        """
        
        # Check if the ball states are available and if any ball is within the fake human hand box constraints.
        # If so, return the position of the first ball that is within the constraints.
        # If no ball is within the constraints, return a default position (2, 0, 0).
        if self.ball_states is not None:
            for ball_state in self.ball_states.ball_states:
                x, y, z = ball_state.position.x, ball_state.position.y, ball_state.position.z
                x_in = self.fake_human_hand_box_constraints[0][0] <= x <= self.fake_human_hand_box_constraints[0][1]
                y_in = self.fake_human_hand_box_constraints[1][0] <= y <= self.fake_human_hand_box_constraints[1][1]
                z_in = self.fake_human_hand_box_constraints[2][0] <= z <= self.fake_human_hand_box_constraints[2][1]
                if x_in and y_in and z_in:
                    return (x, y, z)
        return (2, 0, 0)
    
    def get_distance_ball_to_human_hand(self, ball_pos, hand_pos):
        """ Get the distance from the ball position to the human hand position.
        Args:
            ball_position (tuple): (x, y, z) position of the ball in the world frame.
        Returns:
            float: Distance from the ball position to the human hand position.
        """
        
        if self.hand_tracking:
            dist = np.linalg.norm(hand_pos-ball_pos)
            return dist

        x, y, z = ball_pos[0],ball_pos[1],ball_pos[2]
        x_in = self.fake_human_hand_box_constraints[0][0] <= x <= self.fake_human_hand_box_constraints[0][1]
        y_in = self.fake_human_hand_box_constraints[1][0] <= y <= self.fake_human_hand_box_constraints[1][1]
        z_in = self.fake_human_hand_box_constraints[2][0] <= z <= self.fake_human_hand_box_constraints[2][1]
        if x_in and y_in and z_in:
            return 0.0
        else:
            return np.inf 


if __name__ == '__main__':
    rospy.init_node('ball_interaction_manager_node')
    ball_interaction_manager = BallInteractionManager()
    rospy.spin()