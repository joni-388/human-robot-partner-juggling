#!/usr/bin/env python3
import time
import rospy
import numpy as np
from datetime import datetime, timedelta
import tf2_ros
import gc
# msgs
from geometry_msgs.msg import Vector3
from visualization_msgs.msg import Marker, MarkerArray
from natnet_bridge.msg import NatNetFrame
from partner_juggling_msgs.msg import BallInteractionStates
from partner_juggling_msgs.msg import BallState, BallStates, TDPrediction, TDPredictions
from geometry_msgs.msg import PointStamped, Point
import tf2_geometry_msgs

# partner_juggling_utils imports
from partner_juggling.ball_parabola import calc_targetTime_timestamp_kalman, calc_targetTime_timestamp_lstsq

# tracker imports
from partner_juggling.tracker.multi_target_tracker import MultiTargetTracker, Mahalanobis, Euclidean
from partner_juggling.tracker.data_types import Detection, StateVector, State
from partner_juggling.tracker.least_square import predict_with_lstsq #, predict_with_lstsq_const_acc

# interaction state
from partner_juggling.interaction_state import InteractionState



class OptitrackBallTracker:
    def __init__(self):
        # general params 
        n_balls = rospy.get_param("n_balls")
        n_dims = 3
        self.FIXED_Z_PLANE = rospy.get_param("FIXED_Z_PLANE")        
        # self.height_threshold = rospy.get_param("height_threshold") # Height threshold to consider a ball as in the air
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(5.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.source_frame = "optitrack_custom"
        self.target_frame = "world"


        # evauluation purposes
        self.kalman_prediction = None
        self.lstsq_prediction = None
        self.actual_detection = None

        # variables for initialization
        self.init_poses = []                            #Initial positions of the balls
        self.init_velocities = [0.0,0.0,0.0]*n_balls    #Initial velocities of the balls
        self.init_accel = [0.0,0.0,0.0]*n_balls         #Initial accelerations of the balls
        self.init_times = []
        
        # subscripe to otpitrack detections
        rospy.Subscriber("/natnet_node/natnet_frame", NatNetFrame, self.natnet_frame_callback, queue_size=1) 


        # subscripte to interaction states
        rospy.Subscriber("ball_interaction_states", BallInteractionStates, self.ball_interaction_states_callback, queue_size=1)

        # ball states publisher
        self.ball_publisher = rospy.Publisher("ball_states", BallStates, queue_size=1)
        
        # marker publisher
        self.marker_pub = rospy.Publisher("target_and_prediction_marker", MarkerArray, queue_size=1)
        self.detection_marker_pub = rospy.Publisher("detection_marker", MarkerArray, queue_size=1)
        self.publish_markers([0,0,0], [0,0,0]) 
        self.kalman_prediction_pub = rospy.Publisher("kalman_prediction_marker", MarkerArray, queue_size=1)
        
        
        # TD prediction publisher
        self.TD_prediction_pub = rospy.Publisher('Ball_TD_position_prediction', TDPredictions, queue_size=1)
                 
        
        # prediction mode 
        self.prediction_mode = "lstsq" # "kalman" or "lstsq"
        # self.prediction_mode = "kalman"
        assert self.prediction_mode in ["kalman", "lstsq"], "Prediction mode must be either 'kalman' or 'lstsq'"
        
        # tracking mode
        self.use_no_acc_prediction = True
        self.approximate_velocity_measurement = False

        # tracking mode params
        self.min_marker_size = 0.03 #  in meter
        self.optitrack_detections_dict = None  
        self.max_time_last_detection = 0.018 # normal timediff = 0.004 # 0.05 sec -> 10 missed detections with 250 hz
        self.max_distance_last_detection = 0.05 # in meter for norm of x,y,z
        
        # init interaction states
        self.ball_interaction_states = None
    
        """Initialize tracker"""
        self.ball_init_done = False
        self.start_time = None
        print("[optitrack_tracker] Waiting for natnet_frame callback to set initial ball position")
        while self.start_time is None:
            rospy.sleep(1.5)
    
        state_init = []
        for i in range(n_dims*n_balls):
            if self.approximate_velocity_measurement:
                state_init.append([self.init_poses[i+n_dims*n_balls]])
                state_init.append([(self.init_poses[i+n_dims*n_balls] - self.init_poses[i])/(self.init_times[1]-self.init_times[0]).total_seconds() ])
                state_init.append([self.init_accel[i]])
            else:
                state_init.append([self.init_poses[i]])
                state_init.append([self.init_velocities[i]])
                state_init.append([self.init_accel[i]])
            

        # noise values for tracker model (common sense)
        measurement_noise_var = 2e-5  # optitrack less then 0.2mm -> 0.0002 m -> 2e-4 | maybe worse than that but never better
        init_state_noise_var = 0.1    # 10 cm at beginning -> 0.1 m 
        transition_model_noise_diff_coeff_const_pos = 1e-2 # 1 cm per step -> 1e-2 m
        transition_model_noise_diff_coeff_const_pos = 1e-1 # higher noise works better for simulation as ball artifically jumps
        transition_model_noise_diff_coeff_const_acc = 1e-3 # 1 mm per step -> 1e-3 m
        
        multi_target_tracker_params = {
                                        "transition_model_noise_diff_coeff_const_pos": transition_model_noise_diff_coeff_const_pos,
                                        "transition_model_noise_diff_coeff_const_acc": transition_model_noise_diff_coeff_const_acc,
                                        "measurement_noise_var": measurement_noise_var,
                                        "init_state_noise_var": init_state_noise_var,
                                        "missed_distance": 3000,
                                        "transition_model_constant_derivative": 2,
                                        "distance_measure": Mahalanobis()}
        self.multi_target_tracker = MultiTargetTracker(self.start_time,
                                                       state_init, multi_target_tracker_params,
                                                       n_balls,
                                                       n_dims,
                                                       use_no_acc_prediction = self.use_no_acc_prediction,
                                                       approximate_velocity_measurement = self.approximate_velocity_measurement
)
        self.ball_init_done = True

        print('[optitrack_tracker] Optitrack Ball Tracker Initialized')
        return 
       
 
    def natnet_frame_callback(self, data_frame):
        """
        Callback function to handle NatNetFrame messages from the Optitrack system.
        This function processes the data frame, extracts marker positions, and updates the multi-target tracker.
        It also publishes the ball states and handles the prediction of ball positions.
        """   
        
        measure_callback_runtime = False
        # measure_callback_runtime = True
        if measure_callback_runtime:
            start = time.time()
        
        # Transform unlabeled_markers
        for marker in data_frame.unlabeled_markers:
            transformed = self.transform_point(
                marker.position,
                data_frame.header.stamp,
                self.source_frame,
                self.target_frame
            )
            if transformed:
                marker.position = transformed

        
        timestamp = datetime.fromtimestamp(data_frame.header.stamp.to_sec()) 
        
        # initialize the ball positions
        if self.ball_init_done is False:
            time.sleep(0.1) # wait for initialization
            self.start_time = timestamp
            for marker in data_frame.unlabeled_markers:            
                if marker.size > self.min_marker_size: # 0.02
                    self.init_poses.append(marker.position.x)
                    self.init_poses.append(marker.position.y)
                    self.init_poses.append(marker.position.z)
            self.init_times.append(timestamp) 
            return  
                
        for track in self.multi_target_tracker.tracks:
            if self.ball_interaction_states is not None:  
                if track.id in self.ball_interaction_states:
                    interaction_state = self.ball_interaction_states[track.id]['interaction_state']
                    
                    if interaction_state == InteractionState.HELD_BY_HUMAN or interaction_state ==InteractionState.IN_ROBOT_INTERACTION or interaction_state ==InteractionState.FELL_TO_GROUND:
                        track.transition_model = self.multi_target_tracker.transition_models["ConstantPosition"]
                        # .track.states[-1][1] = 0 # x
                        # .track.states[-1][1] = 0 # x
                    elif interaction_state == InteractionState.THROWN_TO_ROBOT or interaction_state ==InteractionState.THROWN_TO_HUMAN:
                        track.transition_model = self.multi_target_tracker.transition_models["ConstantVelocityXYConstantAccelerationZ"]

        
        
        # no velocity measurement
        if not self.approximate_velocity_measurement:
            self.optitrack_detections_dict = {marker.marker_id: (timestamp,[marker.position.x, marker.position.y, marker.position.z]) 
                                         for marker in data_frame.unlabeled_markers if marker.size > self.min_marker_size}
        
        # approximate velocity measurement
        else:
            if self.optitrack_detections_dict is None:
                x_velocity = 0.0
                y_velocity = 0.0
                z_velocity = 0.0
                self.optitrack_detections_dict = {marker.marker_id: (timestamp,[marker.position.x, x_velocity, marker.position.y, y_velocity, marker.position.z, z_velocity]) for marker in data_frame.unlabeled_markers if marker.size > self.min_marker_size}
            
            else:
                new_ids = set([marker.marker_id for marker in data_frame.unlabeled_markers])
                matched_ids = set() #[52]
                for marker in data_frame.unlabeled_markers: 
                    if marker.size > self.min_marker_size: # 0.02
                        if marker.marker_id in self.optitrack_detections_dict:
                            time_diff = (timestamp - self.optitrack_detections_dict[marker.marker_id][0]).total_seconds()
                            if time_diff < (self.max_time_last_detection):  # own id must be closer in time as other ids otherwise not working correctly
                                x_velocity = (marker.position.x - self.optitrack_detections_dict[marker.marker_id][1][0]) / time_diff
                                y_velocity = (marker.position.y - self.optitrack_detections_dict[marker.marker_id][1][2]) / time_diff
                                z_velocity = (marker.position.z - self.optitrack_detections_dict[marker.marker_id][1][4]) / time_diff

                            else:
                                # print("[optitrack_tracker] Time difference too large:", time_diff)
                                # delete the marker from the last detections as it is not valid anymore
                                del self.optitrack_detections_dict[marker.marker_id]

                        if marker.marker_id not in self.optitrack_detections_dict:
                            
                            # TODO: maybe add if multiple detections are close to same remaining id 
                            
                            min_distance = float('inf')
                            closest_id = None
                            
                            remaining_ids = set(self.optitrack_detections_dict.keys()) - new_ids - matched_ids
                            
                            for remaining_id in remaining_ids:
                                # check if the remaining id is in the last detection
                                    time_diff = (timestamp - self.optitrack_detections_dict[remaining_id][0]).total_seconds()
                                    if time_diff < self.max_time_last_detection:
                                        
                                        x,_, y, _, z, _ = self.optitrack_detections_dict[remaining_id][1]

                                        distance = np.linalg.norm(np.array([marker.position.x, marker.position.y, marker.position.z]) - np.array([x, y, z]))
                                        # Check if this is the closest so far
                                        if distance < min_distance:
                                            min_distance = distance
                                            closest_id = remaining_id
                                    
                                    else:
                                        # delete the marker from the last detections as it is not valid anymore
                                        del self.optitrack_detections_dict[remaining_id]

                            if closest_id is not None and min_distance < self.max_distance_last_detection:
                                print("[optitrack_tracker] Closest detection found with ID:", closest_id, "for marker ID:", marker.marker_id, "with distance:", min_distance)  
                                time_diff = (timestamp - self.optitrack_detections_dict[closest_id][0]).total_seconds()
                                x_velocity = (marker.position.x - self.optitrack_detections_dict[closest_id][1][0]) / time_diff
                                y_velocity = (marker.position.y - self.optitrack_detections_dict[closest_id][1][2]) / time_diff
                                z_velocity = (marker.position.z - self.optitrack_detections_dict[closest_id][1][4]) / time_diff
                                
                                matched_ids.add(closest_id)
                                
                            else:
                                # print("[optitrack_tracker] Closest detection not found or too far away")
                                print("[optitrack_tracker] setting velocity to 0.0 for marker ID:", marker.marker_id)
                                x_velocity = 0.0
                                y_velocity = 0.0
                                z_velocity = 0.0
                        
                        self.optitrack_detections_dict[marker.marker_id] = (timestamp,[marker.position.x, x_velocity, marker.position.y, y_velocity, marker.position.z, z_velocity])

        # tracking and predictions
        if self.ball_init_done and self.optitrack_detections_dict:
            
            # prediction
            try:
                if self.ball_interaction_states is not None:
                    
                    predict_list = [self.ball_interaction_states[track.id]['predict'] for track in self.multi_target_tracker.tracks]
                    # assert predict_list.count(True) <= 1, "More than one track has 'predict' set to True"
                    
                    td_predictions = TDPredictions()
                    td_predictions.header.stamp = rospy.Time.now()
                    
                    for track in self.multi_target_tracker.tracks:
                        if self.ball_interaction_states[track.id]['predict']: 

                            # ball below target height
                            if track.states[-1].state_vector[6] <= self.FIXED_Z_PLANE:
                                if self.actual_detection is None:
                                    self.actual_detection = {"x": track.states[-1].state_vector[0],
                                                                "y": track.states[-1].state_vector[3],
                                                                "z": track.states[-1].state_vector[6],
                                                                "timestamp": track.states[-1].timestamp}
                                    
                                    # print(f"[optitrack_tracker] Lstsq prediction: {self.lstsq_prediction}, Kalman prediction: {self.kalman_prediction}")
                                    # print(f"[optitrack_tracker] actual detection: {self.actual_detection}")

                                    lstsq_time_err = (self.actual_detection["timestamp"] - self.lstsq_prediction["timestamp"]).total_seconds() if self.lstsq_prediction else None
                                    kalman_time_err = (self.actual_detection["timestamp"] - self.kalman_prediction["timestamp"]).total_seconds() if self.kalman_prediction else None
                                    
                                    lstsq_x_err = self.actual_detection["x"] - self.lstsq_prediction["x"] 
                                    lstsq_y_err = self.actual_detection["y"] - self.lstsq_prediction["y"]
                                    lstsq_z_err = self.actual_detection["z"] - self.lstsq_prediction["z"] 

                                    kalman_x_err = self.actual_detection["x"] - self.kalman_prediction["x"]
                                    kalman_y_err = self.actual_detection["y"] - self.kalman_prediction["y"]
                                    kalman_z_err = self.actual_detection["z"] - self.kalman_prediction["z"]

                                    print(f"[optitrack_tracker] Lstsq time error: {lstsq_time_err}, Kalman time error: {kalman_time_err}")
                                    print(f"[optitrack_tracker] Lstsq x error: {lstsq_x_err}, Lstsq y error: {lstsq_y_err}")
                                    print(f"[optitrack_tracker] Kalman x error: {kalman_x_err}, Kalman y error: {kalman_y_err}")
                                    print(f"[optitrack_tracker] Lstsq z error: {lstsq_z_err}, Kalman z error: {kalman_z_err}")
                                    
                                    target_pos = [self.actual_detection['x'], self.actual_detection['y'], self.actual_detection['z']]
                                    kalman_prediction_pos = [self.kalman_prediction['x'],
                                                                self.kalman_prediction['y'],
                                                                self.kalman_prediction['z']]
                                    lstsq_prediction_pos = [self.lstsq_prediction['x'],
                                                                self.lstsq_prediction['y'],
                                                                self.lstsq_prediction['z']]
                                    
                                    print("[optitrack_tracker] Published target position:", target_pos, "Kalman prediction position:", kalman_prediction_pos, "LSTSQ prediction position:", lstsq_prediction_pos)

                                    self.publish_markers(target_pos, kalman_prediction_pos,lstsq_prediction_pos)

                                continue

                            # kamlan prediction
                            # # # if self.prediction_mode == "kalman":
                            try:
                                currentTime_timestamp = timestamp
                                targetTime_timestamp, kalman_catch_time_secs = calc_targetTime_timestamp_kalman(track.states[-1], self.FIXED_Z_PLANE) 
                                state_prediction = self.multi_target_tracker.predictor.predict(track.states[-1], transition_model=track.transition_model,timestamp=targetTime_timestamp, )
                                
                                sample_nb = 20
                                time_steps = np.linspace(0, (targetTime_timestamp - currentTime_timestamp).total_seconds(), sample_nb)
                                all_state_predictions = [
                                    self.multi_target_tracker.predictor.predict(track.states[-1],  transition_model=track.transition_model, timestamp=currentTime_timestamp + timedelta(seconds=t)).state_vector[[0, 3, 6]]
                                    for t in time_steps
                                ]
                    
                                self.publish_kalman_predictions(all_state_predictions)
                                
                                x = state_prediction.state_vector[0]
                                y = state_prediction.state_vector[3]
                                z = state_prediction.state_vector[6]     
                                dx = state_prediction.state_vector[1]
                                dy = state_prediction.state_vector[4]
                                dz = state_prediction.state_vector[7]               
                                kalman_prediction_pos = [x, y, z]
                                kalman_prediction_vel = [dx, dy, dz]
                                
                                self.kalman_prediction = {"x": kalman_prediction_pos[0],
                                                            "y": kalman_prediction_pos[1],
                                                            "z": kalman_prediction_pos[2],
                                                            "dx": kalman_prediction_vel[0],
                                                            "dy": kalman_prediction_vel[1],
                                                            "dz": kalman_prediction_vel[2],
                                                            "timestamp": targetTime_timestamp}

                            except Exception as e:
                                print("[optitrack_tracker] Skip publishing. Error during Kalman prediction:", e)
                                continue
                                
                            
                            # LSTSQ prediction
                            # # # elif self.prediction_mode == "lstsq":
                            try:                                    
                                
                                throw_start_sec = self.ball_interaction_states[track.id]["throw_start_timestamp"].to_sec()
                                throw_start_dt = datetime.fromtimestamp(throw_start_sec)
                                # detections = [s for s in track.states if s.timestamp >= throw_start_dt]
                                detections = [d for d in track.detections if d.timestamp >= throw_start_dt]
                                # detections = [s for s in track.states if s.state_vector[6] > self.height_threshold]
                                if len(detections) < 10:
                                    print("[optitrack_tracker] Not enough detections for LSTSQ prediction for track ID:", track.id)
                                    continue
                                
                                lstsq_prediction_pos, lstsq_prediction_vel, lstsq_catch_time_secs, coeffs = predict_with_lstsq(detections, self.approximate_velocity_measurement, z_plane=self.FIXED_Z_PLANE)
                                self.publish_detections(detections=detections, coeffs=coeffs, t_td=lstsq_catch_time_secs)
                                # lstsq_prediction_pos, lstsq_catch_time_secs, coeffs = predict_with_lstsq_const_acc(track.states)
                                lstsq_timestamp = timestamp + timedelta(seconds=lstsq_catch_time_secs)
                                self.lstsq_prediction = {"x": lstsq_prediction_pos[0],
                                                            "y": lstsq_prediction_pos[1],
                                                            "z": lstsq_prediction_pos[2],
                                                            "dx": lstsq_prediction_vel[0],
                                                            "dy": lstsq_prediction_vel[1],
                                                            "dz": lstsq_prediction_vel[2],
                                                            "timestamp": lstsq_timestamp }
                            except Exception as e:
                                print("[optitrack_tracker] Skip publishing. Error during LSTSQ prediction:", e)
                                continue
                            
                    

                            # publish prediction
                            td_prediction = TDPrediction()
                            td_prediction.ball_id = track.id
                            td_prediction.start_time = self.ball_interaction_states[track.id]['throw_start_timestamp'].to_sec() 
                            
                            td_prediction.header.stamp = rospy.Time.now()
                            td_prediction.header.frame_id = "world"
                            if self.prediction_mode == "kalman":
                                td_prediction.pose.position.x = kalman_prediction_pos[0]
                                td_prediction.pose.position.y = kalman_prediction_pos[1]
                                
                                td_prediction.velocity.x = kalman_prediction_vel[0]
                                td_prediction.velocity.y = kalman_prediction_vel[1]
                                td_prediction.velocity.z = kalman_prediction_vel[2]
                                
                                td_prediction.td_time = kalman_catch_time_secs
                            elif self.prediction_mode == "lstsq":
                                td_prediction.pose.position.x = lstsq_prediction_pos[0] 
                                td_prediction.pose.position.y = lstsq_prediction_pos[1]
                                
                                td_prediction.velocity.x = lstsq_prediction_vel[0]
                                td_prediction.velocity.y = lstsq_prediction_vel[1]
                                td_prediction.velocity.z = lstsq_prediction_vel[2]
                                
                                td_prediction.td_time = lstsq_catch_time_secs  
                            td_prediction.pose.position.z = self.FIXED_Z_PLANE  
                            td_prediction.pose.orientation.w = 1.0
                            td_prediction.pose.orientation.x = 0.0
                            td_prediction.pose.orientation.y = 0.0
                            td_prediction.pose.orientation.z = 0.0 
                            
                            td_predictions.predictions.append(td_prediction)
                            # print("[optitrack_tracker] Published TD prediction for track ID:", track.id, "with position:", td_prediction.pose.position, "and catch time:", td_prediction.td_time)
                
                            # publish target marker and prediction marker
                            target_pos = [0., 0,  0] # TODO get rid of target and only one prediction (kalman or lstsq)
                            # publish target prediction marker
                            self.publish_markers(target_pos, kalman_prediction_pos, lstsq_prediction_pos) # blue, yellow, green
                            
                            
                    self.TD_prediction_pub.publish(td_predictions)
                
                    
        
                        
                                
            except Exception as e:
                print("[optitrack_tracker] Error during prediction:", e)
               
            # print(len(self.optitrack_detections_dict))                
            # tracking
            # use only detections from current timestamp for kalman filter update                
            detections = [
                Detection(
                    StateVector(
                        [[state_value] for state_value  in self.optitrack_detections_dict[marker_id][1]]),
                        measurement_model=self.multi_target_tracker.measurement_model,
                        timestamp=timestamp,
                        marker_id = marker_id
                    )
                for marker_id in self.optitrack_detections_dict.keys() if self.optitrack_detections_dict[marker_id][0] == timestamp
            ]             
            
            # run kalman filter update
            self.multi_target_tracker.measurement_update_with_association(measurement=detections, timestamp=timestamp)
            
            # Publish ball states
            ball_states = BallStates()
            ball_states.header = data_frame.header
            for track in self.multi_target_tracker.tracks:
                
                # print("Track states len:", len(track.states), "detections len:", len(track.detections))
                
                ball_state = BallState()
                ball_state.id = track.id
                ball_state.position = Vector3(
                    x=track.states[-1].state_vector[0],
                    y=track.states[-1].state_vector[3],
                    z=track.states[-1].state_vector[6]
                )
                ball_state.velocity = Vector3(
                    x=track.states[-1].state_vector[1],
                    y=track.states[-1].state_vector[4],
                    z=track.states[-1].state_vector[7]
                )
                ball_state.acceleration = Vector3(
                    x=track.states[-1].state_vector[2],
                    y=track.states[-1].state_vector[5],
                    z=track.states[-1].state_vector[8]
                ) 
                ball_states.ball_states.append(ball_state)
           
            self.ball_publisher.publish(ball_states)
        
        if measure_callback_runtime:
            end = time.time()
            if (end - start)*1000 > 1.0:
                print("[optitrack_tracker] Tracking loop time: ", (end - start)*1000,"ms")
            # else:
            #     print("[optitrack_tracker] Tracking loop time: ", (end - start)*1000,"ms")
        

    def publish_kalman_predictions(self, all_state_predictions):
        """
        Publishes the predicted Kalman trajectory as a MarkerArray for visualization in RViz.
        Each predicted position is shown as a small sphere.
        """
        marker_array = MarkerArray()
        for i, pos in enumerate(all_state_predictions):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = pos[0]
            marker.pose.position.y = pos[1]
            marker.pose.position.z = pos[2]
            marker.scale.x = 0.04
            marker.scale.y = 0.04
            marker.scale.z = 0.04
            marker.color.r = 0.0
            marker.color.g = 0.7
            marker.color.b = 1.0
            marker.color.a = 0.8
            marker.lifetime = rospy.Duration(5.0)
            marker_array.markers.append(marker)
        self.kalman_prediction_pub.publish(marker_array)
        
        
    def publish_markers(self, target, prediction,sec_prediction=None):

        marker_duration = rospy.Duration(secs=15)  
        marker_id_target = 999999999
        marker_id_prediction = 999999998
        marker_id_sec_prediction = 999999997
        marker_array = MarkerArray()
        
        position_target = Vector3(x=target[0], y=target[1], z=target[2])    
        position_prediction = Vector3(x=prediction[0], y=prediction[1], z=prediction[2])
        # position_target = Vector3(x=0.543, y=-0.2685, z=0.57)

        # Create a Marker message
        target_marker = Marker()
        prediction_marker = Marker()

        # Set the marker type to SPHERE for visualization
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD

        # Set the ID (unique for each marker)
        target_marker.id = marker_id_target

        # Set the position of the marker
        target_marker.pose.position = position_target
        target_marker.pose.orientation.x = 0.0
        target_marker.pose.orientation.y = 0.0
        target_marker.pose.orientation.z = 0.0
        target_marker.pose.orientation.w = 1.0

        # Set the scale of the sphere (optional)
        target_marker.scale = Vector3(0.1, 0.1, 0.1)  # Example: 0.1m in each dimension

        # Set the color of the sphere (RGBA format)
        target_marker.color.r = 0.0  # Red
        target_marker.color.g = 0.0  # Green
        target_marker.color.b = 1.0  # Blue
        target_marker.color.a = 1.0  # Alpha (transparency)

        # Set the frame ID and timestamp for RViz
        target_marker.header.frame_id = "world"  # Replace "world" with the frame you're using
        target_marker.header.stamp = rospy.Time.now()
        target_marker.lifetime = marker_duration #rospy.Duration(0.1)  # 0.1s
        


        # Set the marker type to SPHERE for visualization
        prediction_marker.type = Marker.SPHERE
        prediction_marker.action = Marker.ADD

        # Set the ID (unique for each marker)
        prediction_marker.id = marker_id_prediction

        # Set the position of the marker
        prediction_marker.pose.position = position_prediction
        prediction_marker.pose.orientation.x = 0.0
        prediction_marker.pose.orientation.y = 0.0
        prediction_marker.pose.orientation.z = 0.0
        prediction_marker.pose.orientation.w = 1.0

        # Set the scale of the sphere (optional)
        prediction_marker.scale = Vector3(0.1, 0.1, 0.1)  # Example: 0.1m in each dimension

        # Set the color of the sphere (RGBA format)
        prediction_marker.color.r = 1.0  # Red
        prediction_marker.color.g = 1.0  # Green
        prediction_marker.color.b = 0.0  # Blue
        prediction_marker.color.a = 1.0  # Alpha (transparency)

        # Set the frame ID and timestamp for RViz
        prediction_marker.header.frame_id = "world"  # Replace "world" with the frame you're using
        prediction_marker.header.stamp = rospy.Time.now()
        prediction_marker.lifetime = marker_duration #rospy.Duration(0.1)  # 0.1s

        if sec_prediction is not None:
            sec_prediction_marker = Marker()
             # Set the marker type to SPHERE for visualization
            sec_prediction_marker.type = Marker.SPHERE
            sec_prediction_marker.action = Marker.ADD

            # Set the ID (unique for each marker)
            sec_prediction_marker.id = marker_id_sec_prediction

            # Set the position of the marker
            sec_position_prediction = Vector3(x=sec_prediction[0], y=sec_prediction[1], z=sec_prediction[2])
            sec_prediction_marker.pose.position = sec_position_prediction
            sec_prediction_marker.pose.orientation.x = 0.0
            sec_prediction_marker.pose.orientation.y = 0.0
            sec_prediction_marker.pose.orientation.z = 0.0
            sec_prediction_marker.pose.orientation.w = 1.0

            # Set the scale of the sphere (optional)
            sec_prediction_marker.scale = Vector3(0.1, 0.1, 0.1)  # Example: 0.1m in each dimension

            # Set the color of the sphere (RGBA format)
            sec_prediction_marker.color.r = 0.0  # Red
            sec_prediction_marker.color.g = 1.0  # Green
            sec_prediction_marker.color.b = 0.0  # Blue
            sec_prediction_marker.color.a = 1.0  # Alpha (transparency)

            # Set the frame ID and timestamp for RViz
            sec_prediction_marker.header.frame_id = "world"  # Replace "world" with the frame you're using
            sec_prediction_marker.header.stamp = rospy.Time.now()
            sec_prediction_marker.lifetime = marker_duration #rospy.Duration(0.1)  # 0.1s

            marker_array.markers.append(sec_prediction_marker)

        # marker_array.markers.append(prediction_marker)
        # marker_array.markers.append(target_marker)
 
        self.marker_pub.publish(marker_array)
       
    def publish_detections(self, detections,coeffs=None, t_td=None):
        """
        Publish the detections as markers in RViz for visualization.
        """
        
        if isinstance(detections[0], Detection):
            if self.approximate_velocity_measurement:
                points = np.array([d.state_vector[[0,2,4],0] for d in detections]).reshape((-1,3))
            else:
                points = np.array([d.state_vector[[0,1,2],0] for d in detections]).reshape((-1,3))
        elif isinstance(detections[0], State):
            points = np.array([d.state_vector[[0,3,6],0] for d in detections]).reshape((-1,3))
        else:
            raise ValueError("Detections must be of type Detection or State.")
        
        
        marker_array = MarkerArray()
        for i, point in enumerate(points):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = point[2]
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.5
            marker.color.a = 1.0
            marker.lifetime = rospy.Duration(0.3)
            
            marker_array.markers.append(marker)
        
        
        if coeffs is not None and t_td is not None:
            x_coef, y_coef, z_coef = coeffs
            sample_nb = 70
            
            ts = np.array([d.timestamp.timestamp() for d in detections])
            
            t_td = t_td + (ts[-1] - ts[0])
            
            
            t = np.linspace(0, t_td, sample_nb)
            xs = x_coef[1] + x_coef[0] * t 
            ys = y_coef[1] + y_coef[0] * t 
            zs = z_coef[2] + z_coef[1] * t + z_coef[0] * t**2
            
            for i,(x,y,z) in enumerate(zip(xs, ys, zs)):
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = rospy.Time.now()
                marker.id = i + 500 #len(detections) 
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = z
                marker.scale.x = 0.04
                marker.scale.y = 0.04
                marker.scale.z = 0.04
                marker.color.r = 0.3
                marker.color.g = 1.0
                marker.color.b = 0.3
                marker.color.a = 0.7
                marker.lifetime = rospy.Duration(0.3)
                marker_array.markers.append(marker)
        
        
        self.detection_marker_pub.publish(marker_array)

    def transform_point(self, point, stamp, src_frame, target_frame):
        stamped_point = PointStamped()
        stamped_point.header.stamp = stamp
        stamped_point.header.stamp = rospy.Time(0) # takes the latest available transform
        stamped_point.header.frame_id = src_frame
        stamped_point.point = point

        try:
            # Transform point
            transformed_point = self.tf_buffer.transform(stamped_point, target_frame, rospy.Duration(0.0001))
            return transformed_point.point
        except Exception as e:
            rospy.logwarn("TF transform failed: {}".format(e))
            return None

    def ball_interaction_states_callback(self, ball_interaction_states):
        """
        Callback function to handle ball interaction states from the topic. Saves the states in a dictionary.
        """
        self.ball_interaction_states = {
            ball_interaction_state.ball_id: {"interaction_state": InteractionState(ball_interaction_state.interaction_state),
                            "throw_start_timestamp": ball_interaction_state.throw_start_timestamp,
                            "predict": ball_interaction_state.predict
                            }
            for ball_interaction_state in ball_interaction_states.interaction_states
        }
   
        return 


if __name__ == '__main__':
    rospy.init_node('optitrack_ball_tracker', anonymous=True)
    optitrack_subscriber = OptitrackBallTracker()
    rospy.spin()
