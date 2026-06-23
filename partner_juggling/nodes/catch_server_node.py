#!/usr/bin/env python3

"""
    Chatch a ball with the robot arm
    Author: 
    Email:

"""
import time
import os
import numpy as np
import yaml
# pinocchio
import pinocchio as pin
from pinocchio import casadi as sym_pin

# ros
import rospy
import actionlib

# msgs
from control_msgs.msg import JointTrajectoryControllerState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from partner_juggling_msgs.msg import CatchAction, CatchGoal, CatchResult, CatchFeedback
from partner_juggling_msgs.msg import TDPrediction, TDPredictions
from partner_juggling_msgs.msg import BallInteractionStates

# trajectory planning
# from trajectory_planning.planner import TrajectoryPlanner
from trajectory_planning.trajectory import Trajectory
from trajectory_planning import kinematics as kin

# partner_juggling
from partner_juggling.ball_parabola import calc_throw_velocity
from partner_juggling.wait_for_controller import wait_for_controller
from partner_juggling.region import point_clipped_to_region, point_in_region 
from partner_juggling.interaction_state import InteractionState
from partner_juggling.trajectory_utils import build_trajectory_msg_from_traj, goto_joint, build_trajectory_msg, get_joint_states_from_traj_msg, plot_joint_trajectories, combine_trajectories
# from partner_juggling.trajectory_planners import plan_catch_trajectory, plan_catch_and_stop_trajectory, plan_catch_and_throw_and_stop_trajectory

# from partner_juggling.partner_planner import PartnerJugglingPlanner
from partner_juggling.partner_planner_throw_constraint import PartnerJugglingPlanner

from partner_juggling.online_test.online_test import check_trajectory

from partner_juggling.discretisation import compute_timesteps, quadspace, compute_timesteps_on_k,   compute_timesteps_start_catch_beginn_of_throw, compute_timesteps_begin_of_throw_to_stop,compute_timesteps_begin_of_throw_to_stop_from_zero,compute_timesteps_throw_and_stop  
import datetime


ACTIVE_DOFS = [0, 2, 3]
REST_POS = np.array([0.0, -np.pi/2, -np.pi/2, np.pi/2])
SAVE_TRAJ = False
tol = np.deg2rad(10)  
q1_min, q1_max = -2.6 + tol, 2.6 - tol
q2_min, q2_max = -2.0 + tol, 2.0 - tol
q3_min, q3_max = -2.8 + tol, 2.8 - tol
q4_min, q4_max = -0.9 + tol, 2.8 - tol


DEFAULT_SOLVER_OPTIONS = {
    'print_time': 0,
    'ipopt.print_level': 0,
    'ipopt.tol': 1e-5,
    'ipopt.max_iter': 100,
    # 'ipopt.linear_solver': 'ma27',
    'ipopt.acceptable_tol': 1e-6,
    'ipopt.acceptable_constr_viol_tol': 1e-6,
    'ipopt.acceptable_dual_inf_tol': 1e-6,
    'ipopt.acceptable_compl_inf_tol': 1e-6,
    'ipopt.warm_start_init_point': 'yes',
    'ipopt.mu_init': 1e-6,
    'ipopt.check_derivatives_for_naninf': 'no',  # Skip NaN/Inf checks
    'ipopt.derivative_test': 'none',  # Skip derivative testing
}



class MotionController:
    def __init__(self):
        self.exit = False
        # general variables
        self.freq_hz = 60  # Frequency of the motion control loop
        self.motion_enabled = True
        self.active_ball_id = -1
        self.wait_throw_to_finish = False
        self.first_time_after_switch = False 
        self.timestamp_throw_finished = rospy.Time.now()
        self.save_file_idx = 0
        self.min_time_to_catch = 0.15
        self.planning_buffer_time = 0.15
    
        self.goal = None
        self.ball_interaction_states = None
        self.latest_joint_state = None
        self.last_traj_msg = None
        self.last_traj_msgs = []
        
        self.warm_start = False  # TODO aoutdated
        self.reuse_solver = False # TODO aoutdated
        
        # Action server for catching a ball
        self.server = actionlib.SimpleActionServer('CatchBall', CatchAction, self.execute, False)

        # subscribe to the touch down prediction topic
        self.goal_sub = rospy.Subscriber('Ball_TD_position_prediction', TDPredictions, self.goal_callback_plan_repeatedly, queue_size=1) # queue_size=1 asserts always the latest message is processed

        # subscripte to interaction states
        rospy.Subscriber("ball_interaction_states", BallInteractionStates, self.ball_interaction_states_callback, queue_size=1)

        # subscribe to Controller state
        self.joint_state_sub = rospy.Subscriber('/wam_right/joint_trajectory_controller_LinearDerivatives_PDFFID/state', JointTrajectoryControllerState, self.joint_state_callback, queue_size=1)
        
        # get robot description and trajectory command topic name
        try:
            robot_description = rospy.get_param("/robot_description")
            self.robot_description = robot_description
            traj_cmd_topic_name = "/wam_right/joint_trajectory_controller_LinearDerivatives_PDFFID/command"
        except RuntimeError as e:
            print(f"Looks like not all required ros nodes are running:\n{e}")

        # publisher for trajectory commands
        self.traj_cmd_pub = rospy.Publisher(traj_cmd_topic_name, JointTrajectory, queue_size=1)
        rospy.sleep(0.5)  # wait for publisher to be registered
        wait_for_controller(traj_cmd_topic_name.split('/')[-2], timeout=4.0)

        # print robot name 
        robot_name = traj_cmd_topic_name.split('/')[-3]
        print(f"Robot name: {robot_name}")

        # print joint names
        self.joint_names = [f"{robot_name}_joint_{i+1}" for i in range(4)]
        print(f"Joint names: {self.joint_names}")
        self.moving_joint_names = [self.joint_names[dof] for dof in ACTIVE_DOFS]

        # create pinocchio model and data
        self.pin_model = pin.buildModelFromXML(robot_description)
        self.pin_data = self.pin_model.createData()
        self.sym_pin_model = sym_pin.Model(self.pin_model)
        self.sym_pin_data = sym_pin.Data(self.sym_pin_model)
        
        # collision checking
        self.tool_frame_name = robot_name + "_tool"
        self.ellbow_frame_name = robot_name + "_link_4"
        self.ellbow_frame_id = self.pin_model.getFrameId(self.ellbow_frame_name)
        self.tool_frame_id = self.pin_model.getFrameId(self.tool_frame_name )
        
        # get safety boxes
        self.safety_boxes = rospy.get_param("safety_boxes")

        
        # frame names and ids
        self.hand_frame_name = robot_name + "_tool"
        hand_frame_id = self.pin_model.getFrameId(self.hand_frame_name)
        self.hand_frame_id = hand_frame_id
        self.ref_frame = pin.LOCAL_WORLD_ALIGNED # from kinematitics.py
        print(f"Hand frame id: {hand_frame_id}")

    
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
        path = os.path.normpath(path)
        inv_kin_path = os.path.join(path, "inv_kin_results.npz")
        if os.path.exists(inv_kin_path):
            data = np.load(inv_kin_path)
            precalculated_inv_kin = data 
            print(f"Loaded {len(precalculated_inv_kin)} pre-calculated configurations from {inv_kin_path}")
        else:        
            raise FileNotFoundError(f"Pre-calculated configurations file not found at {inv_kin_path}")

        # create planner
        self.planner = PartnerJugglingPlanner(
            urdf_string=robot_description,
            hand_frame_name=self.hand_frame_name,
            joint_names=self.joint_names,
            moving_joint_names=self.moving_joint_names,
            solver_options=DEFAULT_SOLVER_OPTIONS,
            verbose=False,
            precalculated_inv_kin = precalculated_inv_kin
        )
        
        
        
        # epllipse parameters for region
        self.ellipse_center_x = 6.822320486321587e-14
        self.ellipse_center_y = -0.3865181854630024
        self.a = 0.8725810784022197
        self.b = 0.5070394009593238

        # circle parameters for region
        self.circle_r = 0.4
        self.circle_center_x = 0
        self.circle_center_y = 0
        # region bounds
        self.region_x_bounds = (0, 0.8) #(0, 0.85)
        self.region_y_bounds = (-0.8, 0.01)


        # move to rest position
        print("Moving to rest position...")
        goto_joint(REST_POS, self.traj_cmd_pub, self.joint_names, duration=1.0, blocking=True)
        print("Rest position reached.")
        
        

        ###  general planning settings ### 
        path_to_this_file = os.path.abspath(__file__)
        dir_config_file = os.path.join(os.path.dirname(path_to_this_file), "..",  "config")
        config_file_path = os.path.join(dir_config_file, "catch_server_setup_dyn.yaml")
        
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # config load start
        dddq_limit = config.get('DDDQ_LIMIT')
        q_stop = np.array(config.get('q_stop', REST_POS))
        
        self.upper= config.get('upper', None)
        self.upper = np.array(self.upper) if self.upper is not None else None
        self.lower= config.get('lower', None)
        self.lower = np.array(self.lower) if self.lower is not None else None
        
        torque_max = config.get('torque_max', None)
        torque_min = config.get('torque_min', None)
        torque_limit_factor = config.get('torque_limit_factor', 1.0)
        if torque_max is not None:
            torque_max = np.array(torque_max) * torque_limit_factor
        if torque_min is not None:
            torque_min = np.array(torque_min) * torque_limit_factor
        
        self.target_pos = config.get('target_pos',None)
        self.target_pos = np.array(self.target_pos) if self.target_pos is not None else None
        flight_time = config.get('flight_time')
        
        t_min = config.get('t_min', None)
        t_max = config.get('t_max', None)
        
        assert ((self.upper is None and self.lower is None and self.target_pos is not None) or
                (self.upper is not None and self.lower is not None and self.target_pos is None)), \
            "Either both 'upper' and 'lower' must be None and 'target_pos' set, or both 'upper' and 'lower' set and 'target_pos' must be None."

        equality_flag = True if (self.upper is None and self.lower is None) else False    

        self.nb_discretization_points_throw = config.get('nb_discretization_points_throw')
        self.nb_discretization_points_stop = config.get('nb_discretization_points_stop')
        self.nb_discretization_points_catch = config.get('nb_discretization_points_catch')

        self.k_throw =  self.nb_discretization_points_throw - 1
        rollout_steps = range(1, self.k_throw)
        self.post_take_off_idx_nb = config.get('post_take_off_idx')
        post_take_off_idx = [ i + self.k_throw for i in range(1, self.post_take_off_idx_nb)]

        self.duration_throw = config.get('duration_throw')
        self.duration_stop = config.get('duration_stop')
        self.min_duration_catch = config.get('min_duration_catch')
        
        dq_max = np.array(config.get('dq_max', [150.0, 150.0, 150.0, 150.0]))
        dq_min = np.array(config.get('dq_min', [-150.0, -150.0, -150.0, -150.0]))
        ddq_max = np.array(config.get('ddq_max', [1500.0, 1500.0, 1500.0, 1500.0]))
        ddq_min = np.array(config.get('ddq_min', [-1500.0, -1500.0, -1500.0, -1500.0]))
        
        ddq_factor = config.get('ddq_factor', 1.0) 
        ddq_min = ddq_min * ddq_factor
        ddq_max = ddq_max * ddq_factor 
                
        cone_angle_deg = config.get('cone_angle_deg', None)
        assert cone_angle_deg is not None, "cone_angle_deg must be specified in the config file"
        
        dddq_min = np.full(4, -dddq_limit)
        dddq_max = np.full(4, dddq_limit)
        
        CONSTRAINT_PARAMS = {
        'q_min': np.array([q1_min, q2_min, q3_min, q4_min]),
        'q_max': np.array([q1_max, q2_max, q3_max, q4_max]),
        'dq_min': dq_min,
        'dq_max': dq_max,
        'ddq_min': ddq_min,
        'ddq_max': ddq_max,
        'dddq_min': dddq_min,
        'dddq_max': dddq_max,
        }
        
        default_joint_limits_throw = {
                "q_min": CONSTRAINT_PARAMS["q_min"],
                "q_max": CONSTRAINT_PARAMS["q_max"],
                # "dq_min": CONSTRAINT_PARAMS["dq_min"],
                # "dq_max": CONSTRAINT_PARAMS["dq_max"],
                # "ddq_min": CONSTRAINT_PARAMS["ddq_min"],
                # "ddq_max": CONSTRAINT_PARAMS["ddq_max"],
                # "dddq_min": CONSTRAINT_PARAMS["dddq_min"],
                # "dddq_max": CONSTRAINT_PARAMS["dddq_max"],
            }

        default_joint_limits_after_throw = {
                "q_min": CONSTRAINT_PARAMS["q_min"],
                "q_max": CONSTRAINT_PARAMS["q_max"],
                "dq_min": CONSTRAINT_PARAMS["dq_min"],
                "dq_max": CONSTRAINT_PARAMS["dq_max"],
        }
            
        default_rollout_cone_slope = 0.4
        key = "my_throw"
        transcription_method = "multiple_shooting"
        # transcription_method = "single_shooting"
        self.store_warm_start = False
        self.warm_start = {}
        
        
        #### config load end #####
                
        # # warm start specifics 
        q_0 = REST_POS
        dq_0 = np.zeros(len(q_0))
        ddq_0 = np.zeros(len(q_0))
        x_catch = np.array([0.3,-0.4,0.57])
        
        time_to_td = self.min_duration_catch
           
        self.timesteps_throw, _ = compute_timesteps_throw_and_stop(
            self.duration_throw,
            self.duration_stop,
            nb_discretization_points=self.nb_discretization_points_throw + self.nb_discretization_points_stop,
            k_throw=self.k_throw,
        )
 
        timesteps_catch, _ = compute_timesteps_begin_of_throw_to_stop_from_zero(
            time_to_td,
            nb_discretization_points= self.nb_discretization_points_catch,
            k_throw=0,
        )
        
        # throw
        start = time.time()
        self.throw_and_stop_key = "throw_and_stop"
        self.planner.build_throw_and_stop_nlp(
            default_time_steps=self.timesteps_throw,
            default_joint_limits=default_joint_limits_throw,
            default_rollout_cone_slope=default_rollout_cone_slope,
            cone_angle_deg=cone_angle_deg,
            post_take_off_idxs=post_take_off_idx,
            rollout_steps=rollout_steps,
            k_throw=self.k_throw,
            t_min=t_min,
            t_max=t_max,
            transcription_method=transcription_method,
            key=self.throw_and_stop_key,
            flight_time=flight_time,
            p_target=self.target_pos,
            equality=equality_flag,
            upper=self.upper,
            lower=self.lower,
            torque_max=torque_max,
            torque_min=torque_min,
            ddq_max=ddq_max,
            ddq_min=ddq_min,
            dddq_max=dddq_max,
            dddq_min=dddq_min,
        )
        end = time.time()
        print(f"Throw  NLP build time: {end - start:.4f} seconds")

        start = time.time()
        traj_throw = self.planner.plan_throw_and_stop_trajectory(
            q_start=REST_POS,
            q_stop=REST_POS,
            target_pos=self.target_pos,
            lower=self.lower,
            upper=self.upper,
            time_steps=self.timesteps_throw,
            store_warm_start=self.store_warm_start,
            warm_start=self.warm_start,
            key=self.throw_and_stop_key
        )
        end = time.time()
        print(f"Throw  Solve time: {end - start:.4f} seconds")   
        
        # catch
        self.after_throw_to_catch_key = "after_throw_to_catch"
        start = time.time()
        self.planner.build_after_throw_to_catch_nlp(
            default_time_steps=timesteps_catch,
            default_joint_limits=default_joint_limits_after_throw,
            transcription_method=transcription_method,
            key=self.after_throw_to_catch_key,
        )
        end = time.time()
        print(f"Catch NLP build time: {end - start:.4f} seconds")
        start_time = time.time()
        traj_catch = self.planner.plan_after_throw_to_catch_trajectory(
            q_0=q_0,
            dq_0=dq_0,
            ddq_0=ddq_0,
            x_stop=x_catch,
            time_steps=timesteps_catch,
            key=self.after_throw_to_catch_key,
            store_warm_start=self.store_warm_start,
            warm_start=self.warm_start,
        )
        planning_time = time.time() - start_time
        print(f"Catch Solve time: {planning_time:.3f} s")
             
        # print(f"MotionController successfully initialized with the following parameters:")
        # print(f"  - Hand frame name: {self.hand_frame_name}")
        # print(f"  - Hand frame ID: {self.hand_frame_id}")
        # print(f"  - Ellipse center: ({self.ellipse_center_x}, {self.ellipse_center_y})")
        # print(f"  - Ellipse semi-major axis (a): {self.a}")
        # print(f"  - Ellipse semi-minor axis (b): {self.b}")
        # print(f"  - Circle radius: {self.circle_r}")
        # print(f"  - Circle center: ({self.circle_center_x}, {self.circle_center_y})")
        # print(f"  - Frequency: {self.freq_hz} Hz")
        # print(f"  - Total discretization points: {self.nb_discretization_points}")


        # random throw target
        self.targets = [
            np.array([1.0, 0.8, 0.1]),
            np.array([1.5, 0, 0.1]),
            np.array([1, -0.8, 0.1])
        ]
        
        margin = 0.05
        self.lowers = [t - margin for t in self.targets]  # elementwise lower bounds
        self.uppers = [t + margin for t in self.targets]  # elementwise upper bounds

        self.target_counter = 0

        self.server.start()
        rospy.loginfo("[%s] Ready to catch balls!", rospy.get_name())

    def joint_state_callback(self, joint_state_msg):
        """
        Callback function to handle joint state updates from the controller.  
        """
        if self.motion_enabled:
            # Update the latest joint state from the JointState message
            self.latest_joint_state = joint_state_msg
 
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
    
    def _parse_predictions(self, predictions):
        """Extract prediction data into a dict."""
        predictions_by_id = {}
        for pred in predictions:
            predictions_by_id[pred.ball_id] = {
                "position": np.array([pred.pose.position.x, pred.pose.position.y, pred.pose.position.z]),
                "velocity": np.array([pred.velocity.x, pred.velocity.y, pred.velocity.z]),
                "start_time": pred.start_time,
                "td_time": pred.td_time
            }
        return predictions_by_id

        
    def _select_active_ball(self, last_active_ball_id, prediction_time, predictions_by_id):
        """
        Select the most feasible (earliest-starting but still reachable) ball prediction.

        Args:
            last_active_ball_id (int or None): Previously active ball ID.
            predictions_by_id (dict): Output of _parse_predictions().

        Returns:
            tuple: (active_ball_id, switched_flag)
                - active_ball_id: the selected ball's ID, or None if none feasible.
                - switched_flag: True if the selected ID differs from last_active_ball_id.
        """

        if not predictions_by_id:
            rospy.logwarn("[%s] No predictions available.", rospy.get_name())
            return None, False

        # --- Current time ---
        # now = rospy.Time.now()
        # # Minimum time window to be feasible
        # min_feasible_time = now + rospy.Duration(self.planning_buffer_time) + rospy.Duration(self.min_time_to_catch)
        min_feasible_time = prediction_time + rospy.Duration(self.planning_buffer_time) + rospy.Duration(self.min_time_to_catch)

        # --- Find all feasible predictions ---
        feasible_preds = []
        for pred_id, pred_data in predictions_by_id.items():
            # Compute touchdown absolute time
            td_abs_time = prediction_time + rospy.Duration(pred_data["td_time"])

            # Only keep feasible ones (catchable in time)
            if td_abs_time > min_feasible_time:
                feasible_preds.append((pred_id, pred_data))

        if not feasible_preds:
            rospy.loginfo("[%s] No feasible ball predictions (too close in time).", rospy.get_name())
            return None, False

        # --- Select the earliest-starting feasible prediction ---
        feasible_preds.sort(key=lambda x: rospy.Time(x[1]["start_time"])) # ascending -> lowest start time
        selected_id = feasible_preds[0][0]

        # --- Check if we switched targets ---
        switched = (selected_id != last_active_ball_id)

        if last_active_ball_id == -1:
            switched = False

        if switched:
            rospy.loginfo("[%s] Switching active ball from %s to %s", rospy.get_name(), last_active_ball_id, selected_id)

        return selected_id, switched


    def goal_callback_plan_repeatedly(self, TDPredictions_msg):
        """Uses the received touch down prediction to calculate the catch Trajectorie and saves it to the object variable self.latest_JointGoal, so that it can be published to the command topic of the controller.

        This Version uses the planner repeatedly for planning.

        Side Effects: sets self.latest_JointGoal with newest Trajetory

        Args:
            TDPrediction_msg (TDPrediction): contains TD prediction pose and TD time in secs (float)
        """
       
        if self.exit:
            return        

        if self.motion_enabled and self.latest_joint_state:           
            if not hasattr(TDPredictions_msg, 'predictions') or len(TDPredictions_msg.predictions) < 1:
                # rospy.logwarn("[%s] Received empty TDPredictions message, skipping.", rospy.get_name())
                return
            else:
                predictions_by_id = self._parse_predictions(TDPredictions_msg.predictions)

                
            # selcect active ball id
            header_time = TDPredictions_msg.header.stamp
            active_ball_id, switched = self._select_active_ball(self.active_ball_id, header_time, predictions_by_id)
            # no feasible prediction
            if active_ball_id is None:
                return
            else:
                self.active_ball_id = active_ball_id

            assert not(self.wait_throw_to_finish and switched), "wait_throw_to_finish and switch are both true -> not allowed"

            if switched:
                self.wait_throw_to_finish = True
                # rospy.logerr("switch")
                # self.exit = True
                # exit()
                # print("self.timestamp_throw_finished", self.timestamp_throw_finished)

            # check if switch still active
            if self.wait_throw_to_finish:
                earliest_start_time_new_traj = rospy.Time.now() + rospy.Duration(self.planning_buffer_time)
                # print("self.timestamp_throw_finished", self.timestamp_throw_finished)
                if earliest_start_time_new_traj > self.timestamp_throw_finished: 
                    self.wait_throw_to_finish = False
                    self.first_time_after_switch = True
                  
                else:
                    pass



            # get the touchdown prediction and clip it to the feasable region
            x_catch = predictions_by_id[self.active_ball_id]["position"] 
            dx_ball = predictions_by_id[self.active_ball_id]["velocity"]
            real_time_to_td = predictions_by_id[self.active_ball_id]["td_time"]
            
            
            ok = point_in_region(x_catch, self.ellipse_center_x, self.ellipse_center_y, self.a, self.b, self.circle_r, self.circle_center_x, self.circle_center_y, self.region_x_bounds, self.region_y_bounds)
            if not ok:
                rospy.loginfo("[%s] Predicted catch point is outside of the feasible catch region, skipping planning.", rospy.get_name())
                return



            # case 1: go directly to catch
            if not self.wait_throw_to_finish:
                start_time_new_traj = header_time + rospy.Duration(self.planning_buffer_time)
                planning_time_to_td = real_time_to_td - self.planning_buffer_time 

                if self.last_traj_msg is None:
                    # get current desired joint state from the controller
                    q_0 = np.array(self.latest_joint_state.desired.positions, dtype=np.float64)
                    dq_0 = np.array(self.latest_joint_state.desired.velocities, dtype=np.float64)
                    ddq_0 = np.array(self.latest_joint_state.desired.accelerations, dtype=np.float64) if not None in self.latest_joint_state.desired.accelerations else np.zeros(len(q_0))
                    
                else:
                    q_0, dq_0, ddq_0 = get_joint_states_from_traj_msg(self.last_traj_msg,time=start_time_new_traj)  
                    q_0 = np.array(q_0)
                    dq_0 = np.array(dq_0)
                    ddq_0 = np.array(ddq_0)


            # case 2: wait for throw to execture
            else:
                return
          
            
                        
            if planning_time_to_td < self.min_time_to_catch:
                rospy.loginfo("[%s] Touch down time is too close to current time, skipping planning.", rospy.get_name())
                return
            
            if planning_time_to_td > 1.0:
                rospy.loginfo("[%s] Touch down time is too far in the future, skipping planning.", rospy.get_name())
                return

        
            # start planning
            start_time = time.time()
            # wait_time_earlier = 0.05
            wait_time_earlier = 0.0
            # waiting_time_after = 0.05
            waiting_time_after = 0.0
            timesteps_catch, _ = compute_timesteps_begin_of_throw_to_stop_from_zero(
                planning_time_to_td- wait_time_earlier,
                nb_discretization_points= self.nb_discretization_points_catch,
                k_throw=0,
            )
            
            # catch traj
            traj_catch = self.planner.plan_after_throw_to_catch_trajectory(
                q_0=q_0,
                dq_0=dq_0,
                ddq_0=ddq_0,
                x_stop=x_catch,
                time_steps=timesteps_catch,
                key=self.after_throw_to_catch_key,
                store_warm_start=self.store_warm_start,
                warm_start=self.warm_start,
            )
            stats_catch = self.planner.after_throw_to_catch_configs[self.after_throw_to_catch_key].nlp_problem.stats
            # throw traj
            
            q_start_throw = traj_catch.q[-1]
      
            ## use different targets
            # traj_throw = self.planner.plan_throw_and_stop_trajectory(
            #     q_start=q_start_throw,
            #     q_stop=REST_POS,
            #     target_pos=self.targets[self.target_counter % len(self.targets)],
            #     lower=self.lowers[self.target_counter % len(self.targets)],
            #     upper=self.uppers[self.target_counter % len(self.targets)] ,
            #     time_steps=self.timesteps_throw,
            #     store_warm_start=self.store_warm_start,
            #     warm_start=self.warm_start,
            #     key=self.throw_and_stop_key
            # )

            self.target_counter +=1
            self.target_counter = self.target_counter % len(self.targets)
            ### end use different targets 
            
            ## old solution without different targests 
            # timesteps throw start from zero
            traj_throw = self.planner.plan_throw_and_stop_trajectory(
                q_start=q_start_throw,
                q_stop=REST_POS,
                target_pos=self.target_pos,
                lower=self.lower,
                upper=self.upper,
                time_steps=self.timesteps_throw,
                store_warm_start=self.store_warm_start,
                warm_start=self.warm_start,
                key=self.throw_and_stop_key
            )
            
            stats_throw = self.planner.throw_and_stop_configs[self.throw_and_stop_key].nlp_problem.stats
            # combine and adjust time
            traj_throw.time_steps= traj_throw.time_steps.copy()
            traj_throw.time_steps += (planning_time_to_td+waiting_time_after)
            traj_combined = combine_trajectories(traj_catch, traj_throw, self.joint_names)
            

            if stats_throw['success'] == False or stats_catch['success'] == False:
                rospy.logerr("[%s] Planning failed", rospy.get_name())
                if stats_throw['success'] == False:
                    rospy.logerr("[%s] Throw Planning failed", rospy.get_name())
                if stats_catch['success'] == False:
                    rospy.logerr("[%s] Catch Planning failed", rospy.get_name())
                # self.save_trajectory(
                #     traj_combined=traj_combined,
                #     x_catch=x_catch,
                #     dx_ball=dx_ball,
                #     q_0=q_0,
                #     dq_0=dq_0,
                #     ddq_0=ddq_0,
                #     k_catch=self.nb_discretization_points_catch-1,
                #     k_throw=self.k_throw + self.nb_discretization_points_catch-1,
                #     rollout_steps=range(self.nb_discretization_points_catch,self.nb_discretization_points_catch-1 + self.nb_discretization_points_throw-1),
                #     post_take_off_idx=[i + (self.k_throw + self.nb_discretization_points_catch-1) for i in range(1,self.post_take_off_idx_nb+1)],
                #     planning_time_to_td=planning_time_to_td,
                #     joint_names=traj_combined.joint_names,
                #     active_ball_id = self.active_ball_id,
                #     switch_pred = self.switch_pred,
                #     start_time_new_traj = start_time_new_traj,
                #     t_catch_again = self.t_catch_again,
                #     planning_time=planning_time if 'planning_time' in locals() else None,
                #     success=stats_catch['success'] and stats_throw['success'],
                #     safe=None,
                # )
                return
    
  
            # check trajectory
            start_check = time.time()
            traj_save = check_trajectory(self.safety_boxes,
                                        traj_combined,
                                        self.pin_model,
                                        self.pin_data,
                                        ellbow_frame_id=self.ellbow_frame_id,
                                        tool_frame_id=self.tool_frame_id,
                                        joint_names=self.joint_names,
                                        verbose=True)
            end_check = time.time()
            print(f"check_trajectory took {end_check - start_check:.4f} seconds")
            
            if not traj_save:
                rospy.logwarn("[%s] Trajectory is not safe to apply, aborting planning.", rospy.get_name())
                # self.save_trajectory(
                #     traj_combined=traj_combined,
                #     x_catch=x_catch,
                #     dx_ball=dx_ball,
                #     q_0=q_0,
                #     dq_0=dq_0,
                #     ddq_0=ddq_0,
                #     k_catch=self.nb_discretization_points_catch-1,
                #     k_throw=self.k_throw + self.nb_discretization_points_catch-1,
                #     rollout_steps=range(self.nb_discretization_points_catch,self.nb_discretization_points_catch-1 + self.nb_discretization_points_throw-1),
                #     post_take_off_idx=[i + (self.k_throw + self.nb_discretization_points_catch-1) for i in range(1,self.post_take_off_idx_nb+1)],
                #     planning_time_to_td=planning_time_to_td,
                #     joint_names=traj_combined.joint_names,
                #     active_ball_id = self.active_ball_id,
                #     switch_pred = self.switch_pred,
                #     start_time_new_traj = start_time_new_traj,
                #     t_catch_again = self.t_catch_again,
                #     planning_time=planning_time if 'planning_time' in locals() else None,
                #     success=stats_catch['success'] and stats_throw['success'],
                #     safe=None,
                # )
                return
            
            # check planning time
            end_time = time.time()
            planning_time = end_time - start_time
            print(f"planning took {planning_time:.4f} seconds")
                        
            
            if self.planning_buffer_time - planning_time < 0:
                rospy.logerr("[%s] planning took too long", rospy.get_name())
                # self.save_trajectory(
                #     traj_combined=traj_combined,
                #     x_catch=x_catch,
                #     dx_ball=dx_ball,
                #     q_0=q_0,
                #     dq_0=dq_0,
                #     ddq_0=ddq_0,
                #     k_catch=self.nb_discretization_points_catch-1,
                #     k_throw=self.k_throw + self.nb_discretization_points_catch-1,
                #     rollout_steps=range(self.nb_discretization_points_catch,self.nb_discretization_points_catch-1 + self.nb_discretization_points_throw-1),
                #     post_take_off_idx=[i + (self.k_throw + self.nb_discretization_points_catch-1) for i in range(1,self.post_take_off_idx_nb+1)],
                #     planning_time_to_td=planning_time_to_td,
                #     joint_names=traj_combined.joint_names,
                #     active_ball_id = self.active_ball_id,
                #     switch_pred = self.switch_pred,
                #     start_time_new_traj = start_time_new_traj,
                #     t_catch_again = self.t_catch_again,
                #     planning_time=planning_time if 'planning_time' in locals() else None,
                #     success=stats_catch['success'] and stats_throw['success'],
                #     safe=None,
                # )
                return
                                         
                        
            # update the trajectory command
            traj_msg = build_trajectory_msg_from_traj(traj=traj_combined,t0=start_time_new_traj, joint_names=self.joint_names)
            # rospy.loginfo("traj send")
            self.traj_cmd_pub.publish(traj_msg)

            
            if not self.wait_throw_to_finish:
                self.last_traj_msg = traj_msg
                k_throw_finished = self.nb_discretization_points_catch+self.nb_discretization_points_throw+self.post_take_off_idx_nb-1
                k_throw_finished += 0 
                self.timestamp_throw_finished =  rospy.Duration(traj_combined.time_steps[k_throw_finished]) + start_time_new_traj

            # self.last_traj_msgs.append(traj_msg)
            print(f"Stored {len(self.last_traj_msgs)} trajectories")
            
            if SAVE_TRAJ:
                self.save_trajectory(
                        traj_combined=traj_combined,
                        x_catch=x_catch,
                        dx_ball=dx_ball,
                        q_0=q_0,
                        dq_0=dq_0,
                        ddq_0=ddq_0,
                        k_catch=self.nb_discretization_points_catch-1,
                        k_throw=self.k_throw + self.nb_discretization_points_catch-1,
                        rollout_steps=range(self.nb_discretization_points_catch,self.nb_discretization_points_catch-1 + self.nb_discretization_points_throw-1),
                        post_take_off_idx=[i + (self.k_throw + self.nb_discretization_points_catch-1) for i in range(1,self.post_take_off_idx_nb+1)],
                        planning_time_to_td=planning_time_to_td,
                        joint_names=traj_combined.joint_names,
                        active_ball_id = self.active_ball_id,
                        switch_pred = self.wait_throw_to_finish,
                        start_time_new_traj = start_time_new_traj,
                        t_catch_again = self.timestamp_throw_finished,
                        planning_time=planning_time if 'planning_time' in locals() else None,
                        success=stats_catch['success'] and stats_throw['success'],
                        safe=None,
                        predictions_by_id = predictions_by_id,
                        traj_msg = traj_msg,
                    )

            return 



    def save_trajectory(
        self,
        traj_combined,
        x_catch,
        dx_ball,
        q_0,
        dq_0,
        ddq_0,
        k_catch,
        k_throw,
        rollout_steps,
        post_take_off_idx,
        planning_time_to_td,
        joint_names,
        active_ball_id,
        switch_pred,
        start_time_new_traj,
        t_catch_again,
        planning_time=None,
        success=None,
        safe=None,
        predictions_by_id=None,
        traj_msg =None,
    ):
        save_start = time.time()
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "planned_trajectories")
        os.makedirs(save_dir, exist_ok=True)
        base_filename = "traj_combined"
        # idx = 0
        # while True:
            # filename = os.path.join(save_dir, f"{base_filename}_{idx}.npz")
        #     if not os.path.exists(filename):
        #         break
        #     idx += 1
        filename = os.path.join(save_dir, f"{base_filename}_{self.save_file_idx}.npz")
        self.save_file_idx += 1
        np.savez(
            filename,
            time_steps=traj_combined.time_steps,
            q=traj_combined.q,
            dq=traj_combined.dq,
            ddq=traj_combined.ddq,
            dddq=traj_combined.dddq,
            x_catch=x_catch,
            dx_ball=dx_ball,
            q_0=q_0,
            dq_0=dq_0,
            ddq_0=ddq_0,
            k_catch=k_catch,
            k_throw=k_throw,
            rollout_steps=rollout_steps,
            post_take_off_idx=post_take_off_idx,
            planning_time_to_td=planning_time_to_td,
            joint_names=joint_names,
            active_ball_id = active_ball_id,
            switch_pred = switch_pred,
            start_time_new_traj = start_time_new_traj,
            t_catch_again = t_catch_again,
            planning_time=planning_time,
            success=success,
            safe=safe,
            predictions_by_id = predictions_by_id,
            traj_msg = traj_msg,
        )
        save_end = time.time()
        print(f"Saved trajectory to {filename}. Saving took {save_end - save_start:.4f} seconds")
        



    def goal_callback_IK(self, TDPrediction_msg):
        """Uses the received touch down prediction to calculate the catch Trajectorie and saves it to the object variable self.latest_JointGoal, so that it can be published to the command topic of the controller.

        This Version uses only the IK for planning. It sets two points: The starting point and the end point which should be reached at touch down time.

        Side Effects: sets self.latest_JointGoal with newest Trajetory

        Args:
            TDPrediction_msg (TDPrediction): contains TD prediction pose and TD time in secs (float)
        """
        if self.motion_enabled and self.latest_joint_state:

            # get current desired joint state from the controller
            q_0 = np.array(self.latest_joint_state.desired.positions, dtype=np.float64)
            # dq_0 = np.array(self.latest_joint_state.desired.velocities, dtype=np.float64)
            # ddq_0 = np.array(self.latest_joint_state.desired.accelerations, dtype=np.float64) if not None in self.latest_joint_state.desired.accelerations else np.zeros(len(q_0))

            # get the touchdown prediction and clip it to the feasable region
            pred_pos = TDPrediction_msg.pose.position
            real_time_to_td = TDPrediction_msg.td_time
            x_catch = np.array([pred_pos.x, pred_pos.y, pred_pos.z])
            x_catch = point_clipped_to_region(x_catch, self.ellipse_center_x, self.ellipse_center_y, self.a, self.b, self.circle_r, self.circle_center_x, self.circle_center_y) #TODO add x/y bounds


            q_catch = kin.ikin(m=self.planner.pin_model, d=self.planner.pin_data,
                    x_des=x_catch, frame_id=self.hand_frame_id,
                    moving_q_ids=self.planner.moving_q_ids,
                    moving_v_ids=self.planner.moving_v_ids,
                    const_q_ids=self.planner.const_q_ids,
                    return_q_ids=self.planner.q_ids,
                    q_guess_moving=q_0[self.planner.moving_q_ids_in_traj],
                    q_const=q_0[self.planner.const_q_ids_in_traj])
            # print(f"q_catch: {q_catch}")
            q = np.array([q_0, q_catch])
        
            time_stemps_since_t0 = np.array([0,real_time_to_td])
            
            q_catch_traj = Trajectory(time_stemps_since_t0, q, dq=None, ddq=None, dddq=None, joint_names=self.joint_names)
            traj_msg = build_trajectory_msg_from_traj(q_catch_traj, rospy.Time.now(), self.joint_names)
            
            # check trajectory
            start_check = time.time()
            traj_save = check_trajectory(self.safety_boxes, q_catch_traj, self.pin_model,
                             self.pin_data,
                             ellbow_frame_id=self.ellbow_frame_id,
                             tool_frame_id=self.tool_frame_id,
                             joint_names=self.joint_names,
                             verbose=True)
            end_check = time.time()
            print(f"check_trajectory took {end_check - start_check:.4f} seconds")
            
            if not traj_save:
                rospy.logwarn("[%s] Trajectory is not safe to apply, aborting planning.", rospy.get_name())
                return
            
            self.traj_cmd_pub.publish(traj_msg)
            



    def execute(self, goal):
        """  
        This function is called when the action server receives a goal.
        It starts the planning and motion by setting the motion_enabled flag to True.
        The actual planning and motion is controlled by the method goal_callback_plan_repeatedly which recieves the goal via a rostopic.
        This medhod checks if the ball is dropped or caught, is this the case it stops the motion and sets the action as succeeded or aborted accordingly.
        Args:
            goal (CatchGoal): The goal for the action. Does not use the goal directly, but checks the interaction state of the ball.
        """

        rospy.loginfo("[%s] Action goal received, starting motion...", rospy.get_name())
        self.goal = goal
        self.motion_enabled = True

        rate = rospy.Rate(self.freq_hz)
        feedback = CatchFeedback()
        
        predict_list = [self.ball_interaction_states[track_id]['predict'] for track_id in self.ball_interaction_states.keys()]
        # assert predict_list.count(True) <= 1, "More than one track has 'predict' set to True"
        assert predict_list.count(True) != 0, "No track has 'predict' set to True"
        
        track_id = np.where(np.array(predict_list) == True)[0][0]
        
        self.last_traj_msgs = []
        # self.last_traj_msg = None
        print("Starting new action, cleared old trajectories.")
        

        while not rospy.is_shutdown():
            current_interaction_state = InteractionState(self.ball_interaction_states[track_id]['interaction_state'])
                        
            if self.server.is_preempt_requested():
                rospy.loginfo("[%s] Action preempted.", rospy.get_name())
                self.motion_enabled = False
                self.server.set_preempted()
                return

            if current_interaction_state == InteractionState.FELL_TO_GROUND:
                rospy.loginfo("[%s] Ball dropped detected, stopping motion and setting action as aborted.", rospy.get_name())
                self.motion_enabled = False
                result = CatchResult(success=False)
                self.server.set_aborted(result)
                return
            
            if current_interaction_state == InteractionState.IN_ROBOT_INTERACTION:
                rospy.loginfo("[%s] Ball in robot hand detected, stopping motion and setting action as succeeded.", rospy.get_name())
                self.motion_enabled = False
                result = CatchResult(success=True)
                self.server.set_succeeded(result)
                return

            if current_interaction_state != InteractionState.THROWN_TO_ROBOT:
                rospy.logerr("[%s] %s is wrong interaction state for catching.", rospy.get_name(), current_interaction_state.name)
                self.motion_enabled = False
                result = CatchResult(success=False)
                self.server.set_aborted(result)
                return
    
            feedback.status = "Moving toward dynamic goal..."
            self.server.publish_feedback(feedback)

            rate.sleep()
            

if __name__ == '__main__':
    rospy.init_node('motion_controller')
    MotionController()
    rospy.spin()
