from juggle_planning.planner import JugglingTrajectoryPlanner, NlpConfiguration
from trajectory_planning import kinematics as kin
from juggle_planning.constraints.contact_maintanance_constraints import RollOutPreventionConstraint
from juggle_planning.constraints.post_takeoff_collinearity_constraints import PostTakeOffCollinearityConstraintAcc
from juggle_planning.constraints.pre_touchdown_collinearity_constraints import PreTouchDownCollinearityConstraintVel
from trajectory_planning.constraints.joint_space_constraints import JointLimitConstraint, JointStateEqualityConstraint
from trajectory_planning.constraints.cart_space_constraints import CartStateEqualityConstraint
from trajectory_planning.cost_functions import average_of_squared_joint_accelerations_at_breakpoints,average_of_squared_joint_jerks_at_breakpoints

from partner_juggling.cost_functions import throw_cost_fn


# from partner_juggling.constraints.cone_intersection_constraint import ConeIntersectionConstraint
from partner_juggling.constraints.cone_intersection_constraint_paper import ConeIntersectionConstraint
from partner_juggling.constraints.throw_target_constraints import ThrowToTargetConstraint
from partner_juggling.constraints.torque_constraint import TorqueConstraint 
from partner_juggling.constraints.acceleration_constraint import AccelerationConstraint
from partner_juggling.constraints.jerk_constraint import JerkConstraint


from scipy.spatial import KDTree
import numpy as np
import time
from typing import Optional, Sequence, Iterable

REST_POS = np.array([0.0, -np.pi/2, -np.pi/2, np.pi/2])


class PartnerJugglingPlanner(JugglingTrajectoryPlanner):
    """
    Extension of JugglingTrajectoryPlanner for partner juggling scenarios.
    Provides specialized planning methods and constraint preparation.
    """
    def __init__(self, warm_start=True, use_collinarity=True, use_multiple_equality_constraints=False,throw_on_k=None, use_multiple_position_constraints=None, use_cone_intersection_constraint=None, use_position_velocity_for_catch=None,precalculated_inv_kin=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add any partner-specific attributes here

        self.ikin_step_size = 0.05
        self.ikin_max_iter = 1000
        
        self.precalculated_inv_kin = precalculated_inv_kin
        if self.precalculated_inv_kin is not None:
            self.inv_kin_tree = KDTree(self.precalculated_inv_kin["x_des"])
        
        
        self.use_warm_start = warm_start

        self.catch_configs: dict[str, NlpConfiguration] = {}
        self.catch_and_stop_configs: dict[str, NlpConfiguration] = {}
        self.throw_and_stop_configs: dict[str, NlpConfiguration] = {}
        self.catch_and_throw_and_stop_configs: dict[str, NlpConfiguration] = {}
        self.after_throw_and_stop_configs: dict[str, NlpConfiguration] = {}
        self.catch_and_throw_configs: dict[str, NlpConfiguration] = {}
        self.after_throw_to_catch_configs: dict[str, NlpConfiguration] = {}

        # self.q_throw = None
        # self.dq_throw = None
        # self.ddq_throw = None


        # self.cost_function = average_of_squared_joint_accelerations_at_breakpoints
        # self.cost_function =  average_of_squared_joint_jerks_at_breakpoints
        self.cost_function = throw_cost_fn
    
    def precalculate_throw_kinematics(self, x_throw: np.ndarray, dx_throw: np.ndarray, q_start: np.ndarray):
        q_throw, dq_throw, ddq_throw, warning = kin.dikin(
            m=self.nlp_builder.pin_model,
            d=self.nlp_builder.pin_data,
            x_des=x_throw,
            dx_des=dx_throw,
            ddx_des=self.nlp_builder.pin_model.gravity.linear,
            frame_id=self.hand_frame_id,
            moving_q_ids=self.nlp_builder._moving_q_ids_in_model,
            moving_v_ids=self.nlp_builder._moving_v_ids_in_model,
            const_q_ids=self.nlp_builder._const_q_ids_in_model,
            return_q_ids=self.nlp_builder._q_ids_in_model,
            return_v_ids=self.nlp_builder._v_ids_in_model,
            q_guess_moving=REST_POS[self.nlp_builder._moving_q_ids_in_traj],
            q_const=q_start[self.nlp_builder._const_q_ids_in_traj],
            step_size=self.ikin_step_size,
            # max_iter=self.ikin_max_iter
        )
        self.q_throw = q_throw
        self.dq_throw = dq_throw
        self.ddq_throw = ddq_throw

    def build_throw_and_stop_nlp(self, default_time_steps: Iterable[int],
                        default_joint_limits: dict,  # TODO: save joint limits and make sure that at self.plan_* get's the same structure of joint limits / no limits that were not declared at build time
                        k_throw = None,
                        k_catch = None,
                        q_catch = None,
                        t_min = None,
                        t_max = None,
                        cone_angle_deg=None,
                        upper = np.array([2.2, 0.1, 0.01]),
                        lower = np.array([1.8, -0.1, -0.01]),
                        flight_time = 1.5,
                        p_target = np.array([2, -0.35, 0]),
                        equality = False,
                        post_take_off_idxs = None,
                        torque_max =np.array([180, 100, 70, 45]),
                        torque_min =np.array([-180, -100, -70, -45]),
                        ddq_max=np.array([50, 50, 50, 80]),
                        ddq_min=np.array([-50, -50, -50, -80]),   
                        dddq_max=None,
                        dddq_min=None, 
                        rollout_steps: Optional[Iterable[int]] = None,
                        default_rollout_cone_slope: Optional[float] = 0.4,
                        default_rollout_gravity: Optional[np.ndarray] = np.array([0, 0, -9.81]),
                        default_q_throw: Optional[np.ndarray] = None,
                        default_dq_throw: Optional[np.ndarray] = None,
                        default_ddq_throw: Optional[np.ndarray] = None,
                        default_q_stop: Optional[np.ndarray] = None,
                        default_dq_stop: Optional[np.ndarray] = None,
                        default_ddq_stop: Optional[np.ndarray] = None,
                        transcription_method: Optional[str] = 'multiple_shooting',
                        cost_scale: Optional[float] = 0.01,
                        key: Optional[str] = "default"):
        """
        Build NLP for throwing trajectories, staring from a resting joint position, and ending at the time of takeoff, when contact with the ball is broken.
        All floating point arguments can be reparameterized in value, but not in dimension (e.g. number of time steps, or types of joint limits) at solve time.
        
        Args:
            default_time_steps: ....... Needs to have same dimension at solve time!
            default_joint_limits: ..... Dictionary with default joint limit parameters (needs to have same structure as at solve time)
            rollout_steps: ............ Steps for rollout prevention constraint: Default every second step before takeoff
            default_rollout_cone_slope: Optional default rollout cone slope:  Default 0.4
            default_rollout_gravity: .. Optional default rollout gravity: ... Default [0, 0, -9.81]
            default_q_throw: .......... Optional default throw position:  ... Default [0, ..., 0]
            default_dq_throw: ......... Optional default throw velocity: .... Default [0, ..., 0]
            default_ddq_throw: ........ Optional default throw acceleration:  Default [0, ..., 0]
            transcription_method: ..... Optional transcription method: ...... Default 'multiple_shooting'
            key: ...................... Optional key for this NLP configuration: Default "default"
        """
        default_rollout_gravity = default_rollout_gravity if default_rollout_gravity is not None else np.array([0, 0, -9.81])
        default_q_throw = default_q_throw if default_q_throw is not None else np.zeros(self.nlp_builder.nq_moving) 
        default_dq_throw = default_dq_throw if default_dq_throw is not None else np.zeros(self.nlp_builder.nv_moving)
        default_ddq_throw = default_ddq_throw if default_ddq_throw is not None else np.zeros(self.nlp_builder.nv_moving)
        
        default_q_stop = default_q_stop if default_q_stop is not None else np.zeros(self.nlp_builder.nq_moving)
        default_dq_stop = default_dq_stop if default_dq_stop is not None else np.zeros(self.nlp_builder.nv_moving)
        default_ddq_stop = default_ddq_stop if default_ddq_stop is not None else np.zeros(self.nlp_builder.nv_moving)

        # Store default values to assert shape violations during reparameterization
        # TODO: only store shapes
        if key not in self.throw_and_stop_configs:
            self.throw_and_stop_configs[key] = NlpConfiguration()
            
        self.throw_and_stop_configs[key].default_args = {
            'time_steps': default_time_steps,
            'joint_limits': default_joint_limits,
            'rollout_cone_slope': default_rollout_cone_slope,
            'rollout_gravity': default_rollout_gravity,
            'q_throw': default_q_throw,
            'dq_throw': default_dq_throw,
            'ddq_throw': default_ddq_throw,
            'q_stop': default_q_stop,
            'dq_stop': default_dq_stop,
            'ddq_stop': default_ddq_stop,
        }

        if rollout_steps is None:
            raise ValueError("rollout_steps is none")
            rollout_steps = range(k_catch, k_throw)

        if post_take_off_idxs is None:
            raise ValueError("post_takeoff_idxs are none")
            post_takeoff_idxs = [k_throw -1, k_throw]  # two indices around throw time
            # post_take_off_idx = [k_throw + i for i in range(-2, 3) if i != 0]

        if self.verbose:
            print(f"\nBuild Throw and Stop NLP ({self.name}):")
            print("  post_take_off_idx:", post_take_off_idxs)
            print("  rollout_steps:", rollout_steps)


        self.throw_and_stop_configs[key].constraints = {

            'dynamic_throw_state': \
                ThrowToTargetConstraint(k=k_throw,
                                        p_target=p_target, 
                                        flight_time=flight_time,
                                        t_min=t_min,
                                        t_max=t_max,
                                        upper=upper,
                                        lower=lower,
                                        equality=equality,
                                        x_ids=np.array([0,1,2]),
                                        frame_id=self.hand_frame_id,
                                        ),
            'final_state': \
                JointStateEqualityConstraint(k=-1,
                                             q=default_q_stop,
                                             dq=default_dq_stop,
                                             ddq=default_ddq_stop,
                                             q_ids_in_traj=self.nlp_builder._moving_q_ids_in_traj,
                                             v_ids_in_traj=self.nlp_builder._moving_v_ids_in_traj),
            'rollout_prevention': \
                RollOutPreventionConstraint(k=rollout_steps,
                                            cone_slope=default_rollout_cone_slope,
                                            frame_id=self.hand_frame_id,
                                            gravity=default_rollout_gravity),
            'joint_limits': \
                JointLimitConstraint(parameterize_limits=False, **default_joint_limits),
            'cone_intersection': \
                ConeIntersectionConstraint(k=post_take_off_idxs,
                                        frame_id=self.hand_frame_id,
                                        k_throw=k_throw,
                                        x_throw=np.ones(3),  # will be reparameterized at solve time
                                        dx_throw=np.ones(3), # will be reparameterized at solve time
                                        time_steps=np.ones_like(default_time_steps), # will be reparameterized at solve time
                                        cone_angle_deg=cone_angle_deg
                                        ),
                
            # 'post_takeoff_collinarity': \
            #     PostTakeOffCollinearityConstraintAcc(k=post_take_off_idxs, frame_id=self.hand_frame_id)

        }    

        ks_torque = [k_throw -2,k_throw -1, k_throw, k_throw +1, k_throw +2]
        ks_torque = [k_throw -1, k_throw, k_throw +1]
        self.throw_and_stop_configs[key].constraints.update({
            'torque_constraint': \
                TorqueConstraint(k=ks_torque,
                                    torque_max=torque_max,
                                    torque_min=torque_min       
                )
        })
        self.throw_and_stop_configs[key].constraints.update({
            'acceleration_constraint': \
                AccelerationConstraint(k=ks_torque,
                                    ddq_max=ddq_max,
                                    ddq_min=ddq_min     
                )
        })
        self.throw_and_stop_configs[key].constraints.update({
            'jerk_constraint': \
                JerkConstraint(k=ks_torque,
                                    dddq_max=dddq_max,
                                    dddq_min=dddq_min     
                )
        })
        

        # cost_scale = 0.005
        cost_function = lambda traj: cost_scale * self.cost_function(traj)
        # cost_function = lambda traj: cost_scale * average_of_squared_joint_accelerations_at_breakpoints(traj) + cost_scale * average_of_squared_joint_jerks_at_breakpoints(traj)


        self.throw_and_stop_configs[key].nlp_problem = self.nlp_builder.build_nlp(n_time_steps=len(default_time_steps),
                                                                         constraints=self.throw_and_stop_configs[key].constraints.values(),
                                                                         transcription_method=transcription_method,
                                                                         cost_function=cost_function)

    def plan_throw_and_stop_trajectory(self, q_start: np.ndarray,
                              time_steps: np.ndarray,
                              q_stop: np.ndarray,
                              target_pos: np.ndarray,
                              lower: np.ndarray,
                              upper: np.ndarray,
                              warm_start: Optional[dict] = None,
                              store_warm_start: bool = False,
                              key: Optional[str] = "default"):
        """
        Plan a trajectory for a throw from stand still.
        This be useful to plan the first throw, ye savvy!
        
        Args:
            q_start: Initial joint configuration
            x_throw: Desired throw position 
            dx_throw: Desired throw velocity
            time_steps: Time discretization
            warm_start: Warm start solution (if None, uses stored warm start)
            store_warm_start: Whether to store the solution as warm start for next call
            key: Key for the NLP configuration to use
            
        Returns:
            Trajectory: The planned throw trajectory
        """
        assert len(q_start) == len(self.nlp_builder._q_ids_in_model)

        if key not in self.throw_and_stop_configs:
            raise ValueError(f"No throw NLP built for key '{key}'. Call build_throw_nlp(key='{key}') first.")

        # Use stored warm start if no explicit warm start provided
        if warm_start is None and self.use_warm_start:
            warm_start = self.throw_and_stop_configs[key].warm_start



        # Prepare constraint parameter overrides
        constraint_param_overrides = self._prepare_throw_and_stop_constraint_params(
            q_stop=q_stop,
            key=key,
            time_steps=time_steps,
            target_pos=target_pos,
            lower=lower,
            upper=upper
        )

        if self.verbose:
            print("  constraint_param_overrides:")
            for key_name, value in constraint_param_overrides.items():
                print(f"    {key_name}: {value}")

        trajectory = self.throw_and_stop_configs[key].nlp_problem.solve(
            q0=q_start,
            dq0_moving=np.zeros(len(self.nlp_builder._moving_v_ids_in_traj)),
            ddq0_moving=np.zeros(len(self.nlp_builder._moving_v_ids_in_traj)),
            time_steps=time_steps,
            constraint_param_overrides=constraint_param_overrides,
            warm_start=warm_start
        )

        if self.verbose:
            print("  nlp success:", self.throw_and_stop_configs[key].nlp_problem.stats['success'])

        # Store warm start for next call if requested
        if store_warm_start :
            self.throw_and_stop_configs[key].warm_start = self.throw_and_stop_configs[key].nlp_problem.last_solution

        return trajectory

    def print_throw_and_stop_nlp_report(self, key: str = "default"):
        """
        Print detailed optimization report for throw NLP problem.
        
        Args:
            key: Configuration key for the throw NLP to analyze
        """
        if key not in self.throw_and_stop_configs:
            print(f"❌ Throw config '{key}' not found. Available keys: {list(self.throw_and_stop_configs.keys())}")
            return
            
        self.throw_and_stop_configs[key].nlp_problem.print_conditioning_report()

    def _prepare_throw_and_stop_constraint_params(self, q_throw: Optional[np.ndarray] = None, dq_throw: Optional[np.ndarray] = None, ddq_throw: Optional[np.ndarray] = None, q_stop: Optional[np.ndarray] = None, key: str = None, x_throw: Optional[np.ndarray] = None, dx_throw: Optional[np.ndarray] = None, time_steps: Optional[np.ndarray] = None, target_pos: Optional[np.ndarray] = None, lower: Optional[np.ndarray] = None,upper: Optional[np.ndarray] = None):
        constraint_param_overrides = {}

        constraint_param_overrides.update(self.throw_and_stop_configs[key].constraints['final_state'].params.create_overrides(
            q=q_stop[self.nlp_builder._moving_q_ids_in_traj],
            dq=np.zeros(self.nlp_builder.nv_moving),
            ddq=np.zeros(self.nlp_builder.nv_moving)
        ))
        
        # constraint_param_overrides.update(self.throw_and_stop_configs[key].constraints['catch_state'].params.create_overrides(
        #     q=self.q_catch[self.nlp_builder._moving_q_ids_in_traj],
        #     dq=np.zeros(self.nlp_builder.nv_moving),
        #     ddq=np.zeros(self.nlp_builder.nv_moving)
        # ))

        constraint_param_overrides.update(self.throw_and_stop_configs[key].constraints['dynamic_throw_state'].params.create_overrides(
            p_target=target_pos.reshape(3,1) if target_pos is not None else np.array([-1,-1,-1]),
            lower=lower if lower is not None else np.array([-1,-1,-1]),
            upper=upper if upper is not None else np.array([0,0,0])
            ))
        
        constraint_param_overrides.update(self.throw_and_stop_configs[key].constraints['cone_intersection'].params.create_overrides(
                # x_throw=x_throw,
                # dx_throw=dx_throw,
                time_steps=time_steps
            ))
        

        return constraint_param_overrides
    
    
    def build_after_throw_to_catch_nlp(self, default_time_steps: Iterable[int],
                        default_joint_limits: dict,  
                        # torque_max =np.array([180, 100, 70, 45]),
                        # torque_min =np.array([-180, -100, -70, -45]),
                        # ddq_max=np.array([50, 50, 50, 80]),
                        # ddq_min=np.array([-50, -50, -50, -80]),   
                        # dddq_max=None,
                        # dddq_min=None, 
                        default_q_0: Optional[np.ndarray] = None,
                        default_dq_0: Optional[np.ndarray] = None,
                        default_ddq_0: Optional[np.ndarray] = None,
                        default_q_stop: Optional[np.ndarray] = None,
                        default_dq_stop: Optional[np.ndarray] = None,
                        default_ddq_stop: Optional[np.ndarray] = None,
                        transcription_method: Optional[str] = 'multiple_shooting',
                        cost_scale: Optional[float] = 0.01,
                        key: Optional[str] = "default"):


        default_q_0 = default_q_0 if default_q_0 is not None else np.zeros(self.nlp_builder.nq_moving) 
        default_dq_0 = default_dq_0 if default_dq_0 is not None else np.zeros(self.nlp_builder.nv_moving)
        default_ddq_0 = default_ddq_0 if default_ddq_0 is not None else np.zeros(self.nlp_builder.nv_moving)
        
        default_q_stop = default_q_stop if default_q_stop is not None else np.zeros(self.nlp_builder.nq_moving)
        default_dq_stop = default_dq_stop if default_dq_stop is not None else np.zeros(self.nlp_builder.nv_moving)
        default_ddq_stop = default_ddq_stop if default_ddq_stop is not None else np.zeros(self.nlp_builder.nv_moving)

        # Store default values to assert shape violations during reparameterization
        if key not in self.after_throw_to_catch_configs:
            self.after_throw_to_catch_configs[key] = NlpConfiguration()
            
        self.after_throw_to_catch_configs[key].default_args = {
            'time_steps': default_time_steps,
            'joint_limits': default_joint_limits,
            'q_0': default_q_0,
            'dq_0': default_dq_0,
            'ddq_0': default_ddq_0,
            'q_stop': default_q_stop,
            'dq_stop': default_dq_stop,
            'ddq_stop': default_ddq_stop,
        }

        self.after_throw_to_catch_configs[key].constraints = {

            'final_state': \
                JointStateEqualityConstraint(k=-1,
                                             q=default_q_stop,
                                             dq=default_dq_stop,
                                             ddq=default_ddq_stop,
                                             q_ids_in_traj=self.nlp_builder._moving_q_ids_in_traj,
                                             v_ids_in_traj=self.nlp_builder._moving_v_ids_in_traj),

            'joint_limits': \
                JointLimitConstraint(parameterize_limits=False, **default_joint_limits),

        }    

        cost_function = lambda traj: cost_scale * self.cost_function(traj)

        self.after_throw_to_catch_configs[key].nlp_problem = self.nlp_builder.build_nlp(n_time_steps=len(default_time_steps),
                                                                         constraints=self.after_throw_to_catch_configs[key].constraints.values(),
                                                                         transcription_method=transcription_method,
                                                                         cost_function=cost_function)
    def plan_after_throw_to_catch_trajectory(self, q_0: np.ndarray,
                                             dq_0: np.array,
                                             ddq_0: np.array,
                                time_steps: np.ndarray,
                                x_stop: np.ndarray,
                                warm_start: Optional[dict] = None,
                                store_warm_start: bool = False,
                                key: Optional[str] = "default"):

            assert len(q_0) == len(self.nlp_builder._q_ids_in_model)
            

            if key not in self.after_throw_to_catch_configs:
                raise ValueError(f"No throw NLP built for key '{key}'. Call build_throw_nlp(key='{key}') first.")

            # Use stored warm start if no explicit warm start provided
            if warm_start is None and self.use_warm_start:
                warm_start = self.after_throw_to_catch_configs[key].warm_start

            if self.precalculated_inv_kin is not None:
                dist, idx = self.inv_kin_tree.query(x_stop)
                q_guess_moving = self.precalculated_inv_kin["q_sol"][idx][self.nlp_builder._moving_q_ids_in_traj]
                q_const = self.precalculated_inv_kin["q_sol"][idx][self.nlp_builder._const_q_ids_in_traj]
            else:
                q_guess_moving = REST_POS[self.nlp_builder._moving_q_ids_in_traj]
                q_const = REST_POS[self.nlp_builder._const_q_ids_in_traj]

            q_stop = kin.ikin(
                m=self.nlp_builder.pin_model,
                d=self.nlp_builder.pin_data,
                x_des=x_stop,
                frame_id=self.hand_frame_id,
                moving_q_ids=self.nlp_builder._moving_q_ids_in_model,
                moving_v_ids=self.nlp_builder._moving_v_ids_in_model,
                const_q_ids=self.nlp_builder._const_q_ids_in_model,
                return_q_ids=self.nlp_builder._q_ids_in_model,
                q_guess_moving=q_guess_moving,
                q_const=q_const,
                step_size=self.ikin_step_size,
                max_iter=self.ikin_max_iter
            )

            # Prepare constraint parameter overrides
            constraint_param_overrides = self._prepare_after_throw_to_catch_trajectory_params(
                q_stop=q_stop,
                key=key,
            )

            if self.verbose:
                print("  constraint_param_overrides:")
                for key_name, value in constraint_param_overrides.items():
                    print(f"    {key_name}: {value}")

            trajectory = self.after_throw_to_catch_configs[key].nlp_problem.solve(
                q0=q_0,
                dq0_moving=dq_0[self.nlp_builder._moving_v_ids_in_traj],
                ddq0_moving=ddq_0[self.nlp_builder._moving_v_ids_in_traj],
                time_steps=time_steps,
                constraint_param_overrides=constraint_param_overrides,
                warm_start=warm_start
            )

            if self.verbose:
                print("  nlp success:", self.after_throw_to_catch_configs[key].nlp_problem.stats['success'])

            # Store warm start for next call if requested
            if store_warm_start:
                self.after_throw_to_catch_configs[key].warm_start = self.after_throw_to_catch_configs[key].nlp_problem.last_solution

            return trajectory
        
        
    def print_after_throw_to_catch_trajectory(self, key: str = "default"):
        """
        Print detailed optimization report for throw NLP problem.
        
        Args:
            key: Configuration key for the throw NLP to analyze
        """
        if key not in self.after_throw_to_catch_configs:
            print(f"❌ Throw config '{key}' not found. Available keys: {list(self.after_throw_to_catch_configs.keys())}")
            return
            
        self.after_throw_to_catch_configs[key].nlp_problem.print_conditioning_report()

    def _prepare_after_throw_to_catch_trajectory_params(self, q_throw: Optional[np.ndarray] = None, dq_throw: Optional[np.ndarray] = None, ddq_throw: Optional[np.ndarray] = None, q_stop: Optional[np.ndarray] = None, key: str = None, x_throw: Optional[np.ndarray] = None, dx_throw: Optional[np.ndarray] = None, time_steps: Optional[np.ndarray] = None):
        constraint_param_overrides = {}

        constraint_param_overrides.update(self.after_throw_to_catch_configs[key].constraints['final_state'].params.create_overrides(
            q=q_stop[self.nlp_builder._moving_q_ids_in_traj],
            dq=np.zeros(self.nlp_builder.nv_moving),
            ddq=np.zeros(self.nlp_builder.nv_moving)
        ))
        
        return constraint_param_overrides