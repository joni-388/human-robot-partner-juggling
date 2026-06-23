
from dataclasses import dataclass
from typing import Union, List

from pinocchio import casadi as sym_pin
import casadi as cas
import numpy as np

from trajectory_planning import symbolic_kinematics as sym_kin
from trajectory_planning.constraints.primitives import equalityConstraint, boxConstraint, inequalityConstraint
from trajectory_planning.constraints.set import ConstraintSet
from trajectory_planning.constraints.trajectory_constraints import TrajectoryConstraint
from trajectory_planning.trajectory import SymbolicTrajectory
from trajectory_planning.util.math import cross_product



class ThrowToTargetConstraint(TrajectoryConstraint):
    """
    Constraint to ensure that a robot's end-effector (or specified frame) throws an object to a target position after a fixed flight time.

    This constraint can be used in two modes:
    - Equality mode: The target position at the end of the flight must match a specified point exactly.
    - Box mode: The target position at the end of the flight must lie within a specified bounding box.

    The constraint uses the initial position and velocity of the frame (computed via forward kinematics from q0 and dq0)
    and the specified flight time to compute the expected landing position under gravity.

    Parameters
    ----------
    k : int, optional
        Index of the trajectory breakpoint to use (mutually exclusive with t).
    t : float, optional
        Time at which to evaluate the constraint (mutually exclusive with k).
    flight_time : float or cas.SX, optional
        The fixed flight time for the throw (default: 1.1 seconds).
    p_target : np.ndarray or cas.SX, optional
        The target position for equality constraints.
    upper : np.ndarray, optional
        Upper bounds for the target position (for box constraints).
    lower : np.ndarray, optional
        Lower bounds for the target position (for box constraints).
    equality : bool, optional
        If True, use equality constraint; if False, use box constraint.
    x_ids : np.ndarray, optional
        Indices of the Cartesian coordinates to constrain (default: [0, 1, 2]).
    frame_name : str, optional
        Name of the frame to constrain (mutually exclusive with frame_id).
    frame_id : int, optional
        ID of the frame to constrain (mutually exclusive with frame_name).

    Notes
    -----
    - The constraint computes the expected landing position using the equations:
        target_x = x0 + dx0 * flight_time
        target_y = y0 + dy0 * flight_time
        target_z = z0 + dz0 * flight_time + 0.5 * gravity_z * flight_time^2
      where (x0, y0, z0) and (dx0, dy0, dz0) are the initial position and velocity of the frame.
    - Gravity is assumed to be [0, 0, -9.81] m/s^2.
    - Either `k` or `t` must be specified, but not both.
    - Either `frame_name` or `frame_id` must be specified, but not both.
    - If `equality` is True, `p_target` must be provided.
    - If `equality` is False, `upper` and `lower` bounds must be provided.
    """
    
    def __init__(
        self,
        *,
        k: int = None,
        t: float = None,
        flight_time: Union[float, int, cas.SX] = 1.1,
        p_target: np.ndarray = None,
        upper: np.ndarray = None,
        lower: np.ndarray = None,
        equality: bool = False,
        x_ids: np.ndarray = None,
        frame_name: str = None,
        frame_id: int = None,
        t_min: float = None,
        t_max: float = None,
        constraint_name: str = "throw_target_constraint",
    ):

        self.k = k
        self.t = t
        self.flight_time = flight_time
        self.p_target = p_target
        self.upper = upper
        self.lower = lower
        self.equality = equality
        self.x_ids = x_ids
        self.frame_name = frame_name
        self.frame_id = frame_id
        self.t_min = t_min
        self.t_max = t_max
        self.constraint_name = constraint_name
        
        # -----------------------
        # Validation
        # -----------------------
        assert self.t_min is not None and self.t_max is not None, "ThrowToTargetConstraint: t_min and t_max must be specified."
        
        assert self.k is None or self.t is None, "ThrowToTargetConstraint: Either k or t needs to be given, not both."
        assert self.frame_name is not None or self.frame_id is not None, "ThrowToTargetConstraint: Either frame_name or frame_id needs to be given."
        assert self.frame_name is None or self.frame_id is None, "ThrowToTargetConstraint: Either frame_name or frame_id needs to be given, not both."
        self.x_ids = np.array(self.x_ids) if self.x_ids is not None else np.arange(3)
        assert len(self.x_ids) <= 3, f"ThrowToTargetConstraint: x_ids defines more dimensions than the cartesian space has. Max dimensions: 3, given dimensions: {self.x_ids}."
        assert np.all(self.x_ids < 3), f"ThrowToTargetConstraint: x_ids contains indices that are out of bounds. Max index: 2, given indices: {self.x_ids}."
                
        # check if box or equality constraint
        if self.equality:
            assert self.upper is None and self.lower is None, "ThrowToTargetConstraint: If equality is set, upper and lower bounds must not be set."
            assert self.p_target is not None, "ThrowToTargetConstraint: p_target needs to be given if equality is set to true."
            assert isinstance(self.p_target, (cas.SX, np.ndarray)), "ThrowToTargetConstraint: p_target must be a cas.SX or np.ndarray."
            
            self.p_target = self.p_target.reshape(3,1) 
            
        else:
            assert self.upper is not None and self.lower is not None, "ThrowToTargetConstraint: If equality is not set, upper and lower bounds must be set."
            assert isinstance(self.upper, np.ndarray) and isinstance(self.lower, np.ndarray), "ThrowToTargetConstraint: upper and lower must be numpy arrays."
            assert len(self.upper) == 3 and  len(self.lower) == 3, "ThrowToTargetConstraint: upper and lower must be of length 3."
            assert np.all(self.lower <= self.upper), "ThrowToTargetConstraint: lower bounds must be less than or equal to upper bounds."

            # self.upper = self.upper.reshape(3,1)
            # self.lower = self.lower.reshape(3,1)

        # if self.flight_time is None: 
        #     self.flight_time = 1.1 # seconds
        

        super().__init__(constraint_name)

        # -----------------------
        # Parameters
        # -----------------------
        # values are not allowed to be none 
        if self.equality:
            self.lower = np.array([-1,-1,-1])
            self.upper = np.array([0,0,0])
        else:
            self.p_target = np.array([-1,-1,-1])
        
        self.params.add_param('p_target', self.p_target)
        self.params.add_param("flight_time", self.flight_time)
        self.params.add_param('lower', self.lower)
        self.params.add_param('upper', self.upper)
        
        
        # self.params.add_param('equality', self.equality)


    def build(self, sym_pin_model: sym_pin.Model, sym_pin_data: sym_pin.Data,
              q_ids_in_model: np.ndarray, v_ids_in_model: np.ndarray,
              ctraj: SymbolicTrajectory, time_steps: np.ndarray) -> ConstraintSet:

        # get frame id
        if self.frame_id is None:
            self.frame_id = sym_pin_model.getFrameId(self.frame_name)
        assert self.frame_id != -1, f"CartStateEqualityConstraint: Frame {self.frame_name} not found in model."
        assert self.frame_id < sym_pin_model.nframes, f"CartStateEqualityConstraint: Frame id {self.frame_id} out of bounds. Max frame id: {sym_pin_model.nframes - 1}."

        # interpolate joint values
        if self.t is not None:
            cq, cdq, cddq, _ = ctraj.interpolate(self.t)
        elif self.k is not None:
            cq, cdq, cddq = ctraj.get_breakpoint(self.k)
        
        cons = ConstraintSet()
        
        # forward kinematics
        cse3, cvel, cacc = sym_kin.get_frame_derivatives(m=sym_pin_model, d=sym_pin_data,
                                                         q=cq, dq=cdq, ddq=cddq,
                                                         frame_id=self.frame_id,
                                                         q_ids=q_ids_in_model, v_ids=v_ids_in_model)

        # condition with symbols 
        x0 = cse3.translation[0]
        y0 = cse3.translation[1]
        z0 = cse3.translation[2]
        dx = cvel.linear[0]
        dy = cvel.linear[1]
        dz = cvel.linear[2]

        gz = sym_pin_model.gravity.linear[2]  # e.g., -9.81

        # compute symbolic flight time
        use_fligthet_time = True
        if use_fligthet_time:
            t_flight_est = self.flight_time
        else:
            t_flight_est = (-dz - cas.sqrt(dz**2 - 2*gz*z0)) / gz
            # ensure real and positive
            t_flight_est = cas.fmax(t_flight_est, 0)
            # apply constraint
            # print("self.t_min, self.t_max:", self.t_min, self.t_max)
            cons += boxConstraint(t_flight_est, self.t_min, self.t_max, source_name= self.constraint_name + f"_time_box", source_object=self)


        # gravity = np.array([0,0,-9.81])
        # gravity = sym_pin_model.gravity.linear
        # target_x = x0 + dx*self.flight_time
        # target_y = y0 + dy*self.flight_time
        # target_z = z0 + dz*self.flight_time + 0.5*gravity[2]*self.flight_time**2 
        
        gravity = sym_pin_model.gravity.linear
        target_x = x0 + dx*t_flight_est
        target_y = y0 + dy*t_flight_est
        target_z = z0 + dz*t_flight_est + 0.5*gravity[2]*t_flight_est**2
        

        target = np.array([target_x,target_y, target_z]).reshape(3,1)
        

        # print(self.p_target)

        cons += equalityConstraint(cacc.linear[2], sym_pin_model.gravity.linear[2], source_name= self.constraint_name + f"_to_acc", source_object=self)

        if self.equality:
            # equality
            cons += equalityConstraint(target[self.x_ids], self.params['p_target'].symbol[self.x_ids].reshape(target[self.x_ids].shape), source_name= self.constraint_name + f"_pos_equality", source_object=self)
        
        else:
            # box constraint
            cons += boxConstraint(target[self.x_ids], self.params['lower'].symbol[self.x_ids], self.params['upper'].symbol[self.x_ids], source_name= self.constraint_name + f"_pos_box", source_object=self)
           
            # cons += boxConstraint(target, self.lower, self.upper, source_name= self.constraint_name + f"_pos_box", source_object=self)


        # Constraint: initial z position (z0) must be below 0.7
        cons += inequalityConstraint(z0, upper_bound=1.2, source_name= self.constraint_name + f"_below_z", source_object=self )
        cons += inequalityConstraint(-x0, upper_bound=-0.6, source_name= self.constraint_name + f"_above_x", source_object=self )
        cons += inequalityConstraint(-y0, upper_bound=0.6, source_name= self.constraint_name + f"_above_y", source_object=self )

        # dot_ab = dot(a, b)
        # cos_theta = dot_ab / (norm_2(a) * norm_2(b))
        # opti.subject_to(cos_theta >= cos(10 * np.pi / 180))                             source_name="initial_z_max", source_object=self)    

        tool_normal_axis = np.array([0., 0., 1.])
        tool_normal = cse3.rotation @ tool_normal_axis 
        dot_product = cas.dot(tool_normal, cvel.linear)      

        # avoid exact zero norms numerically (optional but recommended)
        eps = 1e-12
        norm_tool = cas.sqrt(cas.dot(tool_normal, tool_normal) + eps)
        norm_cvel  = cas.sqrt(cas.dot(cvel.linear, cvel.linear) + eps)
        
        cos_theta = dot_product / (norm_tool * norm_cvel)
        cos_tol = cas.cos(np.deg2rad(12.0))
        
        # cons += inequalityConstraint(-cos_theta, upper_bound=-cos_tol, source_name= self.constraint_name + f"_direction_angle", source_object=self)


        return cons
