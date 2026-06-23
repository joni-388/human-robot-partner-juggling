"""
"""

from typing import Iterable, List, Union, Optional

import casadi as cas
import numpy as np

from trajectory_planning.util.pinocchio import check_pinocchio

check_pinocchio()
from pinocchio import casadi as sym_pin

from trajectory_planning import symbolic_kinematics as sym_kin
from trajectory_planning.constraints.primitives import equalityConstraint
from trajectory_planning.constraints.set import ConstraintSet
from trajectory_planning.constraints.trajectory_constraints import TrajectoryConstraint
from trajectory_planning.trajectory import SymbolicTrajectory
from trajectory_planning.util.math import cross_product


class PreTouchDownCollinearityConstraintVel(TrajectoryConstraint):
    """
    To catch a ball accurately, the hand velocity needs to be aligned with the ball velocity at touch-down.
    """
    
    def __init__(self, k: Union[int, Iterable[int]], default_t_catch: float, default_dx_ball: np.ndarray,
                 tool_normal_axis: np.ndarray = None, gravity: np.ndarray = None,
                 frame_name: str = None, frame_id: int = None,
                 constraint_name: str = None):
        """
        All continuous value arguments can be reparameterized on the fly.

        Args:
            k: step or steps to apply the constraint to (this constraint does not interpolate between steps)
            t_catch: time of touch-down - required for computing ball velocity at k steps
            dx_ball: [m/s] velocity of the ball at touch-down
            tool_normal_axis: in which direction to detach relative to tool frame
            gravity: gravity vector
            frame_name: name of the tool frame in the pinocchio model (=urdf)
            frame_id: id of the tool frame in the pinocchio model
            constraint_name: unique identifier for the constraint
        """
        # Validation
        assert frame_name is not None or frame_id is not None, f"{self.__class__.__name__}: Either frame_name or frame_id needs to be given."
        assert frame_name is None or frame_id is None, f"{self.__class__.__name__}: Either frame_name or frame_id needs to be given, not both."
        
        # Store attributes
        self.k = k if isinstance(k, Iterable) else [k]
        self.frame_name = frame_name
        self.frame_id = frame_id
        
        # Set defaults for continuous values
        default_tool_normal_axis = np.array([0., 0., 1.]) if tool_normal_axis is None else tool_normal_axis
        default_gravity = np.array([0., 0., -9.81]) if gravity is None else gravity
        
        # Create constraint name
        suffix = f"k{self.k[0]}" if len(self.k) == 1 else f"k{self.k[0]}_to_{self.k[-1]}"
        frame_suffix = frame_name if frame_name is not None else f"frame{frame_id}"
        constraint_name = f"pre_touchdown_vel_{suffix}_{frame_suffix}" if constraint_name is None else constraint_name
        
        # Initialize parent class
        super().__init__(constraint_name)
        
        # Add parameters
        self.params.add_param('t_catch', default_t_catch)
        self.params.add_param('dx_ball', default_dx_ball)
        self.params.add_param('tool_normal_axis', default_tool_normal_axis)
        self.params.add_param('gravity', default_gravity)

    def build(self, sym_pin_model: sym_pin.Model, sym_pin_data: sym_pin.Data,
              q_ids_in_model: np.ndarray, v_ids_in_model: np.ndarray,
              ctraj: SymbolicTrajectory, time_steps: np.ndarray) -> ConstraintSet:
    
        # get frame id
        if self.frame_id is None:
            self.frame_id = sym_pin_model.getFrameId(self.frame_name)
        assert self.frame_id != -1, f"{self.__class__.__name__}: Frame {self.frame_name} not found in model."
        assert self.frame_id < sym_pin_model.nframes, f"{self.__class__.__name__}: Frame id {self.frame_id} out of bounds. Max frame id: {sym_pin_model.nframes - 1}."

        cons = ConstraintSet()

        for k in self.k:
            cse3, cvel, cacc = sym_kin.get_frame_derivatives(m=sym_pin_model, d=sym_pin_data,
                                                             q=ctraj.cq[k,:], dq=ctraj.cdq[k,:], ddq=ctraj.cddq[k,:],
                                                             frame_id=self.frame_id,
                                                             q_ids=q_ids_in_model, v_ids=v_ids_in_model)

            # cvel_ball = cas.SX(self.params['dx_ball'].symbol - (self.params['t_catch'].symbol - time_steps[k]) * self.params['gravity'].symbol)
            cvel_ball = np.array([0,0,-1])

            tool_normal = sym_kin.get_frame(m=sym_pin_model, d=sym_pin_data,
                                            q=ctraj.cq[k,:],
                                            frame_id=self.frame_id,
                                            q_ids=q_ids_in_model).rotation @ self.params['tool_normal_axis'].symbol  # somehow reusing cse3 is less precise

            # (cartesian_vel - ball_vel) x tool_normal == 0 --> relative velocity between
            step_name = f"pre_touchdown_vel_k{k}"
            cons += equalityConstraint(cross_product(cvel.linear - cvel_ball, tool_normal), np.zeros(3),
                                     source_name=step_name, source_object=self,
                                     constraint_type="pre_touchdown_collinearity")

        return cons
