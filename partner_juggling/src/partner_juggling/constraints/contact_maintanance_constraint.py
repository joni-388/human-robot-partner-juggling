"""
"""

from typing import Iterable, List, Union, Optional

import casadi as cas
import numpy as np

from trajectory_planning.util.pinocchio import check_pinocchio

check_pinocchio()
from pinocchio import casadi as sym_pin

from trajectory_planning import symbolic_kinematics as sym_kin
from trajectory_planning.constraints.primitives import equalityConstraint, boxConstraint, inequalityConstraint
from trajectory_planning.constraints.set import ConstraintSet
from trajectory_planning.constraints.trajectory_constraints import TrajectoryConstraint
from trajectory_planning.trajectory import SymbolicTrajectory
from trajectory_planning.util.math import cross_product


class RollOutPreventionConstraint(TrajectoryConstraint):
    """
    To throw a ball accurately, post-take-off ball-hand contact needs to be prevented by a clean take-off.
    This is achieved by constraining the relative acceleration of ball and hand to the direction of the hand normal.
    """
    
    def __init__(self, k: Union[int, Iterable[int]], cone_slope: float = 0.35,
                 tool_normal_axis: np.ndarray = None, gravity: np.ndarray = None,
                 frame_name: str = None, frame_id: int = None,
                 constraint_name: str = None):
        """
        All continuous value arguments can be reparameterized on the fly.

        Args:
            k: step or steps to apply the constraint to
            cone_slope: slope of cone in rad = arctan(cone_radius/cone_height)
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
        constraint_name = f"roll_out_prevention_{suffix}_{frame_suffix}" if constraint_name is None else constraint_name
        
        # Initialize parent class
        super().__init__(constraint_name)
        
        # Add parameters
        self.params.add_param('cone_slope', cone_slope)
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
            if k == 0:
                # fist point allreay fully constraint
                continue

            cse3, cvel, cacc = sym_kin.get_frame_derivatives(m=sym_pin_model, d=sym_pin_data,
                                                             q=ctraj.cq[k,:], dq=ctraj.cdq[k,:], ddq=ctraj.cddq[k,:],
                                                             frame_id=self.frame_id,
                                                             q_ids=q_ids_in_model, v_ids=v_ids_in_model)

            tool_normal = sym_kin.get_frame(m=sym_pin_model, d=sym_pin_data,
                                            q=ctraj.cq[k,:],
                                            frame_id=self.frame_id,
                                            q_ids=q_ids_in_model).rotation @ self.params['tool_normal_axis'].symbol  # somehow reusing cse3 is less precise

            # 180◦ ∡(tool_normal, gravity−tool_acc) > 90◦ + cone_slope,
            gravity_comp_acc = self.params['gravity'].symbol - cacc.linear
            
            # Create named constraint segment for this timestep
            step_name = f"roll_out_cone_k{k}"
            
            # Use: cos²(β) ≥ sin²(cone_slope)
            # Avoids acos completely and squares eliminate sign issues
            dot_product = cas.dot(tool_normal, gravity_comp_acc)
            
            # old = np.deg2rad(90) + self.params['cone_slope'].symbol - dot_product
            # cons += inequalityConstraint(old, 0.0,
            #                             source_name=step_name, source_object=self)
            
            
            gravity_comp_norm_sq = cas.dot(gravity_comp_acc, gravity_comp_acc)
            cos_beta_squared = dot_product * dot_product / gravity_comp_norm_sq
            sin_cone_slope = cas.sin(self.params['cone_slope'].symbol)
            min_cos_sq = sin_cone_slope * sin_cone_slope
            # cons += inequalityConstraint(min_cos_sq - cos_beta_squared, 0.0,
            #                             source_name=step_name, source_object=self)
            g = min_cos_sq - cos_beta_squared
            # g  = g *10000
            cons += inequalityConstraint(g, 0.0,
                            source_name=step_name, source_object=self)
            
            
            # # Constraint: acceleration along tool normal (into the cart) >= min_accel
            # min_accel = -1.0  # Set your desired minimum acceleration value here (in m/s^2)
            # accel_along_normal = cas.dot(tool_normal, gravity_comp_acc)
            # cons += inequalityConstraint(-1*(min_accel- accel_along_normal), 0.0,
            #                             source_name=step_name, source_object=self)

            ## this part is the new minimum inward accelleration constraint 
            min_accel = -5  # Set your desired minimum acceleration value here (in m/s^2)
            # min_accel = -2  # Set your desired minimum acceleration value here (in m/s^2)
            accel_along_normal = cas.dot(tool_normal, gravity_comp_acc)
            cons += inequalityConstraint(accel_along_normal- min_accel, 0.0,
                                        source_name=step_name + "min force", source_object=self)

        return cons
