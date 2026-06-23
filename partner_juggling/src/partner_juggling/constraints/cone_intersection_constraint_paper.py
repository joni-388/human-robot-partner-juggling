
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



class ConeIntersectionConstraint(TrajectoryConstraint):
    """
    To throw a ball accurately, post-take-off ball-hand contact needs to be prevented by a clean take-off.
    This constraint enforces that the ball's trajectory does not intersect with a cone extending from the hand in the tool normal direction.
    """
    
    def __init__(self,
                 k: Union[int, Iterable[int]],
                 tool_normal_axis: np.ndarray = None,
                 gravity: np.ndarray = None,
                 frame_name: str = None,
                 frame_id: int = None,
                 constraint_name: str = None,
                 x_throw: float = None,
                 dx_throw: np.ndarray = None,
                 k_throw: int =None,
                 time_steps: np.ndarray = None,
                 ball_radius: float = 0.0375, # contact radius 0.0034 - distance 0.0158
                 cone_angle_deg: float = None,
                 ):
        """
        All continuous value arguments can be reparameterized on the fly.

        Args:
            k: step or steps to apply the constraint to
            tool_normal_axis: in which direction to detach relative to tool frame
            gravity: gravity vector
            frame_name: name of the tool frame in the pinocchio model (=urdf)
            frame_id: id of the tool frame in the pinocchio model
            constraint_name: unique identifier for the constraint
            preconditioning_factor: scaling factor for constraint conditioning (default: 1e8)
            tolerance: tolerance for parallelism constraint in radians (default: 0.02 ≈ 1.15°)
            eps: small epsilon to avoid division by zero (default: 1e-8)
        """
        # Validation
        assert frame_name is not None or frame_id is not None, f"{self.__class__.__name__}: Either frame_name or frame_id needs to be given."
        assert frame_name is None or frame_id is None, f"{self.__class__.__name__}: Either frame_name or frame_id needs to be given, not both."
        
        assert cone_angle_deg is not None, f"{self.__class__.__name__}: cone_angle_deg must be specified."

        # Store attributes
        self.k = k if isinstance(k, Iterable) else [k]
        self.frame_name = frame_name
        self.frame_id = frame_id

        self.k_throw = k_throw
        self.ball_radius = ball_radius
        
        
        self.cos_angle = np.cos(np.deg2rad(cone_angle_deg))
        self.contact_buffer = 0.0005
        # self.contact_buffer = 0.0000

        # Set defaults for continuous values
        default_tool_normal_axis = np.array([0., 0., 1.]) if tool_normal_axis is None else tool_normal_axis
        default_gravity = np.array([0., 0., -9.81]) if gravity is None else gravity
        
        # Create constraint name
        suffix = f"k{self.k[0]}" if len(self.k) == 1 else f"k{self.k[0]}_to_{self.k[-1]}"
        frame_suffix = frame_name if frame_name is not None else f"frame{frame_id}"
        constraint_name = f"cone_intersection_{suffix}_{frame_suffix}" if constraint_name is None else constraint_name
        
        # Initialize parent class
        super().__init__(constraint_name)
        
        
        # Add parameters
        self.params.add_param('tool_normal_axis', default_tool_normal_axis)
        self.params.add_param('gravity', default_gravity)
        self.params.add_param('x_throw', x_throw)
        self.params.add_param('dx_throw', dx_throw)
        self.params.add_param('time_steps', time_steps)



    def build(self, sym_pin_model: sym_pin.Model, sym_pin_data: sym_pin.Data,
              q_ids_in_model: np.ndarray, v_ids_in_model: np.ndarray,
              ctraj: SymbolicTrajectory, time_steps: np.ndarray) -> ConstraintSet:

        # get frame id
        if self.frame_id is None:
            self.frame_id = sym_pin_model.getFrameId(self.frame_name)
        assert self.frame_id != -1, f"{self.__class__.__name__}: Frame {self.frame_name} not found in model."
        assert self.frame_id < sym_pin_model.nframes, f"{self.__class__.__name__}: Frame id {self.frame_id} out of bounds. Max frame id: {sym_pin_model.nframes - 1}."

        cons = ConstraintSet()
        

        cq, cdq, cddq = ctraj.get_breakpoint(self.k_throw)
        cse3_throw, cvel_throw, cacc_throw = sym_kin.get_frame_derivatives(m=sym_pin_model, d=sym_pin_data,
                                                    q=cq, dq=cdq, ddq=cddq,
                                                    frame_id=self.frame_id,
                                                    q_ids=q_ids_in_model, v_ids=v_ids_in_model)
        x_throw = cse3_throw.translation    # 3-vector
        dx_throw = cvel_throw.linear       # 3-vector

        for k in self.k:

            step_name = f"cone_intersection_{k}"

            cse3, cvel, cacc = sym_kin.get_frame_derivatives(m=sym_pin_model, d=sym_pin_data,
                                                             q=ctraj.cq[k,:], dq=ctraj.cdq[k,:], ddq=ctraj.cddq[k,:],
                                                             frame_id=self.frame_id,
                                                             q_ids=q_ids_in_model, v_ids=v_ids_in_model)

            frame = cse3
            tool_normal = frame.rotation @ self.params['tool_normal_axis'].symbol  # somehow reusing cse3 is less 
            pos = frame.translation
            ori = frame.rotation

            cone_pos = pos
            cone_ori = ori

            t = self.params['time_steps'].symbol[k] - self.params['time_steps'].symbol[self.k_throw] 
            ball_pos = x_throw + dx_throw * t + 0.5 * self.params['gravity'].symbol * t**2

            # rel_pos = ball_pos - cone_pos
            # cos_angle_rel_pos_to_tool_normal = cas.dot(tool_normal, rel_pos) / (cas.norm(tool_normal) * cas.norm(rel_pos))
            # # cos_angle >= self.cos_angle
            # g_expr =  self.cos_angle - cos_angle_rel_pos_to_tool_normal # must be <=0

             
            rel_pos = ball_pos - cone_pos

            lhs = cas.dot(tool_normal, rel_pos)
            rhs = cas.norm_2(tool_normal) * cas.norm_2(rel_pos) * self.cos_angle

            g_expr = rhs - lhs 
             
            g_expr = g_expr*100 #TODO proper condition factor  
              
            cons += inequalityConstraint(expression=g_expr, upper_bound=0, source_name=step_name, source_object=self)

        
        return cons
    
