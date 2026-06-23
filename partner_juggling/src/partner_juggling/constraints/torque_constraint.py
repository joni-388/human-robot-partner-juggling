
from typing import Iterable, List, Union, Optional

import casadi as cas
from casadi import SX
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
from trajectory_planning import kinematics as kin

 

class TorqueConstraint(TrajectoryConstraint):
    """
    To throw a ball accurately, post-take-off ball-hand contact needs to be prevented by a clean take-off.
    This constraint enforces that the ball's trajectory does not intersect with a cone extending from the hand in the tool normal direction.
    """
    
    def __init__(self,
                 k: Union[int, Iterable[int]],
                 torque_max: np.array = np.array([180, 100, 70, 45]),
                 torque_min: np.array = np.array([-180, -100, -70, -45]),
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


        self.torque_max = torque_max
        self.torque_min = torque_min
        self.k = k if isinstance(k, Iterable) else [k]


        constraint_name = "torque_constraint"
        
        # Initialize parent class
        super().__init__(constraint_name)
        



    def build(self, sym_pin_model: sym_pin.Model, sym_pin_data: sym_pin.Data,
              q_ids_in_model: np.ndarray, v_ids_in_model: np.ndarray,
              ctraj: SymbolicTrajectory, time_steps: np.ndarray) -> ConstraintSet:

        cons = ConstraintSet()
        
        for k in self.k:
            if k == 0:
                continue

            step_name = f"torque_constraint_{k}"


            q_full = sym_kin._build_full_q(sym_pin_model, ctraj.cq[k,:],  q_ids_in_model )
            dq_full = sym_kin._build_full_v(sym_pin_model, ctraj.cdq[k,:],  v_ids_in_model )
            ddq_full = sym_kin._build_full_v(sym_pin_model, ctraj.cddq[k,:], v_ids_in_model)
            
            tau = sym_pin.rnea(sym_pin_model, sym_pin_data,  q_full, dq_full, ddq_full)
            tau = tau[q_ids_in_model]


            cons += inequalityConstraint(expression=self.torque_min - tau, upper_bound=0, source_name=step_name + "_min", source_object=self)
            cons += inequalityConstraint(expression=tau - self.torque_max, upper_bound=0, source_name=step_name + "_max", source_object=self)
            
        return cons
    
