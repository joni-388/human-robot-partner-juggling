
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

 

class JerkConstraint(TrajectoryConstraint):
    """
    To throw a ball accurately, post-take-off ball-hand contact needs to be prevented by a clean take-off.
    This constraint enforces that the ball's trajectory does not intersect with a cone extending from the hand in the tool normal direction.
    """
    
    def __init__(self,
                 k: Union[int, Iterable[int]],
                 dddq_max = np.array([50, 50, 50, 80]),
                 dddq_min = np.array([-50, -50, -50, -80])
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


        self.dddq_max = dddq_max
        self.dddq_min = dddq_min
        self.k = k if isinstance(k, Iterable) else [k]


        constraint_name = "jerk_constraint"
        
        # Initialize parent class
        super().__init__(constraint_name)
 



    def build(self, sym_pin_model: sym_pin.Model, sym_pin_data: sym_pin.Data,
              q_ids_in_model: np.ndarray, v_ids_in_model: np.ndarray,
              ctraj: SymbolicTrajectory, time_steps: np.ndarray) -> ConstraintSet:

        cons = ConstraintSet()
        
        for k in self.k:
            step_name = f"dddq_constraint_{k}"

            cdddq = ctraj.cdddq[k,:].T
            
            for i in range(len(self.dddq_min)):
                if i == 1:
                    continue
                
                cons += inequalityConstraint(expression=self.dddq_min[i] - cdddq[i], upper_bound=0, source_name=step_name + f"_min_joint_{i}", source_object=self)
                cons += inequalityConstraint(expression=cdddq[i] - self.dddq_max[i], upper_bound=0, source_name=step_name + f"_max_joint_{i}", source_object=self)

            
        return cons
    
