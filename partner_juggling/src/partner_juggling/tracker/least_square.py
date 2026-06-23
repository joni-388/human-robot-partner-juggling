
import numpy as np
from partner_juggling.tracker.data_types import Detection, State
from typing import List


def _vel_from_coef(coef, t):
    if coef.size == 3:           # quadratic: a*t^2 + b*t + c -> v = 2*a*t + b
        return 2.0 * coef[0] * t + coef[1]
    if coef.size == 2:           # linear: b*t + c -> v = b
        return coef[0]
    raise ValueError("Unexpected coefficient length when computing velocity.")

def predict_with_lstsq(detections: List[Detection], approximate_velocity_measurement_flag, z_plane: float):
    """
    Mehtod that predicts the trajectory of the ball from the provided detections using lstsq method.

    Args:
        detections (list[Detection]): Detections of the ball to use for lstsq prediction.
        z_plane (float): The z-coordinate of the plane where the touchdown is expected.                                                 

    Returns:
        np.array:TochDown position of the ball in the form of [x, y, z].
        float: TouchDown time of the ball in seconds.
        tuple: Coefficients of the polynomial fit for x, y, and z coordinates.
    """
                  
    if len(detections) == 0:
        raise ValueError("No detections provided for prediction.")       
    
    if isinstance(detections[0], Detection):
        if approximate_velocity_measurement_flag:
            points = np.array([d.state_vector[[0,2,4],0] for d in detections]).reshape((-1,3))
        else:
            points = np.array([d.state_vector[[0,1,2],0] for d in detections]).reshape((-1,3))
            
    elif isinstance(detections[0], State):
        
        points = np.array([d.state_vector[[0,3,6],0] for d in detections]).reshape((-1,3))
    else:
        raise ValueError("Detections must be of type Detection or State.")
              
    ts = np.array([d.timestamp.timestamp() for d in detections])
    # Normalize timestamps to start from zero
    ts = ts - ts[0]
    
    # if there is only one detection, we cannot predict the trajectory
    if len(ts) < 3:
        raise ValueError("At least 3 detections are required to predict the trajectory using lstsq method.")
    
    # if there are more than 3 detections, we can predict the trajectory
    T = np.vstack([ts**2, ts, np.ones_like(ts)]).T 
    x_coef, _, _, _ = np.linalg.lstsq(T[:,1:3], points[:, 0], rcond=None)
    y_coef, _, _, _ = np.linalg.lstsq(T[:,1:3], points[:, 1], rcond=None)
    z_coef, _, _, _ = np.linalg.lstsq(T, points[:, 2], rcond=None)
    
    z_pos = z_coef[2]
    z_vel = z_coef[1]
    z_acc = z_coef[0]
    target_height = z_plane 
    # Calculate the touchdown time using the quadratic formula
    t_td = (-z_vel - np.sqrt( z_vel**2 - 4*z_acc*(z_pos - target_height)))/(2*z_acc)
    
    x_td = x_coef[0]*t_td + x_coef[1]
    y_td = y_coef[0]*t_td + y_coef[1]
    z_td = target_height
    
    
    
    # before the time zero means beginning of the trajectory, so we need to adjust it
    # to be relative to the last timestamp of the trajectory  
    t_td = t_td - (ts[-1] - ts[0])
    
    # evaluate velocity at touchdown (convert t_td back to the fit timebase)
    t_fit = t_td + ts[-1]  # coefficients were fitted with t starting at ts[0]=0

    x_coef = np.asarray(x_coef)
    y_coef = np.asarray(y_coef)
    z_coef = np.asarray(z_coef)

    vx_td = _vel_from_coef(x_coef, t_fit)
    vy_td = _vel_from_coef(y_coef, t_fit)
    vz_td = _vel_from_coef(z_coef, t_fit)

    vel_td = np.array([vx_td, vy_td, vz_td])
    speed_td = np.linalg.norm(vel_td)

    # print("Velocity at touchdown (m/s):", vel_td, "speed (m/s):", speed_td)
    
 
    return np.array([x_td, y_td, z_td]), vel_td, t_td , (x_coef, y_coef, z_coef)


