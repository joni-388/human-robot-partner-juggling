"""
Author:

region.py

Utility functions for geometric region operations in the partner juggling context.

This module provides functions to:
- Check if a point lies inside or on the boundary of an ellipse.
- Project a point to the nearest point on an ellipse or a circle.
- Clip or project 3D points to a region defined by an ellipse with a central circular exclusion zone and optional bounds.
- Project a grid of points to such a region.
- Visualize the region and projected points in 3D.

Typical use cases include workspace validation, trajectory planning, and visualization for robot juggling tasks.

Functions:
    is_point_in_ellipse(x, y, center_x, center_y, a, b)
    project_to_ellipse(x, y, ellipse_center_x, ellipse_center_y, a, b)
    project_to_circle(x, y, circle_center_x, circle_center_y, circle_r)
    point_clipped_to_region(point, ellipse_center_x, ellipse_center_y, a, b, circle_r, circle_center_x, circle_center_y, x_bounds, y_bounds)
    project_grid_to_region(grid, ellipse_center_x, ellipse_center_y, a, b, circle_r, circle_center_x, circle_center_y)
    plot_projected_grid(ellipse_center_x, ellipse_center_y, a, b, circle_r, z_plane, num_points)

Example:
    Run this file directly to visualize the region and test the clipping/projection functions.
    
"""

import numpy as np
import os


def is_point_in_ellipse(x, y, center_x, center_y, a, b):
    """
    Determines whether a point (x, y) lies inside or on the boundary of an ellipse.

    The ellipse is defined by its center coordinates (center_x, center_y) and its
    semi-major axis length `a` (along the x-direction) and semi-minor axis length `b` (along the y-direction).

    Args:
        x (float or np.ndarray): X-coordinate(s) of the point(s) to check.
        y (float or np.ndarray): Y-coordinate(s) of the point(s) to check.
        center_x (float): X-coordinate of the ellipse center.
        center_y (float): Y-coordinate of the ellipse center.
        a (float): Semi-major axis length of the ellipse (x-direction).
        b (float): Semi-minor axis length of the ellipse (y-direction).

    Returns:
        bool or np.ndarray: True if the point is inside or on the ellipse, False otherwise.
                            If x and y are arrays, returns a boolean array of the same shape.
    """
    return ((x - center_x) / a) ** 2 + ((y - center_y) / b) ** 2 <= 1


def project_to_ellipse(x, y, ellipse_center_x, ellipse_center_y, a, b):
    """
    Projects a point to the nearest point on an ellipse.

    Args:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        ellipse_center_x (float): X-coordinate of the ellipse center.
        ellipse_center_y (float): Y-coordinate of the ellipse center.
        a (float): Semi-major axis (x-direction).
        b (float): Semi-minor axis (y-direction).

    Returns:
        tuple: Projected point (xe, ye) on the ellipse.
    """
    dx = x - ellipse_center_x
    dy = y - ellipse_center_y
    if dx == 0 and dy == 0:
        # Arbitrary direction
        dx = 1e-6
    theta = np.arctan2(dy / b, dx / a)
    xe = ellipse_center_x + a * np.cos(theta)
    ye = ellipse_center_y + b * np.sin(theta)
    return xe, ye


def project_to_circle(x, y, circle_center_x, circle_center_y, circle_r):
    """
    Projects a point to the nearest point on the circumference of a circle.

    Args:
        x (float): X-coordinate of the point to project.
        y (float): Y-coordinate of the point to project.
        circle_center_x (float): X-coordinate of the circle center.
        circle_center_y (float): Y-coordinate of the circle center.
        circle_r (float): Radius of the circle.

    Returns:
        tuple: A tuple (xc, yc) representing the coordinates of the nearest point 
               on the circle's circumference.
    """
    dx = x - circle_center_x
    dy = y - circle_center_y
    norm = np.hypot(dx, dy)
        
    if norm == 0:
        # If point is exactly at center, pick arbitrary vector
        dx, dy = 1e-6, 1e-6
        norm = np.hypot(dx, dy)
    
    xc = circle_center_x + circle_r * dx / norm
    yc = circle_center_y + circle_r * dy / norm
    return xc, yc


def point_clipped_to_region(point, ellipse_center_x, ellipse_center_y, a, b, circle_r, circle_center_x, circle_center_y, x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)):
    """
    Clips or projects a 3D point to a valid region defined by an ellipse with a central circular exclusion zone and optional x/y bounds.

    The function enforces the following constraints:
      - The point must be inside the ellipse defined by (ellipse_center_x, ellipse_center_y, a, b).
      - The point must be outside the circle defined by (circle_center_x, circle_center_y, circle_r).
      - The point's x and y coordinates must be within the specified x_bounds and y_bounds.

    If the point is outside the ellipse, it is projected onto the ellipse boundary.
    If the point is inside the exclusion circle, it is projected onto the circle boundary (in the positive quadrant).
    After each projection, x and y are clipped to the specified bounds.
    The z-coordinate is preserved.

    Args:
        point (array-like): 3D point [x, y, z] to be clipped or projected.
        ellipse_center_x (float): X-coordinate of the ellipse center.
        ellipse_center_y (float): Y-coordinate of the ellipse center.
        a (float): Semi-major axis length of the ellipse (x-direction).
        b (float): Semi-minor axis length of the ellipse (y-direction).
        circle_r (float): Radius of the exclusion circle.
        circle_center_x (float): X-coordinate of the circle center.
        circle_center_y (float): Y-coordinate of the circle center.
        x_bounds (tuple, optional): (min_x, max_x) bounds for x. Defaults to (-np.inf, np.inf).
        y_bounds (tuple, optional): (min_y, max_y) bounds for y. Defaults to (-np.inf, np.inf).

    Returns:
        np.ndarray: The clipped or projected 3D point [x, y, z] within the valid region.
    """
    x, y, z = point[0], point[1], point[2]
    # Check in ellipse
    # in_ellipse = ((x - ellipse_center_x) / a) ** 2 + ((y - ellipse_center_y) / b) ** 2 <= 1
    in_ellipse = is_point_in_ellipse(x, y, ellipse_center_x, ellipse_center_y, a, b)    
    if not in_ellipse:
        x, y = project_to_ellipse(x, y, ellipse_center_x, ellipse_center_y, a, b)
    # clip x to bounds
    x = np.clip(x, x_bounds[0], x_bounds[1])
    y = np.clip(y, y_bounds[0], y_bounds[1])  
    # check in circle
    in_circle = (x - circle_center_x) ** 2 + (y - circle_center_y) ** 2 < circle_r ** 2
    if in_circle:
        # clip to 4th qudrant of circle
        dx = x - circle_center_x
        dy = y - circle_center_y
        if dx < 0:
            x = circle_center_x + 1e-6
        if dy > 0:
            y = circle_center_y -1e-6
        # Project to circle
        x, y = project_to_circle(x, y, circle_center_x, circle_center_y, circle_r)
        # in case bounds are in circle, clip x to bounds again
        x = np.clip(x, x_bounds[0], x_bounds[1])
        y = np.clip(y, y_bounds[0], y_bounds[1])
    
    return np.array([x, y, z])



def points_in_region(points, ellipse_center_x, ellipse_center_y, a, b,
                     circle_r, circle_center_x, circle_center_y,
                     x_bounds=(-np.inf, np.inf), y_bounds=(-np.inf, np.inf)):
    """
    Filters a set of 3D points, returning only those inside the valid region
    defined by an ellipse with a central circular exclusion zone and optional x/y bounds.

    A point is considered valid if:
      - It lies inside the ellipse defined by (ellipse_center_x, ellipse_center_y, a, b).
      - It lies outside the circle defined by (circle_center_x, circle_center_y, circle_r).
      - Its x and y coordinates are within x_bounds and y_bounds.

    Args:
        points (array-like): Nx3 array of 3D points [[x1, y1, z1], ...].
        ellipse_center_x (float): X-coordinate of the ellipse center.
        ellipse_center_y (float): Y-coordinate of the ellipse center.
        a (float): Semi-major axis length of the ellipse (x-direction).
        b (float): Semi-minor axis length of the ellipse (y-direction).
        circle_r (float): Radius of the exclusion circle.
        circle_center_x (float): X-coordinate of the circle center.
        circle_center_y (float): Y-coordinate of the circle center.
        x_bounds (tuple, optional): (min_x, max_x) bounds for x. Defaults to (-np.inf, np.inf).
        y_bounds (tuple, optional): (min_y, max_y) bounds for y. Defaults to (-np.inf, np.inf).

    Returns:
        np.ndarray: Subset of points that lie within the valid region.
    """
    points = np.asarray(points)
    x, y, _ = points[:, 0], points[:, 1], points[:, 2]

    # Ellipse condition: ((x - xc)/a)^2 + ((y - yc)/b)^2 <= 1
    in_ellipse = ((x - ellipse_center_x) / a) ** 2 + ((y - ellipse_center_y) / b) ** 2 <= 1

    # Circle exclusion: (x - xc)^2 + (y - yc)^2 >= r^2
    outside_circle = (x - circle_center_x) ** 2 + (y - circle_center_y) ** 2 >= circle_r ** 2

    # Bounds
    in_x_bounds = (x >= x_bounds[0]) & (x <= x_bounds[1])
    in_y_bounds = (y >= y_bounds[0]) & (y <= y_bounds[1])

    # Combine conditions
    mask = in_ellipse & outside_circle & in_x_bounds & in_y_bounds

    return points[mask]

def point_in_region(point, ellipse_center_x, ellipse_center_y, a, b,
                     circle_r, circle_center_x, circle_center_y,
                     x_bounds=(-float("inf"), float("inf")),
                     y_bounds=(-float("inf"), float("inf"))):
    """
    Check whether a single 3D point lies inside an ellipse (with an excluded circle
    at the center) and within optional x/y bounds.

    Args:
        point (array-like): length-3 sequence (x, y, z).
        ellipse_center_x, ellipse_center_y, a, b: ellipse parameters (a,b > 0).
        circle_r: exclusion circle radius (>= 0).
        circle_center_x, circle_center_y: circle center.
        x_bounds, y_bounds: (min, max) tuples.

    Returns:
        bool: True if the point is inside the ellipse, outside the circle, and
              inside the x/y bounds.
    """
    x, y, _ = point[0], point[1], point[2]

    if a <= 0 or b <= 0:
        raise ValueError("Ellipse semi-axes 'a' and 'b' must be positive.")
    if circle_r < 0:
        raise ValueError("circle_r must be non-negative.")

    # Ellipse condition
    in_ellipse = ((x - ellipse_center_x) / a) ** 2 + ((y - ellipse_center_y) / b) ** 2 <= 1.0

    # Circle exclusion
    outside_circle = (x - circle_center_x) ** 2 + (y - circle_center_y) ** 2 >= circle_r ** 2

    # Bounds (use Python logical and for scalars)
    in_x_bounds = (x >= x_bounds[0]) and (x <= x_bounds[1])
    in_y_bounds = (y >= y_bounds[0]) and (y <= y_bounds[1])

    return in_ellipse and outside_circle and in_x_bounds and in_y_bounds


def project_grid_to_region(grid, ellipse_center_x, ellipse_center_y, a, b, circle_r, circle_center_x, circle_center_y, x_bounds, y_bounds):
    """
    Projects a grid of 3D points onto a region defined by an ellipse (in the XY-plane) 
    with a central circular exclusion zone.

    Each point in the input grid is clipped or projected such that it lies within the region:
      - inside the ellipse,
      - outside the central circle,
      - and within the specified x-bounds (see point_clipped_to_region).

    Args:
        grid (np.ndarray): Array of shape (N, 3) representing N 3D points to be projected.
        ellipse_center_x (float): X-coordinate of the ellipse center.
        ellipse_center_y (float): Y-coordinate of the ellipse center.
        a (float): Semi-major axis length of the ellipse (x-direction).
        b (float): Semi-minor axis length of the ellipse (y-direction).
        circle_r (float): Radius of the exclusion circle.
        circle_center_x (float): X-coordinate of the circle center.
        circle_center_y (float): Y-coordinate of the circle center.
        x_bounds (tuple): Tuple (min_x, max_x) defining the x-coordinate bounds.
        y_bounds (tuple): Tuple (min_y, max_y) defining the y-coordinate bounds.

    Returns:
        np.ndarray: Array of shape (N, 3) containing the projected points within the region.
    """
    projected_points = []
    for point in grid:
        clipped_point = point_clipped_to_region(point, ellipse_center_x, ellipse_center_y, a, b, circle_r, circle_center_x, circle_center_y, x_bounds=x_bounds,y_bounds=y_bounds) #x_bounds, y_bounds)
        projected_points.append(clipped_point)
    return np.array(projected_points)


def _plot_projected_grid(ellipse_center_x, ellipse_center_y, a, b, circle_r, z_plane, x_bounds, y_bounds, num_points = 100):
    """ 
    Helper function to visualize the projected points in a 3D grid. 
    """    
    import matplotlib.pyplot as plt
    x_min = -1.0
    x_max = 1.0
    y_min = -1.0
    y_max = 1.0
    
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_plane)
    grid = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    

    
    projected_points = project_grid_to_region(grid, ellipse_center_x, ellipse_center_y, a, b, circle_r, circle_center_x=0, circle_center_y=0, x_bounds=x_bounds, y_bounds=y_bounds)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(projected_points[:, 0], projected_points[:, 1], zs=0, c='r', marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Projected Points in Region')
    ax.set_aspect('equal')
    plt.show()
    
    return projected_points
   
def _plot_filtered_grid(ellipse_center_x, ellipse_center_y, a, b, circle_r, z_plane, x_bounds, y_bounds, num_points = 100, show=True):
    """ 
    Helper function to visualize the filtered points in a 3D grid. 
    """    
    import matplotlib.pyplot as plt
    x_min = -1.0
    x_max = 1.0
    y_min = -1.0
    y_max = 1.0
    
    x_min = 0.0
    x_max = 0.85
    y_min = -0.8
    y_max = 0.01

    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_plane)
    grid = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    filtered_points = points_in_region(grid, ellipse_center_x, ellipse_center_y, a, b, circle_r, circle_center_x=0, circle_center_y=0, x_bounds=x_bounds, y_bounds=y_bounds)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(filtered_points[:, 0], filtered_points[:, 1], zs=0, c='r', marker='o', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Filtered Points in Region')
    ax.set_aspect('equal')
    if show:
        plt.show()
    
    return filtered_points

def filtered_grid(ellipse_center_x, ellipse_center_y, a, b, circle_r, z_plane, x_bounds, y_bounds, num_points=100):
        """
        Generates a grid of 3D points in the specified x/y bounds and returns only those
        inside the valid region (ellipse with central circle exclusion and bounds).

        Args:
            ellipse_center_x (float): X-coordinate of the ellipse center.
            ellipse_center_y (float): Y-coordinate of the ellipse center.
            a (float): Semi-major axis length of the ellipse (x-direction).
            b (float): Semi-minor axis length of the ellipse (y-direction).
            circle_r (float): Radius of the exclusion circle.
            z_plane (float): Z-coordinate for all points in the grid.
            x_bounds (tuple): (min_x, max_x) bounds for x.
            y_bounds (tuple): (min_y, max_y) bounds for y.
            num_points (int): Number of points along each axis.

        Returns:
            np.ndarray: Filtered points within the region, shape (N, 3).
        """
        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        x = np.linspace(x_min, x_max, num_points)
        y = np.linspace(y_min, y_max, num_points)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z_plane)
        grid = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

        filtered_points = points_in_region(
            grid,
            ellipse_center_x, ellipse_center_y, a, b,
            circle_r, circle_center_x=0, circle_center_y=0,
            x_bounds=x_bounds, y_bounds=y_bounds
        )
        return filtered_points


def _calc_area_workspace(ellipse_center_x, ellipse_center_y, a, b,
                     circle_r, circle_center_x, circle_center_y,
                     x_bounds=(-float("inf"), float("inf")),
                     y_bounds=(-float("inf"), float("inf")), samples=10000, exclude_arm_box=True):
    """
    Estimate the area of the valid workspace region using Monte Carlo sampling.

    Args:
        ellipse_center_x, ellipse_center_y, a, b: ellipse parameters.
        circle_r, circle_center_x, circle_center_y: exclusion circle parameters.
        x_bounds, y_bounds: bounds for sampling.
        samples: number of random samples.

    Returns:
        float: estimated area of the region.
    """
    # If bounds are infinite, use default bounds for sampling
    if not np.isfinite(x_bounds[0]) or not np.isfinite(x_bounds[1]):
        x_min, x_max = -0.1, 1.0
    else:
        x_min, x_max = x_bounds
    if not np.isfinite(y_bounds[0]) or not np.isfinite(y_bounds[1]):
        y_min, y_max = -1.0, 0.1
    else:
        y_min, y_max = y_bounds
        
    x_bounds = (x_min, x_max)
    y_bounds = (y_min, y_max)
    
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    if exclude_arm_box:
        box_x_low  = 0
        box_x_high =  0.325
        box_y_low  =  -0.6
        box_y_high =  0.0
    else:
        inside_box = False

    # Sample uniformly in the bounding rectangle
    xs = np.random.uniform(x_min, x_max, samples)
    ys = np.random.uniform(y_min, y_max, samples)
    zs = np.zeros_like(xs)
    points = np.stack([xs, ys, zs], axis=1)

    count_in = 0
    for p in points:
        if exclude_arm_box:
            inside_box = (
                box_x_low <= p[0] <= box_x_high and
                box_y_low <= p[1] <= box_y_high
            )        
        if point_in_region(
            p, ellipse_center_x, ellipse_center_y, a, b,
            circle_r, circle_center_x, circle_center_y,
            x_bounds, y_bounds
        ) and not inside_box:
            count_in += 1

    rect_area = (x_max - x_min) * (y_max - y_min)
    area = rect_area * (count_in / samples)
    return area
    




if __name__ == "__main__":
    
    center_x = 6.822320486321587e-14
    center_y = -0.3865181854630024
    a = 0.8725810784022197
    b = 0.5070394009593238
        
    circle_r = 0.4
    circle_center_x = 0
    circle_center_y = 0

    z_plane = 0.57  # z-coordinate for the plane and ellipse
    num_points = 50  # Number of points to generate for the ellipse

    # Define the bounds for the x and y coordinates
    x_bounds = (0, 0.8)
    y_bounds = (-0.8,0.01)

    area = _calc_area_workspace(ellipse_center_x=center_x, ellipse_center_y=center_y, a=a, b=b,
                     circle_r=circle_r, circle_center_x=circle_center_x, circle_center_y=circle_center_y,
                     x_bounds=x_bounds, y_bounds=y_bounds)
    
    print(f"area workspace: {area}")
    exit()

    p = np.array([-1,-1,0])
    # p = np.array([0.5,-0.5,0])
    
    # a= point_in_region(point=p, ellipse_center_x=center_x, ellipse_center_y=center_y, a=a, b=b, circle_r=circle_r, circle_center_x=circle_center_x,circle_center_y=circle_center_y, x_bounds=x_bounds, y_bounds=y_bounds,)

    # print(a)
    projected_points= _plot_projected_grid(center_x, center_y, a, b, circle_r, z_plane, x_bounds, y_bounds, num_points = 100)
    import matplotlib.pyplot as plt

    x_min = -1.0
    x_max = 1.0
    y_min = -1.0
    y_max = 1.0
    
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_plane)
    grid = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T
    projected_points = grid
    
    inside_points = []
    outside_points = []
    for p in projected_points:
        ok = point_in_region(point=p,
                             ellipse_center_x=center_x,
                             ellipse_center_y=center_y,
                             a=a, b=b,
                             circle_r=circle_r,
                             circle_center_x=circle_center_x,
                             circle_center_y=circle_center_y,
                             x_bounds=x_bounds,
                             y_bounds=y_bounds)
        if ok:
            inside_points.append(p)
        else:
            outside_points.append(p)

    inside_points = np.array(inside_points)
    outside_points = np.array(outside_points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if inside_points.size:
        ax.scatter(inside_points[:, 0], inside_points[:, 1], inside_points[:, 2],
                   c='g', marker='o', s=2, label='inside (green)')
    if outside_points.size:
        ax.scatter(outside_points[:, 0], outside_points[:, 1], outside_points[:, 2],
                   c='r', marker='o', s=2, label='outside (red)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Projected Points: green=inside, red=outside')
    ax.legend()
    plt.show()
    
    # plot projected grid
    # _plot_projected_grid(center_x, center_y, a, b, circle_r=0.4, z_plane=z_plane, x_bounds=x_bounds, y_bounds=y_bounds, num_points = 100)
    
    filtered_points = _plot_filtered_grid(center_x, center_y, a, b, circle_r=0.4, z_plane=z_plane, x_bounds=x_bounds, y_bounds=y_bounds, num_points = 10)
    
    print(len(filtered_points))
    
    # 1. Path to the target directory (relative to this file)
    this_file = os.path.abspath(__file__)
    this_dir = os.path.dirname(this_file)
    target_dir = os.path.abspath(os.path.join(this_dir, '..', '..', '..', 'data'))

    # Save filtered points to file in the target directory
    output_path = os.path.join(target_dir, "test_points_in_region.npy")
    np.save(output_path, filtered_points)
    print(f"Filtered points saved to: {output_path}")

    # print("Target directory:", target_dir)
    # print("This file:", this_file)
