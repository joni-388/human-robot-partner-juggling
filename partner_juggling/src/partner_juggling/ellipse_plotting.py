import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class EllipseVisualizer:
    def __init__(self):
        # Ellipse parameters (from your setup)
        self.center_x = 0
        self.center_y = -0.3676
        self.a = 0.87142
        self.b = 0.5269
        self.z_plane = 0.57

    def is_point_in_ellipse(self, x, y, cx, cy, a, b):
        """Check whether points (x, y) are inside the ellipse."""
        return ((x - cx) ** 2) / (a ** 2) + ((y - cy) ** 2) / (b ** 2) <= 1

    def ellipse_minus_circle_plot(self, ax=None, radius=0.4, num_points=200, color='y', linewidth=2, label=None, plot_3d=True):
        """
        Plot an ellipse minus a circle on a given 3D matplotlib axis.
        If no axis is provided, one will be created.

        Args:
            ax: matplotlib 3D axis (optional)
            radius: circle radius to subtract from ellipse
            num_points: number of points for resolution
            color: line color
            linewidth: line width
            label: label for the legend
        """

        # Create an axis if none is provided
        if ax is None:
            if plot_3d:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots()   #

        # Generate ellipse points
        thetas = np.linspace(0, 2 * np.pi, num_points + 1)
        xs = self.center_x + self.a * np.cos(thetas)
        ys = self.center_y + self.b * np.sin(thetas)

        # Filter ellipse points where x > 0
        mask_x = xs > 0
        xs, ys = xs[mask_x], ys[mask_x]

        # Remove points inside the circle
        mask_outside_circle = xs**2 + ys**2 >= radius**2
        false_indices = np.where(~mask_outside_circle)[0]
        xs, ys = xs[mask_outside_circle], ys[mask_outside_circle]

        # Generate circle points
        xs_circle = radius * np.cos(thetas)
        ys_circle = radius * np.sin(thetas)

        # Check which circle points are inside the ellipse
        in_ellipse = self.is_point_in_ellipse(xs_circle, ys_circle,
                                              self.center_x, self.center_y,
                                              self.a, self.b)
        false_indices_circle = np.where(~in_ellipse)[0]
        circle_points = np.vstack((xs_circle[in_ellipse], ys_circle[in_ellipse])).T

        # Ensure proper order continuity and filter x > 0
        if len(false_indices_circle) > 0:
            circle_points = np.concatenate((
                circle_points[false_indices_circle[0]:],
                circle_points[:false_indices_circle[0]]
            ))
        circle_points = circle_points[circle_points[:, 0] > 0]
        circle_points = circle_points[::-1]  # Reverse order for correct connection

        # Combine ellipse and circle boundaries
        if len(false_indices) > 0:
            X = np.concatenate((xs[:false_indices[0]], circle_points[:, 0], xs[false_indices[0]:]))
            Y = np.concatenate((ys[:false_indices[0]], circle_points[:, 1], ys[false_indices[0]:]))
        else:
            X = np.concatenate((xs, circle_points[:, 0]))
            Y = np.concatenate((ys, circle_points[:, 1]))

        Z = np.full_like(X, self.z_plane)

        # Plot on the provided axis
        if plot_3d:
            ax.plot3D(X, Y, Z, color=color, linewidth=linewidth, label=label or "Ellipse minus Circle")
        else: 
            ax.plot(X, Y, color=color, linewidth=linewidth, label=label or "Ellipse minus Circle")

        return ax


# Example usage:
if __name__ == "__main__":
    viz = EllipseVisualizer()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Add the shape to an existing axis
    viz.ellipse_minus_circle_plot(ax=ax, radius=0.4, num_points=200, color='y', linewidth=2, label='My Ellipse')

    # Add another one for demonstration
    viz.z_plane = 0.6
    viz.ellipse_minus_circle_plot(ax=ax, radius=0.3, color='r', linewidth=1.5, label='Smaller One')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title('Ellipse minus Circle (x > 0 region)')
    ax.view_init(elev=25, azim=45)
    plt.show()
