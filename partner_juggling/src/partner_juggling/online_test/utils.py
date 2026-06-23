import numpy as np

def make_full_q(q, pin_model, joint_names):
    q_full = np.zeros(pin_model.nq)
    joint_ids = [pin_model.getJointId(name) for name in joint_names]
    ids = []
    for joint_id in joint_ids:
        joint = pin_model.joints[int(joint_id)]
        ids.extend(range(joint.idx_q, joint.idx_q + joint.nq))

    q_ids = np.array(ids, dtype=int)
    q_full[q_ids] = q

    return q_full, q_ids

def line_box_intersection(A, B, box_min, box_max):
    # Returns intersection points (entry, exit) if any, else None
    tmin = 0.0
    tmax = 1.0
    direction = B - A
    for i in range(3):
        if abs(direction[i]) < 1e-8:
            if A[i] < box_min[i] or A[i] > box_max[i]:
                return None
        else:
            ood = 1.0 / direction[i]
            t1 = (box_min[i] - A[i]) * ood
            t2 = (box_max[i] - A[i]) * ood
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return None
    entry = A + tmin * direction
    exit = A + tmax * direction
    return entry, exit



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    def _visualize_boxes_lines_intersections(boxes, A, B):

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = ['cyan', 'orange', 'lime']
        # Plot boxes
        for i, (box_min, box_max) in enumerate(boxes):
            _plot_box(ax, box_min, box_max, color=colors[i % len(colors)], alpha=0.2)
        # Plot line
        ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color='red', linewidth=2, label='Line')
        # Plot intersection points
        for i, (box_min, box_max) in enumerate(boxes):
            result = line_box_intersection(A, B, box_min, box_max)
            if result is not None:
                entry, exit = result
                ax.scatter(*entry, color='blue', s=50, label=f'Entry {i}' if i == 0 else "")
                ax.scatter(*exit, color='green', s=50, label=f'Exit {i}' if i == 0 else "")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    
    
    def _plot_box(ax, box_min, box_max, color='cyan', alpha=0.2):
        # Draw a 3D box given min and max corners
        r = [
            [box_min[0], box_max[0]],
            [box_min[1], box_max[1]],
            [box_min[2], box_max[2]]
        ]
        # Create list of box vertices
        points = np.array([[x, y, z] for x in r[0] for y in r[1] for z in r[2]])
        # Define the 6 box faces
        faces = [
            [points[0], points[1], points[3], points[2]],
            [points[4], points[5], points[7], points[6]],
            [points[0], points[1], points[5], points[4]],
            [points[2], points[3], points[7], points[6]],
            [points[1], points[3], points[7], points[5]],
            [points[0], points[2], points[6], points[4]],
        ]
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='k', alpha=alpha))
        

    # Example usage
    # Define boxes
    box1_min = np.array([-0.15, -0.15, 0.0])
    box1_max = np.array([0.15, 0.15, 1.5])
    box2_min = np.array([-0.25, -0.25, 0.0])
    box2_max = np.array([0.25, 0.25, 0.5])
    box3_min = np.array([-0.35, -0.35, 0.0])
    box3_max = np.array([0.35, 0.35, 0.3])
    boxes = [(box1_min, box1_max), (box2_min, box2_max), (box3_min, box3_max)]
    
    # Define line segment
    A = np.array([-0.2, -0.2, 0.5])
    B = np.array([0.2, 0.2, 1.0])   
    _visualize_boxes_lines_intersections(boxes, A, B)
    