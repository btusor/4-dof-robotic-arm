"""3D debug plot of the Smarties in the camera frame."""
import os
import matplotlib
matplotlib.use("Agg")  # render to file only, no window
import matplotlib.pyplot as plt
import numpy as np

from camera_coords import load_calibration, SMARTIE_DIAMETER_MM


def plot_camera_xyz(points, debug_dir, diameter_mm=SMARTIE_DIAMETER_MM):
    """Save 08_camera_xyz_3d.png: camera, field of view and Smarties in 3D.

    Camera frame (OpenCV): x right, y down, z along the optical axis.
    Each Smartie is drawn as a disc of its real diameter, parallel to the
    image plane, with the ray from the camera to its centre.
    """
    K, _, (w, h) = load_calibration()
    z_far = max(p[2] for p in points) * 1.1 if points else 200.0

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(projection="3d")

    # Camera: origin + axes
    L = z_far * 0.2
    for d, c, name in zip(np.eye(3), "rgb", "xyz"):
        ax.quiver(0, 0, 0, *(d * L), color=c, arrow_length_ratio=0.15)
        ax.text(*(d * L * 1.15), name, color=c)
    ax.scatter(0, 0, 0, color="k", s=40)
    ax.text(0, 0, -L * 0.3, "camera", color="k")

    # Field of view: rays through the image corners up to z_far
    Kinv = np.linalg.inv(K)
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], float)
    far = (Kinv @ corners.T).T * z_far
    for c in far:
        ax.plot([0, c[0]], [0, c[1]], [0, c[2]], color="gray", lw=0.7)
    ax.plot(*np.vstack([far, far[:1]]).T, color="gray", lw=0.7)

    # Smarties: disc + ray + label
    t = np.linspace(0, 2 * np.pi, 60)
    R = diameter_mm / 2
    for i, p in enumerate(points):
        ax.plot([0, p[0]], [0, p[1]], [0, p[2]], "r--", lw=0.8)
        ax.plot_trisurf(np.append(p[0] + R * np.cos(t), p[0]),
                        np.append(p[1] + R * np.sin(t), p[1]),
                        np.full(len(t) + 1, p[2]), color="red", alpha=0.8)
        ax.text(p[0], p[1], p[2] + R,
                f" #{i+1} ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm", color="darkred")

    ax.set_xlabel("x [mm] (right)")
    ax.set_ylabel("y [mm] (down)")
    ax.set_zlabel("z [mm] (optical axis)")
    ax.set_title("Red Smarties in the camera frame")

    # Equal scale on all axes, camera on top looking down at the table
    lo = np.minimum(far.min(axis=0), 0)
    hi = np.maximum(far.max(axis=0), 0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.invert_zaxis()
    ax.view_init(elev=25, azim=-60)

    path = os.path.join(debug_dir, "08_camera_xyz_3d.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path
