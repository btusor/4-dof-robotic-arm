"""Forward / inverse kinematics of the 4 DoF arm (units: mm, degrees).

Two joint conventions are in use:
    robot (move_j) : q = [J1, J2, J3, J4], all 0 when the arm points straight up
    DH             : theta = [q1, q2 + 90, q3, q4]

Frame {4} is the stylus tip. Frame {5} is the camera mount, offset from {4} by (-15, 45, 0) mm, so the camera and the stylus are NOT at the same place.
"""
import numpy as np

D1 = 50.0
A2 = 93.0
A3 = 93.0
A4 = 50.0
THETA2_OFFSET_DEG = 90.0
# Measured with hand-guided poses (read_pose.py): with the stylus pointing straight
# down, J4 reads about +30..+37 deg while J2 + 90 + J3 is about -45, so the J4 zero is
# turned by about -90 deg relative to the DH angle (theta_4 = q4 - 90).
THETA4_OFFSET_DEG = -90.0

# Joint ranges [deg] that servo_limit_test.py drove through without a stall.
# J4 stops mechanically at about -34 deg on the negative side (stalled at -40).
JOINT_LIMITS_DEG = [(-90, 90), (-100, 20), (-120, 120), (-30, 100)]
LIMIT_MARGIN_DEG = 2.0

# The assignment gives 45 mm along y_4. Fitted to Smarties touched by hand (5 points,
# rms error 4.8 mm) the effective offset is 30 mm.
CAMERA_Y_OFFSET_MM = 30.0
T_4_5 = np.array([[1, 0, 0, -15],
                  [0, 1, 0, CAMERA_Y_OFFSET_MM],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)

# Rotation from the OpenCV camera frame (x right, y down, z out of the lens) to
# frame {5}. The lens looks along x_5 (out of the camera, as in the DH). At
# RECORDING_POSE the sum theta_2 + theta_3 + theta_4 is -90 deg, so x_5 points down.
# Fitted to Smarties touched by hand: image-right = -z_5 (= +y of the base) and
# image-down = -y_5 (= -x of the base, i.e. towards the robot). This is a MIRRORED
# (left-handed) set of axes: the camera image is flipped vertically, so the matrix
# has determinant -1. Only directions are used, so that is fine.
_X_CAM, _Y_CAM, _Z_CAM = np.array([0, 0, -1]), np.array([0, -1, 0]), np.array([1, 0, 0])
R_5_CAM = np.column_stack([_X_CAM, _Y_CAM, _Z_CAM]).astype(float)


def q_to_theta(q_deg):
    """Robot joint angles [deg] -> DH angles [rad]."""
    q = np.asarray(q_deg, dtype=float).copy()
    q[1] += THETA2_OFFSET_DEG
    q[3] += THETA4_OFFSET_DEG
    return np.radians(q)


def theta_to_q(theta_rad):
    """DH angles [rad] -> robot joint angles [deg]."""
    q = np.degrees(np.asarray(theta_rad, dtype=float))
    q[1] -= THETA2_OFFSET_DEG
    q[3] -= THETA4_OFFSET_DEG
    return q


def dh(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0, sa, ca, d],
                     [0, 0, 0, 1]])


def forward_stylus(q_deg):
    """T_4^0 (stylus tip frame in the base frame) for robot joint angles [deg]."""
    t1, t2, t3, t4 = q_to_theta(q_deg)
    return (dh(t1, D1, 0, np.pi / 2) @ dh(t2, 0, A2, 0)
            @ dh(t3, 0, A3, 0) @ dh(t4, 0, A4, 0))


def forward_camera(q_deg):
    """T_5^0 (camera mount frame in the base frame)."""
    return forward_stylus(q_deg) @ T_4_5


def camera_to_base(p_cam, q_deg):
    """Point [x, y, z] in the OpenCV camera frame -> base frame, arm at q_deg."""
    T_5_0 = forward_camera(q_deg)
    p5 = R_5_CAM @ np.asarray(p_cam, dtype=float)
    return (T_5_0 @ np.append(p5, 1.0))[:3]


def camera_ray_to_plane(p_cam, q_deg, z_plane):
    """Base-frame point where the camera ray through p_cam meets the plane z = z_plane.

    Only the viewing direction of p_cam is used (its depth is ignored), which
    is more robust than a depth estimated from the Smartie's apparent size
    when the Smartie lies on the table.
    """
    T = forward_camera(q_deg)
    d = T[:3, :3] @ R_5_CAM @ (np.asarray(p_cam, dtype=float) / p_cam[2])
    if abs(d[2]) < 1e-9:
        raise ValueError("Camera ray is parallel to the table")
    t = (z_plane - T[2, 3]) / d[2]
    if t <= 0:
        raise ValueError("Table plane is behind the camera")
    return T[:3, 3] + t * d


def inverse(p, x_z0=-1.0, elbow="up"):
    """Joint angles [deg, robot convention] that put the stylus tip at p.

    p    : tip position [x, y, z] in the base frame
    x_z0 : z component of x_4 (sin of the stylus pitch); -1 = pointing straight down
    elbow: 'up' or 'down'
    Raises ValueError if p is out of reach.
    """
    ox, oy, oz = p
    theta1 = np.arctan2(oy, ox)
    theta_ee = np.arcsin(x_z0)

    r = np.hypot(ox, oy) - A4 * np.cos(theta_ee)
    s = oz - D1 - A4 * np.sin(theta_ee)

    c3 = (r ** 2 + s ** 2 - A2 ** 2 - A3 ** 2) / (2 * A2 * A3)
    if abs(c3) > 1:
        raise ValueError(f"Target {np.round(p, 1)} is out of reach")
    s3 = -np.sqrt(1 - c3 ** 2) if elbow == "up" else np.sqrt(1 - c3 ** 2)
    theta3 = np.arctan2(s3, c3)
    theta2 = np.arctan2(s, r) - np.arctan2(A3 * s3, A2 + A3 * c3)
    theta4 = theta_ee - theta2 - theta3
    return theta_to_q([theta1, theta2, theta3, theta4])


def within_limits(q_deg, margin=LIMIT_MARGIN_DEG):
    return all(lo + margin <= a <= hi - margin for a, (lo, hi) in zip(q_deg, JOINT_LIMITS_DEG))


def inverse_limited(p, pitch_deg):
    """IK for the stylus tip p with the stylus pitched pitch_deg (-90 = straight down),
    trying both elbow configurations. Returns the first solution inside JOINT_LIMITS_DEG,
    raises ValueError if there is none."""
    x_z0 = np.sin(np.radians(pitch_deg))
    for elbow in ("up", "down"):
        try:
            q = inverse(p, x_z0=x_z0, elbow=elbow)
        except ValueError:
            continue
        if within_limits(q):
            return q
    raise ValueError(f"No solution inside the joint limits for {np.round(p, 1)} at pitch {pitch_deg}")


if __name__ == "__main__":
    # Self-check: IK(FK(q)) must give back the same tip position
    for q in ([0, -30, -30, -120], [20, -40, -50, -100], [-30, -20, -60, -110]):
        T = forward_stylus(q)
        p = T[:3, 3]
        q_ik = inverse(p, x_z0=T[2, 0])
        err = np.linalg.norm(forward_stylus(q_ik)[:3, 3] - p)
        print(f"q={q} tip={np.round(p, 1)} IK={np.round(q_ik, 1)} err={err:.2e} mm")
