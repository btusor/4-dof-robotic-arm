"""Find the red Smarties with the camera and touch them with the stylus.

Usage:
    python main.py                                 # full run: robot + camera
    python main.py --image Data/snapshot_2red.png  # offline: image steps only
"""
import argparse
import os
import time
import cv2

from camera import open_cam, capture_frame
from thresholding import threshold_red
from circle_fit import fit_circles
from camera_coords import circles_to_xyz
from plot3d import plot_camera_xyz
import numpy as np
import kinematics as kin

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Data")     
DEBUG_DIR = os.path.join(HERE, "Debug")   

# Parameters
HOME_POSE = [0, 0, 0, 0]
RECORDING_POSE = [0, -30, -30, -30]
HOVER_MM = 30.0        
SMARTIE_TOP_Z_MM = 6.0
PITCHES_DEG = [-90, -85, -80, -75, -70, -65, -60, -55, -50, -45, -40]  # stylus pitch candidates 


def make_run_dir():
    path = os.path.join(DEBUG_DIR, time.strftime("run_%Y%m%d_%H%M%S"))
    os.makedirs(path, exist_ok=True)
    return path


def process_image(img, run_dir):
    """Image part of the recording phase. Returns the red Smarties found."""
    cv2.imwrite(os.path.join(run_dir, "01_capture.png"), img)

    # Red colour filtering + thresholding & detection -> n white blobs
    mask, spots = threshold_red(img, debug_dir=run_dir)
    print(f"Red Smarties found: {len(spots)}")
    for i, s in enumerate(spots):
        print(f"  #{i+1}: center=({s['center'][0]:.0f}, {s['center'][1]:.0f}) px, area={s['area']:.0f} px")

    # Smallest enclosing circle of each blob -> (u, v, r)
    circles = fit_circles(mask, img, debug_dir=run_dir)
    for i, (u, v, r) in enumerate(circles):
        print(f"  circle #{i+1}: center=({u:.1f}, {v:.1f}) px, r={r:.2f} px")

    # (u, v, r) -> (x, y, z) in the camera frame, depth from the Smartie diameter
    points_cam = circles_to_xyz(circles, img, debug_dir=run_dir)
    for i, p in enumerate(points_cam):
        print(f"  Smartie #{i+1} (camera frame): x={p[0]:.1f}, y={p[1]:.1f}, z={p[2]:.1f} mm")
    plot_camera_xyz(points_cam, run_dir)

    # Camera frame -> base frame with FK of the camera at the recording pose
    points_base = [kin.camera_ray_to_plane(p, RECORDING_POSE, SMARTIE_TOP_Z_MM) for p in points_cam]
    for i, p in enumerate(points_base):
        print(f"  Smartie #{i+1} (base frame): x={p[0]:.1f}, y={p[1]:.1f}, z={p[2]:.1f} mm")
    return points_base


def touch_targets(p_base):
    """Joint angles (hover, touch) that put the STYLUS tip over / on the point.

    The camera sits beside the stylus, so the point is not reached by the
    camera frame but by the stylus tip (frame {4}), pointing straight down.
    """
    p = np.array(p_base, dtype=float)
    # Straight down first (pitch -90). J4 cannot go below about -30 deg, so
    # the stylus is tilted step by step until the pose is inside the joint limits.
    for pitch in PITCHES_DEG:
        try:
            q_touch = kin.inverse_limited(p, pitch)
        except ValueError:
            continue
        for hover in (HOVER_MM, HOVER_MM / 2, 10.0):   # lower the hover if it is out of reach
            try:
                return kin.inverse_limited(p + [0, 0, hover], pitch), q_touch
            except ValueError:
                pass
    raise ValueError(f"Target {np.round(p, 1)} is out of reach inside the joint limits")


def run_robot(run_dir):
    from recording import MyRobot  # imported here so --image runs without the SDK

    cap = open_cam()
    robot = MyRobot()
    if not robot.init_status:
        cap.release()
        return
    try:
        print("Moving to base home position")
        robot.move_j(*HOME_POSE)

        # Recording phase
        print("Moving to recording position")
        robot.move_j(*RECORDING_POSE)
        img = capture_frame(cap)
        os.makedirs(DATA_DIR, exist_ok=True)
        capture_path = os.path.join(DATA_DIR, time.strftime("capture_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(capture_path, img)
        print("Saved image:", capture_path)
        points_base = process_image(img, run_dir)

        print("Returning to base home position")
        robot.move_j(*HOME_POSE)

        n = len(points_base)
        if not 1 <= n <= 3:
            print(f"Warning: expected 1-3 red Smarties, found {n}")

        # Closest to the base first: the far ones are at the edge of the workspace
        for i, p in enumerate(sorted(points_base, key=lambda p: np.hypot(p[0], p[1]))):
            print(f"Touching Smartie #{i+1} at {np.round(p, 1)} mm")
            try:
                q_hover, q_touch = touch_targets(p)
                print("  hover:", np.round(q_hover, 1), "touch:", np.round(q_touch, 1))
                if robot.move_j(*q_hover) and robot.move_j(*q_touch):
                    time.sleep(0.5)
                    robot.move_j(*q_hover)
                else:
                    print("  could not reach the target, skipping")
            except ValueError as e:   # unreachable or outside the joint limits
                print(f"  skipped: {e}")
            robot.move_j(*HOME_POSE)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        robot.disable_motors()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="process this image instead of using the robot and camera")
    args = parser.parse_args()

    run_dir = make_run_dir()
    print("Debug images ->", run_dir)
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            raise SystemExit(f"Could not read {args.image}")
        for p in process_image(img, run_dir):
            try:
                q_hover, q_touch = touch_targets(p)
                print("  touch joints:", np.round(q_touch, 1))
            except ValueError as e:
                print("  unreachable:", e)
    else:
        run_robot(run_dir)
