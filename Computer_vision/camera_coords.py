"""Pixel circle (u, v, r) -> 3D point (x, y, z) in the camera frame.

Depth comes from the known Smartie size: a disc of real radius R at
distance z appears with radius r = f * R / z, so z = f * R / r.
"""
import csv
import os
import cv2
import numpy as np

SMARTIE_DIAMETER_MM = 16.0
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "Camera_calibration", "camera_calib.npz")


def load_calibration(path=CALIB_FILE):
    """Return (K, dist, (width, height)) of the calibrated camera."""
    data = np.load(path)
    return data["K"], data["dist"], (int(data["width"]), int(data["height"]))


def circle_to_xyz(u, v, r, K, dist, diameter_mm=SMARTIE_DIAMETER_MM, n_samples=16):
    """Return np.array([x, y, z]) in mm for one circle given in pixels.

    The centre and n_samples points on the circle are undistorted to
    normalised image coordinates (x/z, y/z). The radius in normalised
    coordinates r_n equals R / z, which gives z without needing fx or fy
    separately, and removes the lens distortion from the radius as well.
    """
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    pts = np.vstack([[u, v], np.column_stack([u + r * np.cos(angles), v + r * np.sin(angles)])])
    norm = cv2.undistortPoints(pts.reshape(-1, 1, 2).astype(np.float64), K, dist).reshape(-1, 2)

    xn, yn = norm[0]
    r_n = np.mean(np.linalg.norm(norm[1:] - norm[0], axis=1))
    z = (diameter_mm / 2) / r_n
    return np.array([xn * z, yn * z, z])


def circles_to_xyz(circles, img, debug_dir=None, diameter_mm=SMARTIE_DIAMETER_MM):
    """Convert every (u, v, r) circle to (x, y, z) in the camera frame [mm].

    If debug_dir is given, saves:
        07_camera_xyz.png : circles labelled with their camera-frame xyz
        camera_xyz.csv    : index, u, v, r, x, y, z of every circle
    """
    K, dist, calib_size = load_calibration()
    h, w = img.shape[:2]
    if (w, h) != calib_size:
        raise ValueError(f"Image is {w}x{h}, calibration is {calib_size[0]}x{calib_size[1]}")

    points = [circle_to_xyz(u, v, r, K, dist, diameter_mm) for u, v, r in circles]

    if debug_dir:
        out = img.copy()
        for i, ((u, v, r), p) in enumerate(zip(circles, points)):
            center = (int(round(u)), int(round(v)))
            cv2.circle(out, center, int(round(r)), (0, 255, 0), 2)
            cv2.circle(out, center, 3, (255, 0, 0), -1)
            cv2.putText(out, f"{i+1}: ({p[0]:.0f}, {p[1]:.0f}, {p[2]:.0f}) mm",
                        (center[0] + int(r) + 5, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        # Principal point = where the optical axis (x = y = 0) hits the image
        cx, cy = int(round(K[0, 2])), int(round(K[1, 2]))
        cv2.drawMarker(out, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.imwrite(os.path.join(debug_dir, "07_camera_xyz.png"), out)

        with open(os.path.join(debug_dir, "camera_xyz.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "u_px", "v_px", "r_px", "x_mm", "y_mm", "z_mm"])
            for i, ((u, v, r), p) in enumerate(zip(circles, points)):
                writer.writerow([i + 1, f"{u:.2f}", f"{v:.2f}", f"{r:.2f}",
                                 f"{p[0]:.2f}", f"{p[1]:.2f}", f"{p[2]:.2f}"])
    return points
