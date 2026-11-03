"""Camera intrinsic calibration with a checkerboard.

1. Print a checkerboard (e.g. https://markhedleyjones.com/projects/calibration-checkerboard-collection)
   and glue it to something flat. Measure one square with a ruler.
2. Set BOARD (number of INNER corners) and SQUARE_MM below.
3. Run:  python calibrate_camera.py
   - Hold the board in view; corners are drawn when detected.
   - Press SPACE to capture a view (take 15-25 from different angles,
     distances and positions, including near the image corners).
   - Press c to calibrate and save, q to quit.

Result is saved to camera_calib.npz (K, dist, width, height) and printed.
Intrinsics are only valid for the resolution used here, so calibrate at
the same resolution you capture with in your robot script.
"""
import os
import numpy as np
import cv2

CAM_INDEX = 1
WIDTH, HEIGHT = 1280, 720          # must match the resolution you use later
BOARD = (6, 4)                     # inner corners (columns, rows)
SQUARE_MM = 35.0                   # printed square size in mm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calib.npz")

# 3D corner coordinates in the board frame (z = 0)
objp = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * SQUARE_MM

obj_points, img_points = [], []
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # fixed focus keeps intrinsics constant (if supported)

size = None
while True:
    ok, frame = cap.read()
    if not ok:
        print("No frame from camera")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    found, corners = cv2.findChessboardCorners(
        gray, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)

    view = frame.copy()
    if found:
        cv2.drawChessboardCorners(view, BOARD, corners, found)
    cv2.putText(view, f"captured: {len(obj_points)}  {size[0]}x{size[1]}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
    cv2.imshow("calibration", view)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" ") and found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners)
        print(f"Captured view {len(obj_points)}")
    if key == ord("c"):
        if len(obj_points) < 10:
            print("Capture at least 10 views first")
            continue
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        print(f"\nRMS reprojection error: {rms:.3f} px  (good: < 0.5, ok: < 1)")
        print("Camera matrix K:\n", np.round(K, 2))
        print(f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        print("Distortion (k1 k2 p1 p2 k3):", np.round(dist.ravel(), 4))
        np.savez(OUT, K=K, dist=dist, width=size[0], height=size[1], rms=rms)
        print("Saved", OUT)
        break

cap.release()
cv2.destroyAllWindows()
