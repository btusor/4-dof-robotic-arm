"""Camera helpers: open the USB camera and grab a fresh, settled frame."""
import time
import cv2

CAM_INDEX = 1
WIDTH, HEIGHT = 1280, 720


def open_cam(index=CAM_INDEX, width=WIDTH, height=HEIGHT):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # fixed focus as during calibration, keeps K valid
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    return cap


def capture_frame(cap, settle_s=1.0, timeout_s=5.0, show=True):
    """Read frames for settle_s seconds and return the last one.
    """
    frame = None
    t_start = time.time()
    
    while time.time() - t_start < settle_s or frame is None:
        if time.time() - t_start > timeout_s:
            raise RuntimeError("Camera returned no frame")
        ok, f = cap.read()
        if not ok:
            continue
        frame = f
        if show:
            cv2.imshow("camera", frame)
            cv2.waitKey(1)

    h, w = frame.shape[:2]
    if (w, h) != (WIDTH, HEIGHT):
        print(f"Warning: frame is {w}x{h}, calibration is {WIDTH}x{HEIGHT}")
    return frame
