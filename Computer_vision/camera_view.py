"""Live view from a USB camera.

Usage:  python camera_view.py [index]
Keys:   q = quit, s = save snapshot, n = switch to next camera
"""
import sys
import cv2

index = 1


def open_cam(i):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow: fast to open on Windows
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


cap = open_cam(index)
shot = 0
while True:
    ok, frame = cap.read()
    if not ok:
        print(f"Camera {index} gave no frame")
        break
    cv2.putText(frame, f"camera {index}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("camera", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        name = f"snapshot_{index}_{shot}.png"
        cv2.imwrite(name, frame)
        print("saved", name)
        shot += 1
    if key == ord("n"):
        cap.release()
        index = (index + 1) % 2
        cap = open_cam(index)

cap.release()
cv2.destroyAllWindows()
